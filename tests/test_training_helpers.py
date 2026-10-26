"""Training helpers: staged schedule, EOS accuracy, warm-start vocabulary transfer."""

import math

import pytest
import torch

from training import eos_metrics, param_groups
from training.warm_start import PretrainedTable, build_warm_start, fit_procrustes, romanize_piece


# ----- param_groups --------------------------------------------------------------------------------
def test_group_of():
    assert param_groups.group_of("flow_lm.conditioner.embed.weight") == "text_embedding"
    assert param_groups.group_of("flow_lm.flow_net.res_blocks.0.mlp.0.weight") == "heads"
    assert param_groups.group_of("flow_lm.out_eos.bias") == "heads"
    assert param_groups.group_of("flow.w_s_t.0.weight") == "heads"
    assert param_groups.group_of("flow_lm.transformer.layers.3.self_attn.in_proj.weight") == "backbone"
    assert param_groups.group_of("flow_lm.speaker_proj_weight") == "backbone"


def test_lr_schedule_stages_warmup_and_cosine_floor():
    s = param_groups.GroupSchedule(peak_lr=1e-4, start_step=400, warmup_steps=400)
    assert param_groups.lr_at_step(0, s, 5000, 0.1) == 0.0
    assert param_groups.lr_at_step(399, s, 5000, 0.1) == 0.0
    assert param_groups.lr_at_step(400, s, 5000, 0.1) == pytest.approx(1e-4 / 400)
    assert param_groups.lr_at_step(799, s, 5000, 0.1) == pytest.approx(1e-4)
    assert param_groups.lr_at_step(5000, s, 5000, 0.1) == pytest.approx(1e-5)
    mid = 800 + (5000 - 800) // 2
    assert param_groups.lr_at_step(mid, s, 5000, 0.1) == pytest.approx(1e-5 + 0.5 * 9e-5, rel=1e-3)
    const = param_groups.GroupSchedule(peak_lr=2e-4, warmup_steps=1000)
    assert param_groups.lr_at_step(4000, const, 5000, 1.0) == pytest.approx(2e-4)


def test_schedule_scales_reference_is_backbone_once_unfrozen():
    model = torch.nn.Module()
    model.flow_lm = torch.nn.Module()
    model.flow_lm.conditioner = torch.nn.Module()
    model.flow_lm.conditioner.embed = torch.nn.Embedding(5, 3)
    model.flow_lm.transformer = torch.nn.Linear(3, 3)
    model.flow_lm.out_eos = torch.nn.Linear(3, 1)
    groups = param_groups.build_param_groups(model, weight_decay=0.1)
    assert {g["name"]: g["weight_decay"] for g in groups} == {"text_embedding": 0.0, "backbone": 0.1, "heads": 0.1}
    opt = torch.optim.AdamW(groups, lr=1.0)
    sched = param_groups.parse_schedules({
        "text_embedding": {"peak_lr": 5e-4, "warmup_steps": 100},
        "backbone": {"peak_lr": 1e-4, "start_step": 400, "warmup_steps": 400},
        "heads": {"peak_lr": 1e-4, "start_step": 400, "warmup_steps": 400}})
    ref, lrs = param_groups.schedule_scales(opt, sched, 50, 5000, 0.1)
    assert lrs["backbone"] == 0.0 and ref == pytest.approx(lrs["text_embedding"])
    for g in opt.param_groups:  # what the official loop does next
        g["lr"] = ref * g["lr_scale"]
    assert {g["name"]: g["lr"] for g in opt.param_groups} == pytest.approx(lrs)
    ref, lrs = param_groups.schedule_scales(opt, sched, 1000, 5000, 0.1)
    assert ref == pytest.approx(lrs["backbone"]) and lrs["text_embedding"] > lrs["backbone"] > 0


# ----- eos_metrics ---------------------------------------------------------------------------------
def _batch(lengths, T=20):
    mask = torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]
    return mask


def test_eos_statistics_perfect_and_early():
    mask = _batch([10, 15, 20])  # the 20-frame row has no known end
    logits = torch.full(mask.shape, -10.0)
    logits[0, 10] = 5.0
    logits[1, 15] = 5.0
    stats = eos_metrics.finalize(eos_metrics.eos_statistics(logits, mask))
    assert stats["eos_rows"] == 2
    assert stats["eos_hit@-4"] == 1.0 and stats["eos_premature@-4"] == 0.0
    assert stats["eos_err_frames@-4"] == 0.0 and stats["eos_frame_acc@-4"] == 1.0
    logits[1, 8] = 0.0  # fires early (after the 6-frame grace period) at -4 but not at 0
    stats = eos_metrics.finalize(eos_metrics.eos_statistics(logits, mask))
    assert stats["eos_premature@-4"] == 0.5 and stats["eos_premature@0"] == 0.0
    assert stats["eos_err_frames@-4"] == pytest.approx((0 + 7) / 2)
    logits[0, 3] = 5.0  # inside the grace period: ignored, like TTSModel does
    assert eos_metrics.finalize(eos_metrics.eos_statistics(logits, mask))["eos_premature@0"] == 0.0


def test_eos_statistics_never_fires_counts_as_padded_length():
    mask = _batch([10])
    stats = eos_metrics.finalize(eos_metrics.eos_statistics(torch.full(mask.shape, -10.0), mask))
    assert stats["eos_hit@-4"] == 0.0 and stats["eos_err_frames@-4"] == 10.0


# ----- warm_start ----------------------------------------------------------------------------------
def test_romanize_piece_handles_fragments():
    assert romanize_piece("नमस्ते") == "namaste"
    assert romanize_piece("ोक") .startswith("o")  # leading vowel sign keeps its vowel
    assert romanize_piece("ीव्यू").startswith("ee")
    assert romanize_piece("ंगे").startswith("n")
    assert romanize_piece("्र") == "ra"
    assert romanize_piece("१२") == "12"


def _teacher():
    pieces = ["<unk>", "▁", "▁ke", "ke", "▁na", "ma", "s", "te", ".", "<0xE0>"]
    torch.manual_seed(0)
    table = torch.randn(len(pieces) + 1, 8) * 0.1
    table[9] = 0.0  # untrained byte row
    table[-1] = 0.0
    index = {p: i for i, p in enumerate(pieces)}

    def segment(text):  # greedy longest match, enough for the toy vocabulary
        ids, i = [], 0
        while i < len(text):
            for j in range(len(text), i, -1):
                if text[i:j] in index:
                    ids.append(index[text[i:j]])
                    i = j
                    break
            else:
                ids.append(index["<0xE0>"])
                i += 1
        return ids

    return PretrainedTable("toy", pieces, table, segment)


def test_build_warm_start_sources_and_scale():
    teacher = _teacher()
    pieces = ["<unk>", "▁ke", "▁के", "▁namaste", "▁नमस्ते", ".", "<0x41>", "▁q"]
    warm = build_warm_start(pieces, teacher, seed=0)
    assert warm.sources == ["exact", "exact", "transliterated", "decomposed", "transliterated", "exact",
                            "random", "decomposed"]
    # Devanagari and Roman spellings of a word start from the same row
    assert torch.allclose(warm.table[pieces.index("▁namaste")], warm.table[pieces.index("▁नमस्ते")])
    parts = teacher.table[[4, 5, 6, 7]]
    assert warm.table[3].norm() == pytest.approx(float(parts.norm(dim=1).mean()), rel=1e-5)
    assert torch.equal(warm.table[-1], teacher.table[-1])
    assert warm.report["by_source"]["random"] == 1


def test_fit_procrustes_recovers_rotation():
    torch.manual_seed(0)
    src = torch.randn(300, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    fit = fit_procrustes(src, 0.5 * src @ q)
    assert fit["heldout_cos"] > 0.999 and fit["scale"] == pytest.approx(0.5, rel=1e-4)
    assert abs(fit["shuffled_cos"]) < 0.3
    assert math.isclose(float((src @ fit["map"] - src @ q).abs().max()), 0.0, abs_tol=1e-3)
