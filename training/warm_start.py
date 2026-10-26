"""Warm-start the Hinglish text lookup table from pretrained Pocket TTS embeddings.

The official recipe for a new tokenizer (`reset_text_embedding: true`, pocket-tts
training/configs/finetune_language.yaml) keeps nn.Embedding's N(0, 1) init: rows of norm ~32 fed to
a backbone whose own text rows have norm ~0.4 (kyutai english_2026-04_24l: median 0.45, 95th
percentile 0.69). With 10 h of data that shock is worth avoiding, so every piece of
tokenizer/hinglish_bilingual_4k.model gets a row in the English teacher's embedding space:

  exact           the piece is in the teacher's tokenizer with a trained row -> that row
  indus           Devanagari piece in Indus-Labs/indus-pocket-tts's Hindi/Hinglish table -> that row
                  mapped into the teacher's space by a scaled orthogonal Procrustes fit on the pieces
                  both tables share; used only if the fit holds on held-out shared pieces
  decomposed      Latin / digit / punctuation piece -> the teacher's own Unigram segmentation of it
                  -> mean of the parts' rows
  transliterated  Devanagari piece -> casual romanisation -> teacher segmentation -> mean of rows
                  ("▁नमस्ते" -> "▁namaste" -> ▁na ma s te): a phonetic hint the backbone can read
  random          nothing usable -> N(0, sigma), sigma = the teacher table's element std

Rows built from parts are rescaled to the parts' mean norm, so every row has the scale the backbone
was trained on. Teacher rows with norm < DEAD_ROW_NORM are unusable (314 of 4,001 are ~0: byte
fallback and foreign-script pieces English training never used). The padding row (index n_bins,
never placed in a sequence) is copied from the teacher.

    python -m training.warm_start configs/finetune_hinglish.yaml [--compare]

writes training/warm_start_report.json: the source of every row, the Procrustes fit, and with
--compare the step-0 validation loss of the official N(0, 1) init against the warm starts.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("warm_start")

SPIECE = "▁"
DEAD_ROW_NORM = 0.05
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
CONTROL = re.compile(r"^<[^>]+>$")  # <unk> <s> </s> <pad> <0x..>
# Pieces can start mid-syllable (vowel sign, virama, nasal mark): give them a readable start.
MATRA_TO_VOWEL = {"ा": "आ", "ि": "इ", "ी": "ई", "ु": "उ", "ू": "ऊ", "ृ": "ऋ", "ॄ": "ॠ", "े": "ए",
                  "ै": "ऐ", "ो": "ओ", "ौ": "औ", "ॉ": "ऑ", "ॅ": "ऍ", "ॆ": "ऎ", "ॊ": "ऒ"}
LEADING_SIGN = {"ं": "n", "ँ": "n", "ः": "h", "्": "", "़": ""}
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
REPORT_PATH = Path(__file__).resolve().parent / "warm_start_report.json"


def romanize_piece(text: str) -> str:
    """Casual romanisation of a Devanagari subword (without its ▁ marker); non-Devanagari kept."""
    from data_pipeline.hinglish_text_processor import romanize_hindi

    text = unicodedata.normalize("NFC", text).translate(DEVA_DIGITS)
    prefix = ""
    while text and text[0] in LEADING_SIGN:
        prefix += LEADING_SIGN[text[0]]
        text = text[1:]
    if text and text[0] in MATRA_TO_VOWEL:
        text = MATRA_TO_VOWEL[text[0]] + text[1:]
    parts = re.split(r"([ऀ-ॿ]+)", text)
    return prefix + "".join(romanize_hindi(p) if DEVANAGARI.search(p) else p for p in parts)


@dataclass
class PretrainedTable:
    """A pretrained Pocket TTS text table and the vocabulary that indexes it."""

    name: str
    pieces: list[str]  # token id -> piece
    table: torch.Tensor  # [len(pieces) + 1, dim] float32, last row = padding
    segment: Callable[[str], list[int]] | None = None  # raw text, ▁ markers included -> ids

    def __post_init__(self):
        if self.table.shape[0] != len(self.pieces) + 1:
            raise ValueError(f"{self.name}: table {tuple(self.table.shape)} vs {len(self.pieces)} pieces")
        self.index = {p: i for i, p in enumerate(self.pieces)}
        self.alive = self.table.norm(dim=1) >= DEAD_ROW_NORM

    def row(self, piece: str) -> int | None:
        """Id of `piece` if it has a trained row."""
        i = self.index.get(piece)
        return i if i is not None and bool(self.alive[i]) else None


def load_pretrained_table(name: str, weights_path: str, tokenizer_path: str) -> PretrainedTable:
    """Text table of a released model.safetensors plus its tokenizer (.json Unigram or .model)."""
    import safetensors
    from pocket_tts.utils.utils import download_if_necessary

    with safetensors.safe_open(str(download_if_necessary(weights_path)), framework="pt") as f:
        table = f.get_tensor("flow_lm.conditioner.embed.weight").float()
    tok_file = download_if_necessary(tokenizer_path)
    if tok_file.suffix == ".json":
        import tokenizers

        unigram = tokenizers.Tokenizer.from_file(str(tok_file)).model
        pieces = [p for p, _ in json.loads(tok_file.read_text())["model"]["vocab"]]
        # model.tokenize is the raw Viterbi segmentation: no normaliser, no ▁ prepended.
        return PretrainedTable(name, pieces, table, lambda text: [t.id for t in unigram.tokenize(text)])
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=str(tok_file))
    return PretrainedTable(name, [sp.id_to_piece(i) for i in range(sp.get_piece_size())], table)


def fit_procrustes(src: torch.Tensor, dst: torch.Tensor, heldout_frac: float = 0.2,
                   seed: int = 0) -> dict[str, Any]:
    """Scaled orthogonal map s * src @ W ~ dst, scored on held-out anchor pairs."""

    def solve(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, float]:
        u, _, vt = torch.linalg.svd(a.T @ b, full_matrices=False)
        return u @ vt, float(b.norm(dim=1).mean() / a.norm(dim=1).mean())

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(src), generator=gen)
    n_test = max(1, int(len(src) * heldout_frac))
    test, train = perm[:n_test], perm[n_test:]
    w, s = solve(src[train], dst[train])
    mapped = s * src[test] @ w
    shuffled = dst[test][torch.randperm(n_test, generator=gen)]
    w_all, s_all = solve(src, dst)
    return {
        "map": w_all,
        "scale": s_all,
        "n_anchors": len(src),
        "heldout_cos": round(float(F.cosine_similarity(mapped, dst[test], dim=1).mean()), 4),
        "shuffled_cos": round(float(F.cosine_similarity(mapped, shuffled, dim=1).mean()), 4),
        "raw_cos": round(float(F.cosine_similarity(src, dst, dim=1).mean()), 4),
    }


@dataclass
class WarmStart:
    table: torch.Tensor  # [len(pieces) + 1, dim]
    sources: list[str]  # per piece: exact | indus | decomposed | transliterated | random
    report: dict[str, Any]


def build_warm_start(pieces: list[str], teacher: PretrainedTable, helper: PretrainedTable | None = None,
                     min_heldout_cos: float = 0.3, seed: int = 0) -> WarmStart:
    if teacher.segment is None:
        raise ValueError("the teacher table needs a Unigram segmenter (tokenizer.json)")
    dim = teacher.table.shape[1]
    trained = teacher.table[:-1][teacher.alive[:-1]]
    sigma = float(trained.std())
    gen = torch.Generator().manual_seed(seed)
    report: dict[str, Any] = {"teacher": teacher.name, "dead_row_norm": DEAD_ROW_NORM,
                              "teacher_trained_rows": int(teacher.alive[:-1].sum()), "random_sigma": round(sigma, 5)}

    helper_map = None
    if helper is not None:
        if helper.table.shape[1] != dim:
            raise ValueError(f"{helper.name} table width {helper.table.shape[1]} != teacher {dim}")
        anchors = [p for p in helper.pieces if not CONTROL.match(p)
                   and teacher.row(p) is not None and helper.row(p) is not None]
        fit = fit_procrustes(torch.stack([helper.table[helper.index[p]] for p in anchors]),
                             torch.stack([teacher.table[teacher.index[p]] for p in anchors]), seed=seed)
        accepted = fit["heldout_cos"] >= min_heldout_cos
        report["helper"] = {"name": helper.name, "accepted": accepted, "min_heldout_cos": min_heldout_cos,
                            **{k: v for k, v in fit.items() if k != "map"}}
        if accepted:
            helper_map = (fit["map"], fit["scale"])
        logger.info(f"{helper.name} -> {teacher.name} Procrustes: {report['helper']}")

    def compose(text: str) -> tuple[torch.Tensor, list[str]] | None:
        ids = [i for i in teacher.segment(text) if teacher.alive[i]]
        if not ids:
            return None
        rows = teacher.table[ids]
        mean = rows.mean(0)
        return mean * (rows.norm(dim=1).mean() / mean.norm().clamp_min(1e-8)), [teacher.pieces[i] for i in ids]

    table = torch.empty(len(pieces) + 1, dim)
    sources: list[str] = []
    examples: dict[str, list[str]] = {}
    for i, piece in enumerate(pieces):
        row, source, note = None, "random", ""
        j = teacher.row(piece)
        if j is not None:
            row, source = teacher.table[j], "exact"
        elif CONTROL.match(piece):
            pass  # control / byte-fallback piece without a trained teacher row
        elif DEVANAGARI.search(piece):
            k = helper.row(piece) if helper_map is not None and helper is not None else None
            if k is not None:
                row, source = helper_map[1] * helper.table[k] @ helper_map[0], "indus"
            else:
                roman = romanize_piece(piece.removeprefix(SPIECE))
                built = compose((SPIECE if piece.startswith(SPIECE) else "") + roman) if roman else None
                if built is not None:
                    (row, parts), source = built, "transliterated"
                    note = f"{roman} -> {' '.join(parts)}"
        else:
            built = compose(piece)
            if built is not None:
                (row, parts), source = built, "decomposed"
                note = " ".join(parts)
        if row is None:
            row = torch.randn(dim, generator=gen) * sigma
        table[i] = row
        sources.append(source)
        if len(examples.setdefault(source, [])) < 12:
            examples[source].append(f"{piece} {note}".strip())
    table[-1] = teacher.table[-1]

    def script(p: str) -> str:
        return "devanagari" if DEVANAGARI.search(p) else ("control" if CONTROL.match(p) else "latin/other")

    norms = table[:-1].norm(dim=1)
    report.update({
        "pieces": len(pieces),
        "by_source": dict(Counter(sources)),
        "by_script_and_source": {f"{s}/{src}": n for (s, src), n in
                                 sorted(Counter((script(p), src) for p, src in zip(pieces, sources)).items())},
        "row_norm": {"mean": round(float(norms.mean()), 4), "median": round(float(norms.median()), 4),
                     "p95": round(float(norms.quantile(0.95)), 4)},
        "teacher_row_norm": {"mean": round(float(trained.norm(dim=1).mean()), 4),
                             "median": round(float(trained.norm(dim=1).median()), 4)},
        "examples": examples,
    })
    return WarmStart(table, sources, report)


def warm_start_from_config(model_config: Any, cfg: dict[str, Any], pieces: list[str]) -> WarmStart:
    """`cfg` is the `hinglish.warm_start` section of a training config."""
    teacher = load_pretrained_table("kyutai/" + Path(str(model_config.weights_path).split("@")[0]).parent.name,
                                    str(model_config.weights_path), cfg["teacher_tokenizer"])
    helper = None
    if cfg.get("helper_weights"):
        helper = load_pretrained_table(cfg.get("helper_name", "helper"), cfg["helper_weights"], cfg["helper_tokenizer"])
    return build_warm_start(pieces, teacher, helper, min_heldout_cos=float(cfg.get("min_heldout_cos", 0.3)),
                            seed=int(cfg.get("seed", 0)))


def apply_warm_start(flow_lm: torch.nn.Module, warm: WarmStart) -> None:
    weight = flow_lm.conditioner.embed.weight
    if tuple(weight.shape) != tuple(warm.table.shape):
        raise ValueError(f"text table {tuple(weight.shape)} vs warm start {tuple(warm.table.shape)}")
    with torch.no_grad():
        weight.copy_(warm.table.to(weight))


def tokenizer_pieces(flow_lm: torch.nn.Module) -> list[str]:
    tok = flow_lm.conditioner.tokenizer
    if hasattr(tok, "sp"):
        return [tok.sp.id_to_piece(i) for i in range(tok.sp.get_piece_size())]
    return [tok.tokenizer.id_to_token(i) for i in range(tok.tokenizer.get_vocab_size())]


def main() -> None:
    from training.train_hinglish import load_config_sections

    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="training config (official keys + hinglish: section)")
    ap.add_argument("--compare", action="store_true",
                    help="also score the official N(0,1) init and the warm starts by step-0 validation loss")
    ap.add_argument("--valid-batches", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")

    from training.modules.builders import build_models, load_model_config

    args, hcfg = load_config_sections(cli.config)
    ws_cfg = hcfg["warm_start"]
    model_config = load_model_config(args.model_config, args.model_overrides)
    args.compile = False
    model, mimi, _ = build_models(args)
    pieces = tokenizer_pieces(model.flow_lm)
    warm = warm_start_from_config(model_config, ws_cfg, pieces)
    report = {"config": cli.config, **warm.report}
    if cli.compare:
        from training.train import validate

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        mimi.to(device)
        args.num_valid_batches = cli.valid_batches
        inits = {"official_reset_N(0,1)": model.flow_lm.conditioner.embed.weight.detach().clone().cpu()}
        no_helper = warm_start_from_config(model_config, {**ws_cfg, "helper_weights": None}, pieces)
        inits["warm_start_teacher_only"] = no_helper.table
        if warm.report.get("helper", {}).get("accepted"):
            inits["warm_start_teacher+helper"] = warm.table
        scores = {}
        for name, table in inits.items():
            apply_warm_start(model.flow_lm, WarmStart(table, [], {}))
            torch.manual_seed(1234)
            metrics = validate(model, mimi, args, device, rank=0, world_size=1, step=0)
            scores[name] = {k: round(v, 4) for k, v in metrics.items()}
            logger.info(f"{name}: {scores[name]}")
        report["step0_validation"] = {"manifest": args.data.valid_jsonl, "batches": cli.valid_batches,
                                      "batch_size": args.batch_size, "scores": scores}
    cli.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote {cli.out}: {report['by_source']}")


if __name__ == "__main__":
    main()
