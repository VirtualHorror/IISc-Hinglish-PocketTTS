"""Hinglish fine-tuning entrypoint: pocket-tts's own training loop (training/train.py: data loading,
CFG dropout, EOS + LSD flow loss, EMA, checkpoints, model.safetensors export, SIGTERM resume) with
these hooks swapped in, all configured by a `hinglish:` section the official parser never sees:

  build_models     + warm-started text table (training/warm_start.py) instead of N(0, 1) rows
  build_optimizer  three AdamW groups with their own peak lr / start step / warmup
  lr_at            staged unfreeze + per-group cosine decay (training/param_groups.py)
  validate         per-script validation (Devanagari / Roman / code-switched manifests, identical
                   cuts), plus EOS accuracy at the inference threshold (training/eos_metrics.py)
  write_samples    the configured held-out sentences with fixed studio voice prompts

    cd ~/vaani_pocket_tts_work/repo && source ../env.sh
    python -m training.train_hinglish configs/finetune_hinglish.yaml

The training log prints the backbone's lr once it trains (the text table's before); every group's
lr is recorded with each validation in <run_dir>/progress.jsonl ("lr/<group>").
"""

import os

if __name__ == "__main__":  # read by the CUDA caching allocator at init, so before torch loads
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import safetensors.torch
import soundfile
import torch
import yaml

import training.train as official
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import init_states
from training import eos_metrics, param_groups
from training.args import TrainArgs, _from_dict
from training.dataloader import DataLoader, encode_batch
from training.dataloader.types import Batch
from training.distributed import avg_across_ranks
from training.modules.model import TrainableTTS
from training.warm_start import apply_warm_start, tokenizer_pieces, warm_start_from_config

logger = logging.getLogger("train")

SAMPLE_MAX_FRAMES = 250  # 20 s at 12.5 Hz: a sample still running then never emitted EOS


class _Run:
    """What the hooks share; one training run per process."""

    cfg: dict[str, Any] = {}
    schedules: dict[str, param_groups.GroupSchedule] = {}
    optimizer: torch.optim.Optimizer | None = None
    lrs: dict[str, float] = {}
    voices: dict[str, torch.Tensor] = {}


RUN = _Run()


def load_config_sections(path: str | Path) -> tuple[TrainArgs, dict[str, Any]]:
    """(official TrainArgs, the `hinglish:` section) of a training config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    hinglish = raw.pop("hinglish", None) or {}
    return _from_dict(TrainArgs, raw), hinglish


def _load_args(path: str) -> TrainArgs:
    args, RUN.cfg = load_config_sections(path)
    RUN.schedules = param_groups.parse_schedules(RUN.cfg["param_groups"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "hinglish.yaml").write_text(
        yaml.safe_dump(RUN.cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return args


_official_build_models = official.build_models


def _build_models(args: TrainArgs):
    model, mimi, config = _official_build_models(args)
    ws = RUN.cfg.get("warm_start", {})
    if ws.get("enabled") and args.start_from_pretrained and args.reset_text_embedding:
        warm = warm_start_from_config(config, ws, tokenizer_pieces(model.flow_lm))
        apply_warm_start(model.flow_lm, warm)
        (args.run_dir / "warm_start.json").write_text(
            json.dumps(warm.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"warm-started text table: {warm.report['by_source']}, row norm {warm.report['row_norm']}")
    return model, mimi, config


def _build_optimizer(model: TrainableTTS, args: TrainArgs, device: torch.device, rank: int):
    if args.optim.type != "adamw":
        raise ValueError(f"train_hinglish builds AdamW groups only, not {args.optim.type!r}")
    groups = param_groups.build_param_groups(model, args.optim.weight_decay)
    RUN.optimizer = torch.optim.AdamW(groups, lr=args.optim.lr, betas=args.optim.betas, eps=args.optim.eps,
                                      weight_decay=args.optim.weight_decay, fused=device.type == "cuda")
    if rank == 0:
        sizes = {g["name"]: f"{sum(p.numel() for p in g['params']) / 1e6:.1f}M" for g in groups}
        logger.info(f"param groups {sizes}; schedules {RUN.schedules}")
    return RUN.optimizer


def _lr_at(step: int, args: TrainArgs) -> float:
    assert RUN.optimizer is not None
    floor = float(RUN.cfg.get("lr_floor_ratio", 0.1))
    reference, RUN.lrs = param_groups.schedule_scales(RUN.optimizer, RUN.schedules, step, args.max_steps, floor)
    if step == RUN.schedules["backbone"].start_step and step > 0:
        logger.info(f"step {step}: unfreezing backbone and heads (lrs {RUN.lrs})")
    return reference


def _validation_batches(manifest: str, tokenize: Callable[[str], list[int]], args: TrainArgs,
                        mimi: MimiModel, rank: int, world_size: int) -> Iterator[Batch]:
    """Every row of `manifest` once (whole batches), with the official validation loader settings
    (seed 0, no shuffle, no augmentation): the same cuts at every validation, and across scripts."""
    loader = DataLoader(manifest, tokenize, args.batch_size, mimi.sample_rate, mimi.frame_rate,
                        args.data.max_duration_sec, args.data.max_voice_prompt_sec, rank, world_size,
                        seed=0, shuffle=False)
    n_batches = min(args.num_valid_batches, len(loader.entries) // args.batch_size)
    batches = loader._batches()  # unthreaded: closing it leaves no prefetch thread behind
    try:
        for _ in range(n_batches):
            yield next(batches)
    finally:
        batches.close()


@torch.no_grad()
def _validate(model: TrainableTTS, mimi: MimiModel, args: TrainArgs, device: torch.device, rank: int,
              world_size: int, step: int) -> dict[str, float]:
    model.eval()
    teacher = model.distill_teacher
    tokenize = model.flow_lm.conditioner.tokenizer.encode
    manifests = RUN.cfg.get("valid_manifests") or {"mix": args.data.valid_jsonl}
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    captured: list[torch.Tensor] = []
    hook = model.flow_lm.out_eos.register_forward_hook(lambda _m, _i, out: captured.append(out.detach()))
    results: dict[str, float] = {}
    pooled_loss: dict[str, list[float]] = {}
    pooled_eos: dict[str, float] = {}
    try:
        for script, manifest in manifests.items():
            totals: dict[str, float] = {}
            eos_sums: dict[str, float] = {}
            n = 0
            for batch in _validation_batches(manifest, tokenize, args, mimi, rank, world_size):
                latents, mask, voice, n_voice = encode_batch(mimi, batch, device)
                captured.clear()
                with autocast:
                    _, metrics = model(latents, mask, batch.text_tokens, voice, num_voice_prompt_frames=n_voice)
                    if teacher is not None:
                        # Distillation returns only the distillation loss (out_eos never runs): also
                        # score the student's own flow / EOS losses through its frozen heads.
                        model.__dict__["distill_teacher"] = None
                        try:
                            _, own = model(latents, mask, batch.text_tokens, voice, num_voice_prompt_frames=n_voice)
                        finally:
                            model.__dict__["distill_teacher"] = teacher
                        metrics = {**own, **metrics}  # "loss" stays the distillation loss
                for k, v in metrics.items():
                    if v.numel() == 1:
                        totals[k] = totals.get(k, 0.0) + v.item()
                for k, v in eos_metrics.eos_statistics(captured[-1].squeeze(-1), mask).items():
                    eos_sums[k] = eos_sums.get(k, 0.0) + v
                n += 1
            for k, v in totals.items():
                results[f"{script}/{k}"] = avg_across_ranks(v / max(1, n))
                pooled_loss.setdefault(k, []).append(results[f"{script}/{k}"])
            for k, v in eos_metrics.finalize(eos_sums).items():
                results[f"{script}/{k}"] = v
            for k, v in eos_sums.items():
                pooled_eos[k] = pooled_eos.get(k, 0.0) + v
    finally:
        hook.remove()
    results.update({k: sum(v) / len(v) for k, v in pooled_loss.items()})  # mean over scripts
    results.update(eos_metrics.finalize(pooled_eos))
    results.update({f"lr/{g}": lr for g, lr in RUN.lrs.items()})
    if rank == 0:
        keys = ("loss", "flow_loss", "eos_loss", "eos_hit@-4", "eos_premature@-4", "eos_err_frames@-4")
        per_script = {s: round(results[f"{s}/loss"], 4) for s in manifests}
        logger.info(f"valid @ step {step}: { {k: round(results[k], 4) for k in keys if k in results} } "
                    f"loss by script {per_script}")
    return results


def _sample_voices(device: torch.device) -> dict[str, torch.Tensor]:
    if not RUN.voices:
        for name, path in (RUN.cfg.get("sample_voices") or {}).items():
            RUN.voices[name] = safetensors.torch.load_file(path)["latents"].float()
    return {k: v.to(device) for k, v in RUN.voices.items()}


@torch.no_grad()
def _write_samples(model: TrainableTTS, mimi: MimiModel, tokenize: Callable[[str], list[int]], args: TrainArgs,
                   run_dir: Path, step: int, voice_latents: torch.Tensor, device: torch.device):
    """The configured sentences with every configured studio voice (raw weights, the run's sample
    temperature and CFG), plus step<N>.json listing text, voice and EOS outcome per file. Falls
    back to the official behaviour (first batch's prompt) when no voices are configured."""
    voices = _sample_voices(device) or {"batch_prompt": voice_latents}
    out_dir = run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in args.sample_sentences]
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    index = []
    for voice_name, voice in voices.items():
        torch.manual_seed(step)
        outs = model.generate(tokens, [voice] * len(tokens), max_frames=SAMPLE_MAX_FRAMES,
                              temp=args.sample_temp, cfg_coef=args.sample_cfg_coef)
        for i, latents in enumerate(outs):
            entry = {"voice": voice_name, "text": args.sample_sentences[i], "frames": int(latents.shape[0]),
                     "no_eos": int(latents.shape[0]) >= SAMPLE_MAX_FRAMES}
            if latents.shape[0] >= 8:  # the Mimi decoder needs a few frames of context
                state = init_states(mimi, 1, (latents.shape[0] + 8) * ratio)
                audio = mimi.decode_from_latent(latents[None].to(device), state)[0, 0]
                entry["wav"] = f"step{step:08d}_{voice_name}_{i:02d}.wav"
                soundfile.write(str(out_dir / entry["wav"]), audio.float().cpu().numpy(), mimi.sample_rate)
            index.append(entry)
    (out_dir / f"step{step:08d}.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    model.train()
    n_no_eos = sum(e["no_eos"] for e in index)
    logger.info(f"wrote {len(index)} samples at step {step} ({n_no_eos} without EOS by {SAMPLE_MAX_FRAMES} frames)")


def main(config_path: str) -> None:
    official.load_args = _load_args
    official.build_models = _build_models
    official.build_optimizer = _build_optimizer
    official.lr_at = _lr_at
    official.validate = _validate
    official.write_samples = _write_samples
    official.main(config_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python -m training.train_hinglish <config.yaml>"
    exit_code = 0
    try:
        main(sys.argv[1])
    except SystemExit as exc:  # e.g. train.py's "non-finite gradient at step N"
        logger.error(f"training stopped: {exc}")
        exit_code = 0 if exc.code in (None, 0) else 1
    except BaseException:
        logger.exception("training failed")
        exit_code = 1
    # SubprocessDataLoader.close() SIGKILLs its loaders, orphaning the torch_shm_manager processes
    # they spawned. Those keep the multiprocessing resource tracker's pipe open, so interpreter
    # shutdown would wait on the tracker forever (holding the GPU memory). Checkpoints and logs are
    # written by now: exit without the atexit hooks; the managers then see us go and exit too.
    logging.shutdown()
    os._exit(exit_code)
