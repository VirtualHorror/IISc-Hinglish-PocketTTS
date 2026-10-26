"""Check that fine-tuning can start immediately, using pocket-tts's own code paths.

1. the training config parses (training.args.load_args) and the model config validates;
2. the tokenizer has exactly lookup_table.n_bins pieces;
3. the latent cache next to train_jsonl is fresh for the configured Mimi weights
   (the test train.py's ensure_train_latents() applies);
4. manifests are big enough for the loader shards (train_utils.check_manifest_sizes);
5. the official DataLoader yields latent batches and encode_batch produces training tensors;
6. (--forward) the warm-started model computes a finite loss on one batch.

    python -m data_pipeline.verify_training_ready [--config configs/finetune_hinglish_24l.yaml] [--forward]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from data_pipeline.common import REPO_ROOT, setup_logging, write_json

sys.path.insert(0, os.environ.get("POCKET_TTS_REPO", str(Path.home() / "vaani_pocket_tts_work" / "pocket-tts")))
from pocket_tts.modules.text_conditioner import build_tokenizer  # noqa: E402
from training.args import load_args  # noqa: E402
from training.dataloader import DataLoader, encode_batch  # noqa: E402
from training.modules.builders import build_models, load_model_config  # noqa: E402
from training.scripts.precompute_latents import load_frozen_mimi, mimi_encode_hash  # noqa: E402
from training.train_utils import check_manifest_sizes  # noqa: E402

logger = setup_logging("verify_training_ready")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "finetune_hinglish_24l.yaml"))
    ap.add_argument("--forward", action="store_true", help="also build the model and run one forward pass")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data_pipeline" / "training_readiness.json")
    args_cli = ap.parse_args()
    report: dict = {"config": args_cli.config}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = load_args(args_cli.config)
    config = load_model_config(args.model_config, args.model_overrides)
    lt = config.flow_lm.lookup_table
    tok = build_tokenizer(lt.n_bins, lt.tokenizer_path, lt.tokenizer)  # asserts vocab == n_bins
    report["tokenizer"] = {"path": lt.tokenizer_path, "n_bins": lt.n_bins, "ok": True,
                           "example_ids": tok.encode("कल सुबह meeting है, please reminder सेट कर दो.")[:16]}

    train_path = Path(args.data.train_jsonl)
    latents_manifest = train_path.with_name(train_path.stem + "_latents.jsonl")
    meta = json.loads(latents_manifest.with_suffix(".meta.json").read_text())
    mimi = load_frozen_mimi(config).to(device)
    current = mimi_encode_hash(mimi)
    with latents_manifest.open() as f:
        first = json.loads(f.readline())
    fresh = meta["mimi_hash"] == current and "latents_file" in first
    report["latent_cache"] = {"manifest": str(latents_manifest), "fresh": fresh, "mimi_hash": current[:8],
                              "stitch_frames": meta["stitch_frames"]}
    if not fresh:
        raise SystemExit(f"latent cache is stale or missing: {report['latent_cache']}")

    check_manifest_sizes(args, world_size=1)
    report["manifest_sizes"] = "ok"

    loader = DataLoader(str(latents_manifest), tok.encode, 4, mimi.sample_rate, mimi.frame_rate,
                        args.data.max_duration_sec, args.data.max_voice_prompt_sec, rank=0, world_size=1,
                        seed=0, num_bucket_batches=1, prompt_trim_max_sec=args.data.prompt_trim_max_sec,
                        final_punct_dropout=args.data.final_punct_dropout)
    it = iter(loader)
    batch = next(it)
    latents, mask, voice, n_voice = encode_batch(mimi, batch, device)
    report["batch"] = {"latents": list(latents.shape), "mask_frames": [int(x) for x in mask.sum(1)],
                       "voice_prompt_latents": list(voice.shape), "voice_prompt_frames": [int(x) for x in n_voice],
                       "text_tokens": [len(t) for t in batch.text_tokens],
                       "decoded_text_0": tok.decode(batch.text_tokens[0].tolist())}
    logger.info(f"batch: {report['batch']}")

    if args_cli.forward:
        args.compile = False
        model, _, _ = build_models(args)
        model.to(device).train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = model(latents, mask, batch.text_tokens, voice, num_voice_prompt_frames=n_voice)
        report["forward"] = {"loss": float(loss), "finite": bool(torch.isfinite(loss)),
                             "metrics": {k: round(float(v), 4) for k, v in metrics.items() if v.numel() == 1},
                             "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
                             "peak_gpu_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None}
        logger.info(f"forward: {report['forward']}")
    write_json(args_cli.out, report)
    logger.info(f"READY: wrote {args_cli.out}")


if __name__ == "__main__":
    main()
