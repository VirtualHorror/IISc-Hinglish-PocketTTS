"""Precompute Mimi 24 kHz latents (the Pocket TTS training targets) and voice-prompt references.

Latents: the official pocket-tts precompute (training/scripts/precompute_latents.py) is called on
each word-aligned training manifest, so the cache is exactly what training/train.py's
ensure_train_latents() accepts as fresh:
    <manifests>/latents/<mimi_hash[:8]>/<stem>_<idx>.safetensors   {"latents": [T, 32]} at 12.5 Hz
    <manifests>/<stem>_latents.jsonl        manifest rows + "latents_file"
    <manifests>/<stem>_latents.meta.json    stitch_frames, noise_floor, frame_rate, weights_path, mimi_hash
As long as the training config's model_config points at the same Mimi weights, train.py skips the
precompute and starts training straight away. Validation manifests are encoded too (for analysis);
train.py itself must keep reading the plain valid_*.jsonl.

Voice-prompt conditioning: during training the prompt is the start of the same utterance up to a
word boundary (word timings in the manifests + the latents above). For inference/evaluation this
script also exports per-speaker reference prompts (cleanest 4-8 s clip, end_on_pause-processed as
pocket-tts does for prompts): WAV + Mimi latents, indexed in voice_prompts/index.json.

    python -m data_pipeline.extract_mimi_latents [--model-config configs/hinglish_pocket_tts_24l.yaml]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import torch

from data_pipeline.common import DATA_ROOT, REPO_ROOT, read_jsonl, setup_logging, write_json

sys.path.insert(0, os.environ.get("POCKET_TTS_REPO", str(Path.home() / "vaani_pocket_tts_work" / "pocket-tts")))
from pocket_tts.data.audio_utils import end_on_pause  # noqa: E402
from pocket_tts.modules.stateful_module import init_states  # noqa: E402
from training.modules.builders import load_model_config  # noqa: E402
from training.modules.utils import stamp_state_names  # noqa: E402
from training.scripts.precompute_latents import load_frozen_mimi, mimi_encode_hash, precompute_manifest  # noqa: E402

logger = setup_logging("extract_mimi_latents")

TRAIN_MANIFESTS = ["train_mix_aligned", "train_deva_aligned", "train_roman_aligned", "train_cs_aligned"]
VALID_MANIFESTS = ["valid_mix_aligned"]


def check_cache(manifest: Path, mimi_hash: str, frame_rate: float) -> dict:
    lat_manifest = manifest.with_name(manifest.stem + "_latents.jsonl")
    meta = json.loads(lat_manifest.with_suffix(".meta.json").read_text())
    assert meta["mimi_hash"] == mimi_hash, "cache written with different Mimi weights"
    rows = read_jsonl(lat_manifest)
    frames, dims, bad = [], set(), 0
    for r in rows[:: max(1, len(rows) // 400)]:  # sample ~400 files
        lat = safetensors.torch.load_file(str(manifest.parent / r["latents_file"]))["latents"]
        dims.add(int(lat.shape[1]))
        frames.append(lat.shape[0])
        if abs(lat.shape[0] - r["duration"] * frame_rate) > 2:
            bad += 1
    return {"rows": len(rows), "latent_dims": sorted(dims), "stitch_frames": meta["stitch_frames"],
            "median_frames": float(np.median(frames)), "length_mismatches_in_sample": bad,
            "latents_dir": str(manifest.parent / "latents" / mimi_hash[:8])}


@torch.no_grad()
def latent_stats(manifest: Path) -> dict:
    rows = read_jsonl(manifest.with_name(manifest.stem + "_latents.jsonl"))
    lat = torch.cat([safetensors.torch.load_file(str(manifest.parent / r["latents_file"]))["latents"]
                     for r in rows[:: max(1, len(rows) // 1000)]])
    return {"frames": int(lat.shape[0]), "mean_abs_mean": round(float(lat.mean(0).abs().mean()), 4),
            "mean_std": round(float(lat.std(0).mean()), 4), "per_dim_mean": [round(float(x), 4) for x in lat.mean(0)],
            "per_dim_std": [round(float(x), 4) for x in lat.std(0)]}


def pick_voice_prompts(rows: list[dict], n_openslr: int = 8) -> list[dict]:
    """Cleanest 4-8 s training clip per studio speaker and language, plus a few OpenSLR narrators."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["split"] != "train" or not (4.0 <= r["duration_s"] <= 8.0):
            continue
        key = (r["speaker_id"], r["source_language"])  # recording language (en for IndicTTS-English)
        by[key].append(r)
    picks = []
    for (spk, lang), rs in sorted(by.items()):
        if spk.startswith("openslr104"):
            continue
        picks.append(dict(max(rs, key=lambda r: (r["snr_db"], -abs(r["duration_s"] - 6.0))), prompt_lang=lang))
    slr = sorted((max(rs, key=lambda r: r["snr_db"]) for (spk, _), rs in by.items() if spk.startswith("openslr104")),
                 key=lambda r: -r["snr_db"])[:n_openslr]
    picks += [dict(r, prompt_lang="hi-en") for r in slr]
    return picks


@torch.no_grad()
def export_voice_prompts(picks: list[dict], mimi, device: torch.device, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for r in picks:
        wav, sr = sf.read(r["audio_path"], dtype="float32")
        assert sr == mimi.sample_rate
        prompt = end_on_pause(torch.from_numpy(wav)[None], sr)  # [1, T]
        name = f"{r['speaker_id']}__{r['prompt_lang']}"
        sf.write(out_dir / f"{name}.wav", prompt[0].numpy(), sr, subtype="PCM_16")
        lat = mimi.encode_to_latent(prompt[None].to(device))[0].cpu()  # [T, 32]
        safetensors.torch.save_file({"latents": lat.contiguous()}, str(out_dir / f"{name}.safetensors"))
        index.append({"name": name, "speaker_id": r["speaker_id"], "language": r["prompt_lang"],
                      "source_utt_id": r["utt_id"], "text": r["text_code_switched"],
                      "duration_s": round(prompt.shape[-1] / sr, 3), "latent_frames": int(lat.shape[0]),
                      "wav": str(out_dir / f"{name}.wav"), "latents": str(out_dir / f"{name}.safetensors")})
    write_json(out_dir / "index.json", index)
    return index


@torch.no_grad()
def roundtrip_check(mimi, device: torch.device, wav_path: str, out_path: Path) -> dict:
    wav, sr = sf.read(wav_path, dtype="float32")
    x = torch.from_numpy(wav)[None, None].to(device)
    lat = mimi.encode_to_latent(x)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, 1, (lat.shape[1] + 8) * ratio)
    y = mimi.decode_from_latent(lat, state)[0, 0].float().cpu().numpy()
    n = min(len(wav), len(y))
    sf.write(out_path, y[:n], sr, subtype="PCM_16")
    corr = float(np.corrcoef(wav[:n], y[:n])[0, 1])
    return {"source": wav_path, "decoded": str(out_path), "frames": int(lat.shape[1]),
            "rms_ratio_db": round(float(20 * np.log10((np.sqrt(np.mean(y[:n] ** 2)) + 1e-9) / (np.sqrt(np.mean(wav[:n] ** 2)) + 1e-9))), 2),
            "waveform_correlation": round(corr, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "hinglish_pocket_tts_24l.yaml"))
    ap.add_argument("--manifests", type=Path, default=DATA_ROOT / "manifests")
    ap.add_argument("--final", type=Path, default=DATA_ROOT / "processed" / "final.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--decode-workers", type=int, default=12, help="CPU decode processes (shared machine)")
    ap.add_argument("--summary", type=Path, default=REPO_ROOT / "data_pipeline" / "latents_summary.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    config = load_model_config(args.model_config, {})
    mimi = load_frozen_mimi(config).to(device)
    stamp_state_names(mimi)  # the streaming decoder (round-trip check) needs absolute state names
    mimi_hash = mimi_encode_hash(mimi)
    logger.info(f"Mimi from {config.weights_path}: hash {mimi_hash[:8]}, {mimi.sample_rate} Hz, {mimi.frame_rate} fps")

    summary: dict = {"mimi_weights": str(config.weights_path), "mimi_hash": mimi_hash,
                     "frame_rate": mimi.frame_rate, "sample_rate": mimi.sample_rate, "manifests": {}}
    t0 = time.time()
    for stem in TRAIN_MANIFESTS + VALID_MANIFESTS:
        manifest = args.manifests / f"{stem}.jsonl"
        t1 = time.time()
        precompute_manifest(manifest, mimi, device, args.batch_size, args.decode_workers, str(config.weights_path))
        summary["manifests"][stem] = check_cache(manifest, mimi_hash, mimi.frame_rate)
        summary["manifests"][stem]["seconds"] = round(time.time() - t1, 1)
        logger.info(f"{stem}: {summary['manifests'][stem]}")
    summary["latent_stats_train_mix"] = latent_stats(args.manifests / "train_mix_aligned.jsonl")
    summary["precompute_seconds"] = round(time.time() - t0, 1)

    rows = read_jsonl(args.final)
    prompts = export_voice_prompts(pick_voice_prompts(rows), mimi, device, DATA_ROOT / "voice_prompts")
    summary["voice_prompts"] = {"dir": str(DATA_ROOT / "voice_prompts"), "count": len(prompts),
                                "names": [p["name"] for p in prompts]}
    rt_dir = Path(os.environ.get("WORK", REPO_ROOT / "outputs")) / "outputs"
    rt_dir.mkdir(parents=True, exist_ok=True)
    summary["roundtrip_check"] = roundtrip_check(mimi, device, prompts[0]["wav"], rt_dir / "mimi_roundtrip_check.wav")
    write_json(args.summary, summary)
    logger.info(f"wrote {args.summary}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
