#!/usr/bin/env python3
"""Pre-render the studio's preset takes (demo/cache/) so they play instantly in demo_app.py.

Every preset of demo/presets.json x every studio voice x both front-ends ("phonetic", "raw") x every
model variant present, synthesised through release/inference.py exactly as the studio does live, one
fixed seed per take (no best-of-n). Each take records the time to first audio and RTF measured while
rendering it (after warm-up, voice state cached: the benchmark's protocol) and, with --asr, what
Whisper large-v3-turbo hears (CER / WER in canonical Devanagari, eval/asr.py), so the studio can show
where and how every number was measured.

    python scripts/build_demo_cache.py --model-dir $WORK/release --device cuda --asr

Writes demo/cache/<key>.wav (16-bit PCM, 24 kHz) and demo/cache/index.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "release"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from demo_app import cpu_name, take_key  # noqa: E402
from inference import VARIANTS, HinglishTTS  # noqa: E402

MODES = {"phonetic": True, "raw": False}  # mode -> auto_normalize


def device_label(device: str) -> str:
    return torch.cuda.get_device_name() if device.startswith("cuda") else cpu_name()


def render(tts: HinglishTTS, text: str, voice: str, auto_normalize: bool, seed: int) -> tuple[list, float, float]:
    """(chunks, ttfa_s, total_s), timed as eval/evaluate_hinglish_tts.py times streaming."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    chunks, ttfa = [], None
    for chunk in tts.stream(text, voice=voice, auto_normalize=auto_normalize):
        if ttfa is None:
            ttfa = time.perf_counter() - t0
        chunks.append(chunk)
    return chunks, ttfa or 0.0, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "demo" / "cache")
    ap.add_argument("--variants", default="24l,12l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--asr", action="store_true", help="transcribe every take with Whisper (needs the GPU box)")
    cli = ap.parse_args()

    import numpy as np

    presets = json.loads((REPO_ROOT / "demo" / "presets.json").read_text(encoding="utf-8"))
    items = [(c["id"], p) for c in presets["categories"] for p in c["presets"]]
    voices = [v["name"] for v in json.loads((cli.model_dir / "voices" / "index.json").read_text(encoding="utf-8"))]
    cli.out.mkdir(parents=True, exist_ok=True)
    for old in cli.out.glob("*.wav"):
        old.unlink()
    label = device_label(cli.device)
    takes = []
    for variant in [v for v in cli.variants.split(",") if v in VARIANTS]:
        sub = cli.model_dir / VARIANTS[variant] if VARIANTS[variant] else cli.model_dir
        if not (sub / "model.safetensors").exists():
            print(f"variant {variant}: not exported, skipped", flush=True)
            continue
        tts = HinglishTTS.from_pretrained(str(cli.model_dir), device=cli.device, variant=variant)
        layers = len(tts.model.flow_lm.transformer.layers)
        for voice in voices:
            tts.voice_state(voice)
        for k in range(2):  # warm-up: kernels, allocator
            render(tts, items[k][1]["text"], voices[0], True, seed=k)
        for category, preset in items:
            for voice in voices:
                for mode, auto in MODES.items():
                    key = take_key(variant, voice, mode, preset["text"])
                    seed = zlib.crc32(key.encode()) & 0x7FFFFFFF
                    chunks, ttfa, total = render(tts, preset["text"], voice, auto, seed)
                    audio = np.concatenate(chunks)
                    audio_s = len(audio) / tts.sample_rate
                    sf.write(str(cli.out / f"{key}.wav"), audio, tts.sample_rate, subtype="PCM_16")
                    takes.append({"key": key, "engine": variant, "layers": layers, "category": category,
                                  "preset": preset["id"], "voice": voice, "mode": mode, "text": preset["text"],
                                  "model_input": tts.normalize(preset["text"], auto_normalize=auto),
                                  "file": f"{key}.wav", "audio_s": round(audio_s, 3), "ttfa_ms": round(ttfa * 1000, 1),
                                  "rtf": round(total / audio_s, 4), "synth_s": round(total, 3), "seed": seed,
                                  "device": label, "precision": "fp32"})
            print(f"{variant} {preset['id']}: rendered {len(voices) * len(MODES)} takes", flush=True)
        del tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    if cli.asr and takes:
        from eval.asr import HinglishScorer, WhisperASR

        asr, scorer = WhisperASR(device=cli.device), HinglishScorer()
        hyps = asr.transcribe([str(cli.out / t["file"]) for t in takes], language="hi")
        scores = scorer.score([t["text"] for t in takes], hyps)
        for t, hyp, s in zip(takes, hyps, scores["per_utt"], strict=True):
            t["asr"] = {"hyp": hyp, "cer": round(s["cer"], 4), "wer": round(s["wer"], 4)}
        for variant in sorted({t["engine"] for t in takes}):
            for mode in MODES:
                sel = [t for t in takes if t["engine"] == variant and t["mode"] == mode]
                s = scorer.score([t["text"] for t in sel], [t["asr"]["hyp"] for t in sel])
                summary[f"{variant}/{mode}"] = {"cer": round(s["cer"], 4), "wer": round(s["wer"], 4), "n": s["n"]}
        summary["asr_model"] = asr.model_id
    index = {"rendered_on": label, "device": cli.device, "time": time.strftime("%Y-%m-%d %H:%M"),
             "protocol": "one take per preset x voice x front-end x model, fixed seed; TTFA = stream call (text "
                         "normalisation included) to the first decoded chunk, voice state cached, after warm-up",
             "asr_summary": summary, "takes": takes}
    (cli.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(takes)} takes to {cli.out}" + (f"; ASR {summary}" if summary else ""), flush=True)


if __name__ == "__main__":
    main()
