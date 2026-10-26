#!/usr/bin/env python3
"""Where the CPU time goes: the released model's speed on this machine, per precision, thread count and
model variant, with the two stages pocket-tts runs in parallel timed separately.

pocket-tts generates in two threads: the FlowLM (backbone + sampler head) produces one 80 ms latent frame
per step while the Mimi decoder turns queued frames into audio. Real time needs the slower stage under
80 ms per frame. This probe times both (wrapping TTSModel._run_flow_lm_and_increment_step and
mimi.decode_from_latent) on top of the benchmark's end-to-end TTFA / RTF protocol (stream call to the
first decoded chunk, text normalisation included, voice state cached, after warm-up).

    python scripts/cpu_speed_probe.py --model-dir artifacts/release --threads 1,2 --out eval/results/cpu_speed_local.json

Note: pocket-tts calls torch.set_num_threads(1) when it is imported; the probe (like demo_app.py) sets the
thread count afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "release"))

import torch  # noqa: E402

from demo_app import cpu_name  # noqa: E402
from inference import VARIANTS, HinglishTTS  # noqa: E402


def probe(tts: HinglishTTS, sentences: list[str], voice: str) -> dict:
    model = tts.model
    flow_ms: list[float] = []
    mimi: list[tuple[float, int]] = []
    run_flow, decode = model._run_flow_lm_and_increment_step, model.mimi.decode_from_latent

    def timed_flow(*a, **k):
        t = time.perf_counter()
        out = run_flow(*a, **k)
        flow_ms.append((time.perf_counter() - t) * 1000)
        return out

    def timed_decode(x, *a, **k):
        t = time.perf_counter()
        out = decode(x, *a, **k)
        mimi.append(((time.perf_counter() - t) * 1000, x.shape[1]))
        return out

    model._run_flow_lm_and_increment_step, model.mimi.decode_from_latent = timed_flow, timed_decode
    try:
        per = []
        for k, text in enumerate(sentences):
            flow_ms.clear()
            mimi.clear()
            torch.manual_seed(5000 + k)
            t0 = time.perf_counter()
            stream = tts.stream(text, voice=voice)
            first = next(stream)
            ttfa = time.perf_counter() - t0
            n = len(first) + sum(len(c) for c in stream)
            total = time.perf_counter() - t0
            steps = flow_ms[1:]  # the first call is the text prefill
            per.append({"ttfa_ms": ttfa * 1000, "rtf": total / (n / 24000), "audio_s": n / 24000, "seconds": total,
                        "prefill_ms": flow_ms[0], "flow_step_ms": statistics.mean(steps),
                        "mimi_ms_per_frame": sum(m for m, _ in mimi) / max(1, sum(f for _, f in mimi)),
                        "mimi_frames_per_call": sum(f for _, f in mimi) / max(1, len(mimi))})
    finally:
        model._run_flow_lm_and_increment_step, model.mimi.decode_from_latent = run_flow, decode
    audio = sum(p["audio_s"] for p in per)
    return {"n": len(per), "rtf_total": round(sum(p["seconds"] for p in per) / audio, 4),
            "x_real_time": round(audio / sum(p["seconds"] for p in per), 2),
            "ttfa_ms_p50": round(statistics.median(p["ttfa_ms"] for p in per), 1),
            **{k: round(statistics.mean(p[k] for p in per), 2)
               for k in ("prefill_ms", "flow_step_ms", "mimi_ms_per_frame", "mimi_frames_per_call")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "release")
    ap.add_argument("--variants", default="24l,12l")
    ap.add_argument("--precisions", default="int8,fp32")
    ap.add_argument("--threads", default="1,2")
    ap.add_argument("--sentences", type=int, default=6, help="first N default-set benchmark sentences (2 per script)")
    ap.add_argument("--voice", default="indian_female_ananya")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results" / "cpu_speed_local.json")
    cli = ap.parse_args()

    bench = json.loads((REPO_ROOT / "eval" / "benchmark_sentences.json").read_text(encoding="utf-8"))
    rows = bench["sentences"] if isinstance(bench, dict) else bench
    by_script: dict[str, list[str]] = {}
    for r in rows:
        by_script.setdefault(r["script"], []).append(r["text"])
    per_script = max(1, cli.sentences // max(1, len(by_script)))
    sentences = [t for texts in by_script.values() for t in texts[:per_script]]

    results = []
    for variant in [v for v in cli.variants.split(",") if v in VARIANTS]:
        sub = cli.model_dir / VARIANTS[variant] if VARIANTS[variant] else cli.model_dir
        if not (sub / "model.safetensors").exists():
            continue
        for precision in cli.precisions.split(","):
            t = time.perf_counter()
            tts = HinglishTTS.from_pretrained(str(cli.model_dir), quantize=precision == "int8", variant=variant)
            load_s = time.perf_counter() - t
            layers = len(tts.model.flow_lm.transformer.layers)
            tts.voice_state(cli.voice)
            for threads in [int(x) for x in cli.threads.split(",")]:
                torch.set_num_threads(threads)
                for s in sentences[:2]:  # warm-up
                    tts.synthesize(s, voice=cli.voice)
                r = {"variant": variant, "layers": layers, "precision": precision, "threads": threads,
                     "load_s": round(load_s, 1), **probe(tts, sentences, cli.voice)}
                results.append(r)
                print(json.dumps(r), flush=True)
            del tts
    record = {"machine": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "platform": platform.platform(),
                          "torch": torch.__version__, "quantized_engine": torch.backends.quantized.engine},
              "sentences": sentences, "voice": cli.voice,
              "protocol": "TTFA = stream call (text normalisation included) to the first decoded chunk, voice state "
                          "cached, 2 warm-up sentences per setting. flow_step_ms = one FlowLM step (24 or 6 "
                          "transformer layers + LSD head) = one 80 ms frame; mimi_ms_per_frame = Mimi decoding "
                          "time per frame (runs in a second thread). Real time needs both under 80 ms.",
              "results": results}
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
