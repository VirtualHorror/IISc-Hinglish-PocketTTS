"""Objective benchmark of the exported Pocket TTS Indian Hinglish model on 24 Hinglish sentences
(eval/benchmark_sentences.json: 8 Roman, 8 Devanagari, 8 code-switched), run through
release/inference.py's HinglishTTS, i.e. exactly the code evaluators use.

Quality (GPU): round-trip ASR (Whisper large-v3-turbo, Hindi) CER / WER against each sentence's
canonical spoken Devanagari (eval/asr.py), per input script and per voice:
  default       the inference default: auto_normalize on, rendered in Devanagari, every voice
  keep_script   auto_normalize on, input script kept (script="auto"), every preset voice
  raw           auto_normalize off (terminal punctuation only), every preset voice
  invariance    every sentence re-rendered by the normaliser in all three scripts, first voice:
                same content, so differences come from the script alone
plus generations that hit the length cap without EOS, and speaking rate.

Speed: real-time factor (synthesis seconds / audio seconds; < 1 is faster than real time) and
time-to-first-audio (first streamed chunk, text normalisation included) per sentence, after a
warm-up, with the voice state cached: on the GPU, and on the CPU for each --cpu-threads value in
fp32 and dynamic int8 (pocket-tts quantize=True).

    python -m eval.evaluate_hinglish_tts --model-dir $WORK/release [--cpu-threads 1,4]
    python -m eval.evaluate_hinglish_tts --model-dir $WORK/release --variant 12l \
        --out eval/evaluation_results_student_12l.json --audio-dir $WORK/eval_audio/benchmark_student_12l

writes eval/evaluation_results.json, audio under $WORK/eval_audio/benchmark/.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from data_pipeline.common import REPO_ROOT, WORK, setup_logging, write_json
from eval.asr import HinglishScorer, WhisperASR
from release.inference import HinglishTTS, detect_script

logger = setup_logging("evaluate_hinglish_tts")

SCRIPTS = ("roman", "devanagari", "code_switched")


class NoEosCounter(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "without EOS" in record.getMessage():
            self.count += 1


def generate_set(tts: HinglishTTS, jobs: list[dict], out_dir: Path) -> list[dict]:
    """jobs: {id, text, voice, auto_normalize, script} -> rows with wav path, duration, EOS flag."""
    counter = NoEosCounter()
    tts_logger = logging.getLogger("pocket_tts.models.tts_model")
    tts_logger.addHandler(counter)
    rows = []
    try:
        for k, job in enumerate(jobs):
            torch.manual_seed(2000 + k)
            before = counter.count
            prepared = tts.normalize(job["text"], job["auto_normalize"], job.get("script"))
            audio = tts.synthesize(job["text"], voice=job["voice"], auto_normalize=job["auto_normalize"],
                                   script=job.get("script"))
            path = out_dir / job["voice"] / f"{job['id']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio, tts.sample_rate)
            rows.append({**job, "model_input": prepared, "wav": str(path.relative_to(out_dir)),
                         "duration_s": round(len(audio) / tts.sample_rate, 3), "no_eos": counter.count > before})
    finally:
        tts_logger.removeHandler(counter)
    return rows


def score_rows(rows: list[dict], scorer: HinglishScorer, group_keys: tuple[str, ...]) -> dict[str, Any]:
    def stats(sel: list[dict]) -> dict[str, Any]:
        s = scorer.score([r["reference"] for r in sel], [r["hyp"] for r in sel])
        chars = sum(len(p["ref"].replace(" ", "")) for p in s["per_utt"])
        return {"cer": round(s["cer"], 4), "wer": round(s["wer"], 4), "n": s["n"],
                "no_eos": sum(r["no_eos"] for r in sel),
                "chars_per_second": round(chars / max(1e-9, sum(r["duration_s"] for r in sel)), 2)}

    out = {"overall": stats(rows)}
    for key in group_keys:
        for value in sorted({r[key] for r in rows}):
            out[f"{key}={value}"] = stats([r for r in rows if r[key] == value])
    s = scorer.score([r["reference"] for r in rows], [r["hyp"] for r in rows])
    for r, p in zip(rows, s["per_utt"], strict=True):
        r.update(ref_canonical=p["ref"], hyp_canonical=p["hyp"], cer=round(p["cer"], 4), wer=round(p["wer"], 4))
    return out


def time_streaming(tts: HinglishTTS, sentences: list[dict], voice: str, warmup: int = 2) -> dict[str, Any]:
    tts.voice_state(voice)  # cached: speed excludes encoding the prompt
    for s in sentences[:warmup]:
        tts.synthesize(s["text"], voice=voice)
    per = []
    for k, s in enumerate(sentences):
        torch.manual_seed(3000 + k)
        t0 = time.perf_counter()
        stream = tts.stream(s["text"], voice=voice)
        first = next(stream)
        ttfa = time.perf_counter() - t0
        n = len(first) + sum(len(c) for c in stream)
        total = time.perf_counter() - t0
        audio_s = n / tts.sample_rate
        per.append({"id": s["id"], "ttfa_ms": round(ttfa * 1000, 1), "seconds": round(total, 3),
                    "audio_s": round(audio_s, 3), "rtf": round(total / audio_s, 4)})
    rtfs, ttfas = [p["rtf"] for p in per], [p["ttfa_ms"] for p in per]
    return {"voice": voice, "n": len(per),
            "rtf_mean": round(statistics.mean(rtfs), 4), "rtf_total": round(sum(p["seconds"] for p in per) / sum(p["audio_s"] for p in per), 4),
            "x_real_time": round(sum(p["audio_s"] for p in per) / sum(p["seconds"] for p in per), 2),
            "ttfa_ms_p50": round(statistics.median(ttfas), 1), "ttfa_ms_p90": round(sorted(ttfas)[int(0.9 * (len(ttfas) - 1))], 1),
            "ttfa_ms_mean": round(statistics.mean(ttfas), 1), "per_sentence": per}


def display_path(path: Path) -> str:
    """Repo-relative path for files of this repo, the bare name otherwise (no workstation paths)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=WORK / "release")
    ap.add_argument("--sentences", type=Path, default=REPO_ROOT / "eval" / "benchmark_sentences.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "evaluation_results.json")
    ap.add_argument("--audio-dir", type=Path, default=WORK / "eval_audio" / "benchmark")
    ap.add_argument("--cpu-threads", default="1,4")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--variant", default="24l", help="model variant (inference.VARIANTS), e.g. a distilled student")
    cli = ap.parse_args()

    sentences = json.loads(cli.sentences.read_text(encoding="utf-8"))["sentences"]
    for s in sentences:  # the input script must be what the id claims
        assert detect_script(s["text"]) == s["script"], s
    results: dict[str, Any] = {
        "model": "MSRIT-NBM/pocket-tts-indian-hinglish (exported folder: " + cli.model_dir.name + "/)",
        "variant": cli.variant,
        "audio_dir": str(cli.audio_dir),
        "benchmark": {"file": display_path(cli.sentences), "n_sentences": len(sentences),
                      "by_script": {sc: sum(s["script"] == sc for s in sentences) for sc in SCRIPTS}},
        "environment": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "cpu": cpu_name(), "torch": torch.__version__, "python": platform.python_version()},
        "settings": {"temperature": None, "eos_threshold": None, "sampler_decode_steps": None},
    }

    # ---------------- quality + GPU speed ----------------
    tts = HinglishTTS.from_pretrained(str(cli.model_dir), device="cuda", variant=cli.variant)
    m = tts.model
    results["settings"] = {"temperature": m.temp, "eos_threshold": m.eos_threshold,
                           "sampler_decode_steps": m.sampler_decode_steps, "default_script": tts.script,
                           "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
    voices = tts.voices()
    results["voices"] = voices
    processor = tts.processor
    scorer = HinglishScorer(processor)
    jobs = []
    for v in voices:
        for s in sentences:
            jobs.append({"id": s["id"], "set": "default", "voice": v, "script": None, "input_script": s["script"],
                         "auto_normalize": True, "text": s["text"]})
            jobs.append({"id": s["id"] + "__keep", "set": "keep_script", "voice": v, "script": "auto",
                         "input_script": s["script"], "auto_normalize": True, "text": s["text"]})
            jobs.append({"id": s["id"] + "__raw", "set": "raw", "voice": v, "script": None, "input_script": s["script"],
                         "auto_normalize": False, "text": s["text"]})
    for s in sentences:
        for target in SCRIPTS:
            jobs.append({"id": f"{s['id']}__as_{target}", "set": "invariance", "voice": voices[0], "script": target,
                         "input_script": target, "auto_normalize": True, "text": s["text"]})
    logger.info(f"generating {len(jobs)} utterances on GPU with voices {voices}")
    rows = generate_set(tts, jobs, cli.audio_dir)
    asr = WhisperASR(device="cuda")
    for r, hyp in zip(rows, asr.transcribe([cli.audio_dir / r["wav"] for r in rows]), strict=True):
        r["hyp"] = hyp
        r["reference"] = r["text"]  # canonicalised (numbers verbalised, Devanagari) by the scorer
    results["asr"] = {"model": asr.model_id, "language": "hi", "metric": "corpus CER (no spaces) / WER in canonical spoken Devanagari"}
    del asr
    torch.cuda.empty_cache()
    quality = {}
    for name, keys in (("default", ("input_script", "voice")), ("keep_script", ("input_script", "voice")),
                       ("raw", ("input_script", "voice")), ("invariance", ("input_script",))):
        quality[name] = score_rows([r for r in rows if r["set"] == name], scorer, keys)
        logger.info(f"{name}: {quality[name]['overall']} | by script "
                    f"{ {k: v['cer'] for k, v in quality[name].items() if k.startswith('input_script=')} }")
    results["quality"] = quality
    results["samples"] = [{k: r[k] for k in ("set", "id", "voice", "text", "model_input", "hyp", "cer", "wer",
                                             "duration_s", "no_eos", "wav")} for r in rows]
    speed = {"gpu_fp32": time_streaming(tts, sentences, voices[0])}
    logger.info(f"GPU speed: RTF {speed['gpu_fp32']['rtf_total']} TTFA p50 {speed['gpu_fp32']['ttfa_ms_p50']} ms")
    del tts
    torch.cuda.empty_cache()

    # ---------------- CPU speed ----------------
    if not cli.skip_cpu:
        for quantize in (False, True):
            cpu_tts = HinglishTTS.from_pretrained(str(cli.model_dir), device="cpu", quantize=quantize,
                                                  variant=cli.variant)
            for n in [int(x) for x in cli.cpu_threads.split(",") if x]:
                torch.set_num_threads(n)
                key = f"cpu_{'int8' if quantize else 'fp32'}_{n}thread{'s' if n > 1 else ''}"
                speed[key] = time_streaming(cpu_tts, sentences, voices[0])
                logger.info(f"{key}: RTF {speed[key]['rtf_total']} TTFA p50 {speed[key]['ttfa_ms_p50']} ms")
            del cpu_tts
        torch.set_num_threads(1)
    results["speed"] = speed
    results["speed_notes"] = ("RTF = synthesis seconds / audio seconds (lower is faster; 1/RTF = x real time). "
                              "TTFA = time from calling stream() (text normalisation included) to the first "
                              "decoded audio chunk, voice state cached, after 2 warm-up sentences. pocket-tts "
                              "runs generation and Mimi decoding in two threads; 'N threads' is torch's intra-op "
                              "thread count.")
    write_json(cli.out, results)
    logger.info(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
