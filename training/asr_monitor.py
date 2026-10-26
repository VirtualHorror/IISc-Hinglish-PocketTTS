"""Score training checkpoints by round-trip ASR, through the release inference path.

For every checkpoint_<step>.pt of a run (oldest first, each once): load it with pocket_tts TTSModel
(EMA weights; config defaults: temperature 0.3, EOS threshold -4), synthesize the monitoring set
(eval/monitor_set.json: held-out validation utterances, each in Devanagari, Roman Hinglish and
code-switched form) with each studio voice prompt, transcribe with Whisper, and score CER / WER in
canonical Devanagari (eval/asr.py). Also logged: generations that hit the length cap without EOS
(TTSModel's "without EOS" warning), the duration ratio to the real recording, and the GPU speed.
The real recordings are transcribed once as the ASR floor.

Appends one record per checkpoint to <run_dir>/asr_progress.jsonl and rewrites
eval/results/asr_monitor_<run>.json; audio goes to $WORK/eval_audio/monitor/<run>/step<N>/.

    python -m training.asr_monitor --run-dir $WORK/runs/hinglish_24l_ws [--watch]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from data_pipeline.common import REPO_ROOT, WORK, setup_logging, write_json
from eval.asr import HinglishScorer, WhisperASR, asr_language

logger = setup_logging("asr_monitor")

SCRIPTS = ("deva", "roman", "cs")
DEFAULT_VOICES = {
    "indian_female": str(WORK / "data/voice_prompts/indictts_hindifull_female__hi.wav"),
    "indian_male": str(WORK / "data/voice_prompts/indictts_hindifull_male__hi.wav"),
}


class _NoEosCounter(logging.Handler):
    """Counts TTSModel's 'Maximum generation length reached without EOS' warnings."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "without EOS" in record.getMessage():
            self.count += 1


def synthesize(tts: Any, voices: dict[str, Any], items: list[dict], out_dir: Path) -> list[dict]:
    """One generation per (voice, item); returns rows with the wav path and EOS / timing facts."""
    counter = _NoEosCounter()
    logging.getLogger("pocket_tts.models.tts_model").addHandler(counter)
    rows = []
    try:
        for voice_name, state in voices.items():
            for k, item in enumerate(items):
                torch.manual_seed(1000 + k)
                before = counter.count
                t0 = time.perf_counter()
                audio = tts.generate_audio(state, item["text"]).float().cpu().numpy()
                seconds = time.perf_counter() - t0
                path = out_dir / f"{voice_name}__{item['id']}.wav"
                sf.write(str(path), audio, tts.sample_rate)
                duration = len(audio) / tts.sample_rate
                rows.append({**item, "voice": voice_name, "wav": str(path), "duration_s": round(duration, 3),
                             "no_eos": counter.count > before, "gen_seconds": round(seconds, 3)})
    finally:
        logging.getLogger("pocket_tts.models.tts_model").removeHandler(counter)
    return rows


def summarize(rows: list[dict], scorer: HinglishScorer) -> dict[str, Any]:
    def group(sel: list[dict]) -> dict[str, Any]:
        s = scorer.score([r["text"] for r in sel], [r["hyp"] for r in sel])
        ratios = [r["duration_s"] / r["ref_duration_s"] for r in sel if r.get("ref_duration_s")]
        return {"cer": round(s["cer"], 4), "wer": round(s["wer"], 4), "n": s["n"],
                "no_eos": sum(r["no_eos"] for r in sel),
                "duration_ratio": round(sum(ratios) / len(ratios), 3) if ratios else None,
                "rtf_gpu": round(sum(r["gen_seconds"] for r in sel) / max(1e-9, sum(r["duration_s"] for r in sel)), 4)}

    out = {"overall": group(rows)}
    for key in ("script", "language", "voice"):
        for value in sorted({r[key] for r in rows}):
            out[f"{key}={value}"] = group([r for r in rows if r[key] == value])
    return out


def monitor_items(path: Path) -> list[dict]:
    """Every distinct script rendering of every utterance (English Roman = code-switched and
    Hindi Devanagari = code-switched renderings are generated once, under the first script)."""
    items = []
    for utt in json.loads(path.read_text(encoding="utf-8")):
        seen = set()
        for script in SCRIPTS:
            text = utt["text"][script]
            if text in seen:
                continue
            seen.add(text)
            items.append({"id": f"{utt['utt_id']}__{script}", "utt_id": utt["utt_id"], "script": script,
                          "language": utt["language"], "text": text, "ref_duration_s": utt["duration_s"]})
    return items


def asr_floor(monitor_set: Path, asr: WhisperASR, scorer: HinglishScorer) -> dict[str, Any]:
    utts = json.loads(monitor_set.read_text(encoding="utf-8"))
    hyps = asr.transcribe([u["audio"] for u in utts], language=[asr_language(u["language"]) for u in utts])
    s = scorer.score([u["text"]["deva"] for u in utts], hyps)
    return {"cer": round(s["cer"], 4), "wer": round(s["wer"], 4), "n": s["n"],
            "per_utt": [{"utt_id": u["utt_id"], **p} for u, p in zip(utts, s["per_utt"], strict=True)]}


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for p in run_dir.glob("checkpoint_*.pt"):
        m = re.fullmatch(r"checkpoint_(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "hinglish_pocket_tts_24l.yaml"))
    ap.add_argument("--monitor-set", type=Path, default=REPO_ROOT / "eval" / "monitor_set.json")
    ap.add_argument("--voice", action="append", default=[], help="name=path.wav (default: two studio voices)")
    ap.add_argument("--steps", default="", help="comma-separated steps to score (default: every new one)")
    ap.add_argument("--watch", action="store_true", help="keep polling for new checkpoints until the run is done")
    ap.add_argument("--tag", default="", help="suffix for the output files (a second monitoring set)")
    ap.add_argument("--device", default="cuda")
    cli = ap.parse_args()

    from pocket_tts.models.tts_model import TTSModel

    run = cli.run_dir.name
    suffix = f"_{cli.tag}" if cli.tag else ""
    progress_path = cli.run_dir / f"asr_progress{suffix}.jsonl"
    summary_path = REPO_ROOT / "eval" / "results" / f"asr_monitor_{run}{suffix}.json"
    audio_root = WORK / "eval_audio" / "monitor" / f"{run}{suffix}"
    voices = dict(v.split("=", 1) for v in cli.voice) or DEFAULT_VOICES
    items = monitor_items(cli.monitor_set)
    done = set()
    records = []
    if progress_path.exists():
        records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
        done = {r["step"] for r in records}
    wanted = {int(s) for s in cli.steps.split(",") if s}

    asr = WhisperASR(device=cli.device)
    scorer = HinglishScorer()
    floor = asr_floor(cli.monitor_set, asr, scorer)
    logger.info(f"ASR floor (real recordings): CER {floor['cer']:.4f} WER {floor['wer']:.4f}")

    while True:
        todo = [(s, p) for s, p in checkpoints(cli.run_dir) if s not in done and (not wanted or s in wanted)]
        for step, path in todo:
            time.sleep(5)  # let a checkpoint that is still being renamed into place settle
            tts = TTSModel.load_model(config=cli.model_config, checkpoint=str(path)).to(cli.device)
            states = {name: tts.get_state_for_audio_prompt(Path(wav)) for name, wav in voices.items()}
            out_dir = audio_root / f"step{step:08d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = synthesize(tts, states, items, out_dir)
            del tts, states
            torch.cuda.empty_cache()
            hyps = asr.transcribe([r["wav"] for r in rows], language=[asr_language(r["language"]) for r in rows])
            for r, hyp in zip(rows, hyps, strict=True):
                r["hyp"] = hyp
            record = {"step": step, "checkpoint": path.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      **summarize(rows, scorer),
                      "samples": [{k: r[k] for k in ("voice", "id", "text", "hyp", "duration_s", "no_eos")}
                                  for r in rows]}
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            done.add(step)
            ov = record["overall"]
            by_script = {s: record[f"script={s}"]["cer"] for s in SCRIPTS if f"script={s}" in record}
            logger.info(f"step {step}: CER {ov['cer']:.4f} WER {ov['wer']:.4f} no-EOS {ov['no_eos']}/{ov['n']} "
                        f"dur ratio {ov['duration_ratio']} | CER by script {by_script}")
            write_json(summary_path, {
                "run": run, "model_config": cli.model_config, "monitor_set": str(cli.monitor_set),
                "voices": voices, "asr": {"model": asr.model_id, "language": "hi; en for English-matrix utterances"},
                "asr_floor": floor,
                "steps": [{k: v for k, v in r.items() if k != "samples"} for r in sorted(records, key=lambda r: r["step"])],
            })
        finished = (cli.run_dir / "model.safetensors").exists() and not _training_alive(cli.run_dir)
        if not cli.watch or (finished and not [s for s, _ in checkpoints(cli.run_dir) if s not in done]):
            break
        time.sleep(60)


def _training_alive(run_dir: Path) -> bool:
    """True while a train_hinglish process for this run is running (checked via /proc)."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "training.train_hinglish" in cmd:
            return True
    return False


if __name__ == "__main__":
    main()
