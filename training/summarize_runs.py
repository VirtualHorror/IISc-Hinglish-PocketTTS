"""Collect training runs into training/training_summary.json: validation curves (loss per script,
EOS accuracy, group learning rates) from <run_dir>/progress.jsonl, the training loss around each
validation, wall clock / speed, and round-trip ASR from <run_dir>/asr_progress*.jsonl.

    python -m training.summarize_runs --run hinglish_24l_ws=configs/finetune_hinglish.yaml \
        --run hinglish_24l_official=configs/ablation_official_recipe.yaml \
        --selected hinglish_24l_ws:3000 --reason "..."
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from data_pipeline.common import REPO_ROOT, WORK, write_json

VALID_KEYS = ("loss", "flow_loss", "eos_loss", "eos_hit@-4", "eos_premature@-4", "eos_frame_acc@-4",
              "eos_err_frames@-4", "eos_hit@0", "eos_premature@0", "deva/loss", "roman/loss", "cs/loss",
              "lr/text_embedding", "lr/backbone", "lr/heads")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_run(run_dir: Path, config: str) -> dict[str, Any]:
    events = read_jsonl(run_dir / "progress.jsonl")
    train = [e for e in events if e["type"] == "train"]
    valid = [e for e in events if e["type"] == "valid"]
    out: dict[str, Any] = {"run_dir": str(run_dir), "config": config}
    if train:
        t0 = datetime.strptime(events[0]["time"], "%Y-%m-%d %H:%M:%S%z")
        t1 = datetime.strptime(events[-1]["time"], "%Y-%m-%d %H:%M:%S%z")
        out["wall_clock_minutes"] = round((t1 - t0).total_seconds() / 60, 1)
        out["steps"] = max(e["step"] for e in train)
        out["it_s_median"] = round(statistics.median(e["it_s"] for e in train if e.get("it_s")), 2)
    rows = []
    for v in valid:
        near = [e["metrics"]["loss"] for e in train if v["step"] - 250 < e["step"] <= v["step"]]
        row = {"step": v["step"], **{k: round(v["metrics"][k], 5) for k in VALID_KEYS if k in v["metrics"]}}
        if near:
            row["train_loss_last250"] = round(statistics.mean(near), 4)
        rows.append(row)
    out["valid"] = rows
    if rows:
        best = min(rows, key=lambda r: r["loss"])
        out["best_valid_loss"] = {"step": best["step"], "loss": best["loss"]}
    for path in sorted(run_dir.glob("asr_progress*.jsonl")):
        tag = path.stem.removeprefix("asr_progress").lstrip("_") or "monitor"
        recs = []
        for r in read_jsonl(path):
            rec = {"step": r["step"], **{k: r["overall"][k] for k in ("cer", "wer", "no_eos", "n", "duration_ratio")}}
            rec["cer_by_script"] = {k.split("=")[1]: v["cer"] for k, v in r.items() if k.startswith("script=")}
            rec["cer_by_language"] = {k.split("=")[1]: v["cer"] for k, v in r.items() if k.startswith("language=")}
            rec["cer_by_voice"] = {k.split("=")[1]: v["cer"] for k, v in r.items() if k.startswith("voice=")}
            recs.append(rec)
        out[f"asr_{tag}"] = sorted(recs, key=lambda r: r["step"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=config (run dir under $WORK/runs)")
    ap.add_argument("--selected", default="", help="run:step chosen for release")
    ap.add_argument("--reason", default="")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "training" / "training_summary.json")
    args = ap.parse_args()
    summary: dict[str, Any] = {"runs": {}}
    for spec in args.run:
        name, config = spec.split("=", 1)
        summary["runs"][name] = summarize_run(WORK / "runs" / name, config)
    floors = {}
    for name in summary["runs"]:
        for path in sorted((REPO_ROOT / "eval" / "results").glob(f"asr_monitor_{name}*.json")):
            floors[path.stem.removeprefix(f"asr_monitor_{name}").lstrip("_") or "monitor"] = {
                k: v for k, v in json.loads(path.read_text())["asr_floor"].items() if k != "per_utt"}
    summary["asr_floor_real_recordings"] = floors
    if args.selected:
        run, step = args.selected.split(":")
        summary["selected"] = {"run": run, "step": int(step), "checkpoint": f"checkpoint_{int(step):08d}.pt",
                               "weights": "EMA", "reason": args.reason}
    write_json(args.out, summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
