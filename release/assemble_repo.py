"""Add the benchmark samples, evaluation results and training records to the release folder
(after release.export_release and eval.evaluate_hinglish_tts, before release.model_card):

  samples/<voice>/<sentence id>.wav   every benchmark sentence with every voice (default front-end):
                                      roman_*, deva_* and mixed_* = the three input scripts
  samples/index.json                  text, model input, Whisper transcript, CER/WER per file
  evaluation_results.json             eval/evaluation_results.json
  training/                           training config, warm-start report, training summary
  eval/                               benchmark, selection and monitor sentence sets

    python -m release.assemble_repo [--release $WORK/release]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from data_pipeline.common import REPO_ROOT, WORK, setup_logging, write_json

logger = setup_logging("assemble_repo")

TRAINING_RECORDS = [
    REPO_ROOT / "configs" / "finetune_hinglish.yaml",
    REPO_ROOT / "configs" / "hinglish_pocket_tts_24l.yaml",
    REPO_ROOT / "training" / "warm_start_report.json",
    REPO_ROOT / "training" / "training_summary.json",
    REPO_ROOT / "eval" / "results" / "voice_prompt_search.json",
]
EVAL_INPUTS = [  # the sentence sets behind the reported numbers
    REPO_ROOT / "eval" / "benchmark_sentences.json",
    REPO_ROOT / "eval" / "selection_set.json",
    REPO_ROOT / "eval" / "monitor_set.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=WORK / "release")
    args = ap.parse_args()
    results_path = REPO_ROOT / "eval" / "evaluation_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    samples = args.release / "samples"
    if samples.exists():
        shutil.rmtree(samples)
    index = []
    for s in results["samples"]:
        if s["set"] != "default":
            continue
        dst = samples / s["voice"] / f"{s['id']}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(results["audio_dir"]) / s["wav"], dst)
        index.append({"file": str(dst.relative_to(args.release)), "voice": s["voice"], "text": s["text"],
                      "model_input": s["model_input"], "whisper": s["hyp"], "cer": s["cer"], "wer": s["wer"],
                      "duration_s": s["duration_s"]})
    write_json(samples / "index.json", index)
    public = {k: v for k, v in results.items() if k != "audio_dir"}  # no workstation paths in the repo
    write_json(args.release / "evaluation_results.json", public)
    (args.release / "training").mkdir(exist_ok=True)
    for path in TRAINING_RECORDS:
        shutil.copy2(path, args.release / "training" / path.name)
    (args.release / "eval").mkdir(exist_ok=True)
    for path in EVAL_INPUTS:
        shutil.copy2(path, args.release / "eval" / path.name)
    logger.info(f"{len(index)} samples, evaluation results and {len(TRAINING_RECORDS)} training records "
                f"copied into {args.release}")


if __name__ == "__main__":
    main()
