"""Final dataset = selected utterances that also passed word alignment (align_words.py).

Utterances whose alignment failed or whose word boundaries sit implausibly far from the trimmed
audio edges usually have a transcript that does not match the audio; they are left out of both
the training manifests and the published dataset, so the two always agree.

Writes $WORK/data/processed/final.jsonl and regenerates data_pipeline/dataset_stats.json with the
whole curation funnel (measured -> passed filters -> selected -> aligned).

    python -m data_pipeline.finalize_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data_pipeline.common import DATA_ROOT, REPO_ROOT, read_jsonl, setup_logging, write_json, write_jsonl
from data_pipeline.prepare_dataset import build_stats

logger = setup_logging("finalize_dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data_pipeline.yaml")
    ap.add_argument("--processed", type=Path, default=DATA_ROOT / "processed")
    ap.add_argument("--manifests", type=Path, default=DATA_ROOT / "manifests")
    ap.add_argument("--stats", type=Path, default=REPO_ROOT / "data_pipeline" / "dataset_stats.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    selected = read_jsonl(args.processed / "selected.jsonl")
    aligned_ids = {r["utt_id"] for name in ("train_deva_aligned.jsonl", "valid_deva_aligned.jsonl")
                   for r in read_jsonl(args.manifests / name)}
    final = [r for r in selected if r["utt_id"] in aligned_ids]
    write_jsonl(args.processed / "final.jsonl", final)

    candidates = read_jsonl(args.processed / "candidates.jsonl")
    old = json.loads(args.stats.read_text()) if args.stats.exists() else {}
    stats = build_stats(final, candidates, cfg, old.get("timing_s", {}))
    alignment = json.loads((args.manifests / "alignment_report.json").read_text())
    stats["curation_funnel"] = {
        "measured": len(candidates),
        "passed_text_and_audio_filters": sum(1 for r in candidates if r.get("ok")),
        "selected_by_quota": len(selected),
        "passed_word_alignment": len(final),
    }
    stats["alignment"] = alignment
    stats["training_manifests"] = {
        "dir": str(args.manifests),
        "train_mix_rows": alignment["counts"]["train_mix_rows"],
        "valid_mix_rows": alignment["counts"]["valid_mix_rows"],
        "note": "one row per (utterance, script); English rows identical in Roman and code-switched form are kept once",
    }
    write_json(args.stats, stats)
    logger.info(f"final dataset: {len(final)} utterances, {stats['total']['hours']} h, "
                f"{stats['total']['speakers']} speakers; splits {stats['splits']}")


if __name__ == "__main__":
    main()
