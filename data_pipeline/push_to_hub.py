"""Publish the curated dataset as a PRIVATE Hugging Face dataset repo, with its dataset card.

Audio is the processed 24 kHz mono signal, stored losslessly as FLAC. Columns: audio,
text_devanagari, text_roman, text_code_switched, speaker_id, source_dataset, source_url,
human_verified, duration_s, plus provenance/quality columns (utt_id, text_original, language,
gender, source_license, source_utt_id, source_sample_rate, snr_db).

    python -m data_pipeline.push_to_hub [--repo MSRIT-NBM/iisc-hinglish-pocket-tts-dataset]
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import soundfile as sf
import yaml
from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import DatasetCard, HfApi

from data_pipeline.common import DATA_ROOT, REPO_ROOT, hf_token, load_dotenv, read_jsonl, setup_logging
from data_pipeline.dataset_card import render_card

logger = setup_logging("push_to_hub")

FEATURES = Features({
    "audio": Audio(sampling_rate=24000),
    "utt_id": Value("string"),
    "text_devanagari": Value("string"),
    "text_roman": Value("string"),
    "text_code_switched": Value("string"),
    "text_original": Value("string"),
    "language": Value("string"),
    "speaker_id": Value("string"),
    "gender": Value("string"),
    "source_dataset": Value("string"),
    "source_url": Value("string"),
    "source_license": Value("string"),
    "source_utt_id": Value("string"),
    "source_sample_rate": Value("int32"),
    "human_verified": Value("bool"),
    "duration_s": Value("float32"),
    "snr_db": Value("float32"),
})


def flac_bytes(path: str) -> bytes:
    wav, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def to_row(r: dict, sources: dict) -> dict:
    row = {k: r[k] for k in FEATURES if k != "audio"}
    row["source_license"] = sources[r["source_key"]]["license"]  # the config is authoritative
    row["audio"] = {"bytes": flac_bytes(r["audio_path"]), "path": f"{r['utt_id']}.flac"}
    return row


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO", "MSRIT-NBM/iisc-hinglish-pocket-tts-dataset"))
    ap.add_argument("--final", type=Path, default=DATA_ROOT / "processed" / "final.jsonl")
    ap.add_argument("--stats", type=Path, default=REPO_ROOT / "data_pipeline" / "dataset_stats.json")
    ap.add_argument("--card-only", action="store_true", help="only (re)upload README.md")
    args = ap.parse_args()
    token = hf_token("write")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(args.repo)
    if not info.private:
        raise SystemExit(f"{args.repo} exists and is PUBLIC; refusing to upload")
    stats = json.loads(args.stats.read_text())

    if not args.card_only:
        rows = read_jsonl(args.final)
        sources = yaml.safe_load((REPO_ROOT / "configs" / "data_pipeline.yaml").read_text())["sources"]
        splits = {}
        for split in ("train", "validation"):
            sub = [to_row(r, sources) for r in rows if r["split"] == split]
            # Build audio as a plain {bytes, path} struct and cast: datasets' Audio.encode_example
            # imports torchcodec (FFmpeg) even for ready-made bytes; the Arrow cast needs neither.
            raw = Features({**FEATURES, "audio": {"bytes": Value("binary"), "path": Value("string")}})
            splits[split] = Dataset.from_list(sub, features=raw).cast_column("audio", FEATURES["audio"])
            logger.info(f"{split}: {len(sub)} rows")
        DatasetDict(splits).push_to_hub(args.repo, private=True, token=token, max_shard_size="400MB",
                                        commit_message="Session 1: curated human-verified Hinglish TTS dataset (24 kHz)")

    api.upload_file(path_or_fileobj=str(args.stats), path_in_repo="dataset_stats.json",
                    repo_id=args.repo, repo_type="dataset", commit_message="dataset_stats.json")
    # dataset card: keep the dataset_info/configs metadata push_to_hub generated, add ours
    card = DatasetCard.load(args.repo, token=token)
    meta = card.data.to_dict()
    meta.update({
        "pretty_name": "IISc Hinglish Pocket TTS Dataset (human-verified, 24 kHz, dual-script)",
        "language": ["hi", "en"],
        "license": "other",
        "license_name": "mixed-source-licenses",
        "license_link": f"https://huggingface.co/datasets/{args.repo}/blob/main/LICENSE_NOTES.md",
        "task_categories": ["text-to-speech"],
        "tags": ["hinglish", "code-switching", "hindi", "indian-english", "devanagari", "romanized-hindi",
                 "tts", "pocket-tts", "speech"],
        "size_categories": ["1K<n<10K"],
    })
    examples = _examples(read_jsonl(args.final))
    body = render_card(stats, args.repo, examples)
    api.upload_file(path_or_fileobj=io.BytesIO(_license_notes(stats).encode()), path_in_repo="LICENSE_NOTES.md",
                    repo_id=args.repo, repo_type="dataset", commit_message="Per-source licence notes")
    new_card = DatasetCard(f"---\n{_yaml(meta)}---\n{body}")
    new_card.push_to_hub(args.repo, token=token, commit_message="Dataset card (sources, provenance, preprocessing)")
    logger.info(f"pushed https://huggingface.co/datasets/{args.repo} (private)")


def _examples(rows: list[dict]) -> list[dict]:
    """One validation example per source (code-switched ones preferred) for the card."""
    out = []
    for key in ("indictts_hindi", "syspin_hindi_male", "indictts_english", "openslr104_hinglish"):
        cands = sorted((r for r in rows if r["source_key"] == key and r["split"] == "validation"),
                       key=lambda r: (r["language"] != "hi-en", len(r["text_roman"]) > 110, r["utt_id"]))
        if cands:
            out.append(cands[0])
    return out


def _yaml(meta: dict) -> str:
    import yaml

    return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)


def _license_notes(stats: dict) -> str:
    import yaml as _y

    cfg = _y.safe_load((REPO_ROOT / "configs" / "data_pipeline.yaml").read_text())
    lines = ["# Licence notes", "",
             "This dataset is a derivative of several public corpora; each row's `source_license` column "
             "names the terms of the corpus it came from. Redistribution and use must follow the terms of "
             "each source corpus.", ""]
    for key, src in cfg["sources"].items():
        n = stats["by_source"].get(key, {}).get("utterances", 0)
        lines.append(f"- **{src['repo']}** ({n} utterances): {src['license']}. Origin: {src['origin']}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
