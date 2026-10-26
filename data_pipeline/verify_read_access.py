"""Verify the evaluators' READ-ONLY token can access the private dataset (and that it is private).

Uses only HF_READ_TOKEN, explicitly: HF_TOKEN is removed from this process so nothing can fall
back to the write token. Checks: token role, anonymous access is refused, repo metadata and file
list, one sample via a raw parquet shard (decoded with soundfile), and one sample via
datasets.load_dataset(streaming=True).

    python -m data_pipeline.verify_read_access [--repo MSRIT-NBM/iisc-hinglish-pocket-tts-dataset]
"""

from __future__ import annotations

import argparse
import io
import os

os.environ.pop("HF_TOKEN", None)  # before huggingface_hub reads it
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError  # noqa: E402

from data_pipeline.common import REPO_ROOT, hf_token, load_dotenv, setup_logging, write_json  # noqa: E402

logger = setup_logging("verify_read_access")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO", "MSRIT-NBM/iisc-hinglish-pocket-tts-dataset"))
    ap.add_argument("--out", type=__import__("pathlib").Path, default=REPO_ROOT / "data_pipeline" / "read_access_verification.json")
    args = ap.parse_args()
    read = hf_token("read")
    api = HfApi(token=read)
    report: dict = {"repo": args.repo}

    who = api.whoami(token=read)
    report["token"] = {"user": who["name"], "role": who.get("auth", {}).get("accessToken", {}).get("role")}
    assert report["token"]["role"] == "read", report["token"]

    try:
        HfApi(token=False).dataset_info(args.repo)
        report["anonymous_access"] = "ALLOWED (repo is public!)"
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
        report["anonymous_access"] = f"refused ({type(exc).__name__})"

    info = api.dataset_info(args.repo, token=read)
    files = api.list_repo_files(args.repo, repo_type="dataset", token=read)
    report["metadata"] = {"private": info.private, "n_files": len(files), "files": files}
    assert info.private, "dataset repo is not private"

    shard = sorted(f for f in files if f.startswith("data/validation") and f.endswith(".parquet"))[0]
    path = hf_hub_download(args.repo, shard, repo_type="dataset", token=read)
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    report["parquet_sample"] = {
        "shard": shard, "rows_in_shard": table.num_rows, "utt_id": row["utt_id"],
        "sample_rate": sr, "decoded_seconds": round(len(wav) / sr, 3), "duration_s": round(row["duration_s"], 3),
        "text_devanagari": row["text_devanagari"], "text_roman": row["text_roman"],
        "text_code_switched": row["text_code_switched"], "speaker_id": row["speaker_id"],
        "source_dataset": row["source_dataset"], "human_verified": row["human_verified"],
    }

    from datasets import Audio, load_dataset

    ds = load_dataset(args.repo, split="validation", streaming=True, token=read)
    ex = next(iter(ds.cast_column("audio", Audio(decode=False))))
    wav2, sr2 = sf.read(io.BytesIO(ex["audio"]["bytes"]))
    report["load_dataset_streaming_sample"] = {"utt_id": ex["utt_id"], "sample_rate": sr2,
                                               "decoded_seconds": round(len(wav2) / sr2, 3),
                                               "columns": sorted(ex.keys())}
    report["result"] = "PASS: read-only token can list and download the private dataset; anonymous access refused"
    write_json(args.out, report)
    logger.info(report["result"])
    for k in ("token", "anonymous_access", "parquet_sample", "load_dataset_streaming_sample"):
        logger.info(f"{k}: {report[k]}")


if __name__ == "__main__":
    main()
