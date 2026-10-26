"""Download the raw source files listed in configs/data_pipeline.yaml into the HF cache.

Only the shards that hold the selected speaker groups are fetched (IndicTTS-English shards are
sorted by speaker group, so 7 of its 28 shards cover everything we keep). Resumable: files
already in the cache are not downloaded again.

    python -m data_pipeline.download_sources [--config configs/data_pipeline.yaml]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from data_pipeline.common import DATA_ROOT, REPO_ROOT, hf_token, setup_logging, write_json

logger = setup_logging("download_sources")


def resolve_files(api: HfApi, repo: str, files: list[str] | str) -> list[str]:
    if files == "all":
        return sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    return list(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data_pipeline.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    token = hf_token("write")
    api = HfApi(token=token)

    jobs = []
    for name, src in cfg["sources"].items():
        for f in resolve_files(api, src["repo"], src["files"]):
            jobs.append((name, src["repo"], f))
    logger.info(f"{len(jobs)} files to fetch")

    def fetch(job: tuple[str, str, str]) -> tuple[str, str, str, str, float]:
        name, repo, f = job
        t0 = time.time()
        path = hf_hub_download(repo, f, repo_type="dataset", token=token)
        return name, repo, f, path, time.time() - t0

    manifest: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, repo, f, path, dt in pool.map(fetch, jobs):
            size_mb = Path(path).stat().st_size / 1e6
            logger.info(f"{name:22s} {f:40s} {size_mb:8.0f} MB  {dt:6.1f}s")
            manifest.setdefault(name, {})[f] = path
    out = DATA_ROOT / "raw" / "download_manifest.json"
    write_json(out, manifest)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
