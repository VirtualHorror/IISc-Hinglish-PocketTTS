"""Publish the release folder as the PRIVATE Hugging Face model repo (write token from .env).

Creates MSRIT-NBM/pocket-tts-indian-hinglish with private=True (or reuses it), refuses to continue
if the repo is not private, uploads the folder in one commit, and writes release/push_report.json
(commit, file list with sizes).

    python -m release.push_model [--folder $WORK/release] [--repo MSRIT-NBM/pocket-tts-indian-hinglish]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from data_pipeline.common import REPO_ROOT, WORK, hf_token, load_dotenv, setup_logging, write_json

logger = setup_logging("push_model")

REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
IGNORE = ["export_check.wav", "*/export_check.wav", "**/__pycache__/**", "*.pyc"]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=WORK / "release")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--message", default="Pocket TTS Indian Hinglish: 24-layer fine-tune, voices, inference helper, samples, model card")
    args = ap.parse_args()
    for required in ("model.safetensors", "config.yaml", "tokenizer.model", "inference.py", "README.md", "voices/index.json"):
        if not (args.folder / required).exists():
            raise SystemExit(f"{args.folder / required} is missing: export and render the card first")

    api = HfApi(token=hf_token("write"))
    user = api.whoami()["name"]
    url = api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(args.repo)
    if not info.private:
        raise SystemExit(f"{args.repo} exists and is PUBLIC; refusing to upload (make it private first)")
    commit = api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(args.folder),
                               ignore_patterns=IGNORE, commit_message=args.message)
    info = api.model_info(args.repo, files_metadata=True)
    assert info.private, "repo became public"
    files = {s.rfilename: s.size for s in info.siblings}
    report = {"repo": args.repo, "url": str(url), "private": info.private, "uploader": user,
              "commit": commit.oid, "commit_url": commit.commit_url, "n_files": len(files),
              "total_mb": round(sum(v or 0 for v in files.values()) / 1e6, 1), "files": files}
    write_json(REPO_ROOT / "release" / "push_report.json", report)
    logger.info(f"pushed {len(files)} files ({report['total_mb']} MB) to {args.repo} (private={info.private}) "
                f"at {commit.oid[:10]}")


if __name__ == "__main__":
    main()
