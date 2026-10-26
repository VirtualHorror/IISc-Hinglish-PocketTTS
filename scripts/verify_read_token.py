#!/usr/bin/env python3
"""Read-only token check of both private Hugging Face repositories (the evaluators' access path).

With only the read token (HF_READ_TOKEN from .env, or --token-env NAME): the token's role is "read"; each repo
is private, lists its files and serves README.md and a weights / data file over HTTPS. Without a token, the Hub
API refuses both repos and the same HTTPS requests return 401. No token value is printed or stored.
release/verify_model_access.py is the full end-to-end check (fresh process, synthesis, Whisper round trip).

    python scripts/verify_read_token.py            # writes release/read_token_check.json
"""

from __future__ import annotations

import os

os.environ.pop("HF_TOKEN", None)  # before huggingface_hub reads it: only the read token is used
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import requests  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPOS = {"MSRIT-NBM/pocket-tts-indian-hinglish": "model", "MSRIT-NBM/iisc-hinglish-pocket-tts-dataset": "dataset"}


def env_token(name: str) -> str:
    value = os.environ.get(name)
    if not value and (REPO_ROOT / ".env").exists():
        for line in (REPO_ROOT / ".env").read_text().splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip()
    if not value:
        raise SystemExit(f"{name} not set (environment or .env)")
    return value


def status(url: str, token: str | None, head: bool = False) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return (requests.head if head else requests.get)(url, headers=headers, allow_redirects=False, timeout=60).status_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-env", default="HF_READ_TOKEN")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "release" / "read_token_check.json")
    cli = ap.parse_args()
    token = env_token(cli.token_env)
    api, anon = HfApi(token=token), HfApi(token=False)
    who = api.whoami()
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "token": {"user": who["name"], "role": who.get("auth", {}).get("accessToken", {}).get("role")},
              "repos": {}}
    ok = report["token"]["role"] == "read"
    for repo, kind in REPOS.items():
        info = api.repo_info(repo, repo_type=kind)
        files = api.list_repo_files(repo, repo_type=kind)
        sample = next((f for f in files if f.endswith((".safetensors", ".parquet"))), files[0])
        prefix = "datasets/" if kind == "dataset" else ""
        base = f"https://huggingface.co/{prefix}{repo}/resolve/main"
        api_url = f"https://huggingface.co/api/{kind}s/{repo}"
        try:
            anon.repo_info(repo, repo_type=kind)
            anonymous_api = "ALLOWED (public!)"
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
            anonymous_api = f"refused ({type(exc).__name__})"
        entry = {
            "type": kind, "private": info.private, "sha": info.sha, "n_files": len(files),
            "read_token": {"api": status(api_url, token), "README.md": status(f"{base}/README.md", token),
                           sample: status(f"{base}/{sample}", token, head=True)},
            "anonymous": {"api_client": anonymous_api, "api": status(api_url, None),
                          "README.md": status(f"{base}/README.md", None), sample: status(f"{base}/{sample}", None, head=True)},
        }
        entry["readme_session_mentions"] = sum(
            line.lower().count("session") for line in requests.get(f"{base}/README.md", headers={"Authorization": f"Bearer {token}"},
                                                                   timeout=60).text.splitlines())
        passed = (info.private and all(v in (200, 302, 307) for v in entry["read_token"].values())
                  and entry["anonymous"]["api_client"].startswith("refused")
                  and all(v in (401, 403, 404) for k, v in entry["anonymous"].items() if k != "api_client"))
        entry["result"] = "PASS" if passed else "FAIL"
        ok = ok and passed
        report["repos"][repo] = entry
    report["result"] = "PASS" if ok else "FAIL"
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
