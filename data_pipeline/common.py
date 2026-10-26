"""Shared helpers: environment/token loading, paths, logging, JSON IO."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# On the GPU workstation everything heavy lives under $WORK (see env.sh there); locally it falls
# back to the repo's git-ignored outputs/ folder so scripts can be smoke-tested anywhere.
WORK = Path(os.environ.get("WORK", REPO_ROOT / "outputs"))
DATA_ROOT = Path(os.environ.get("HINGLISH_DATA_ROOT", WORK / "data"))


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a KEY=VALUE .env file without overriding existing variables."""
    for candidate in [path] if path else [REPO_ROOT / ".env", WORK / ".env"]:
        if candidate is None or not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def hf_token(kind: str = "write") -> str:
    """HF_WRITE_TOKEN or HF_READ_TOKEN from the environment / .env (never logged)."""
    load_dotenv()
    var = {"write": "HF_WRITE_TOKEN", "read": "HF_READ_TOKEN"}[kind]
    token = os.environ.get(var)
    if not token:
        raise SystemExit(f"{var} is not set (expected in .env)")
    return token


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    return logging.getLogger(name)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
