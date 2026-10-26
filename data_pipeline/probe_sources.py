"""Probe candidate Indian speech datasets on the Hugging Face Hub.

For each candidate: does it exist, is it gated, can our token actually read it, what licence
does its card declare, which configs/splits/features does the dataset viewer report, and what
do the first transcripts look like. The result (data_pipeline/source_probe.json) documents why
each source was kept or dropped, so the selection in configs/data_pipeline.yaml is auditable.

    python -m data_pipeline.probe_sources [--out data_pipeline/source_probe.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from data_pipeline.common import REPO_ROOT, hf_token, setup_logging, write_json

logger = setup_logging("probe_sources")

VIEWER = "https://datasets-server.huggingface.co"

# Human-verified Indian speech candidates (TTS-grade first), plus Hinglish-tagged sets that
# need vetting for provenance (synthetic or ASR-labelled audio is rejected).
CANDIDATES = [
    "SPRINGLab/IndicTTS-Hindi",
    "SPRINGLab/IndicTTS-English",
    "ai4bharat/Rasa",
    "SYSPIN/LIMMITS24_target_speaker_fewshot_samples",
    "SayantanJoker/SYSPIN_Hindi_Female_TTS",
    "SayantanJoker/SYSPIN_Hindi_Male_TTS",
    "google/fleurs",
    "ai4bharat/indicvoices_r",
    "ai4bharat/Svarah",
    "ai4bharat/MANGO",
    "ai4bharat/IndicCMix",
    "ARTPARK-IISc/Vaani-transcription-part",
    "ujs/hinglish",
    "HumynLabs/banking-customersupport-hinglish-audio",
    "taresh18/indic-speech",
]

TEXT_HINTS = ("text", "sentence", "transcri", "normalized", "raw", "verbatim", "speaker", "gender",
              "style", "lang", "emotion", "category", "duration", "domain", "id")


def viewer_get(endpoint: str, token: str, **params: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{VIEWER}/{endpoint}", params=params,
                         headers={"Authorization": f"Bearer {token}"}, timeout=60)
    except requests.RequestException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    return r.json()


def summarize_rows(first_rows: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    out = []
    for row in (first_rows.get("rows") or [])[:n]:
        cells = row.get("row", {})
        keep = {}
        for k, v in cells.items():
            if isinstance(v, (str, int, float, bool)) and any(h in k.lower() for h in TEXT_HINTS):
                keep[k] = v if not isinstance(v, str) else v[:200]
            elif isinstance(v, list) and v and isinstance(v[0], dict) and "src" in v[0]:
                keep[k] = "<audio>"
        out.append(keep)
    return out


def probe(api: HfApi, repo_id: str, token: str) -> dict[str, Any]:
    res: dict[str, Any] = {"repo_id": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}"}
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError:
        res.update(exists=False, usable=False, reason="repository not found (or private)")
        return res
    except Exception as exc:  # noqa: BLE001 -- recorded verbatim in the probe report
        res.update(exists=None, usable=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        return res
    card = info.card_data.to_dict() if info.card_data else {}
    siblings = info.siblings or []
    res.update(
        exists=True,
        gated=info.gated,
        license=card.get("license"),
        languages=card.get("language"),
        task_categories=card.get("task_categories"),
        downloads=info.downloads,
        last_modified=str(info.last_modified),
        n_files=len(siblings),
        n_parquet=sum(s.rfilename.endswith(".parquet") for s in siblings),
        has_loading_script=any(s.rfilename.endswith(".py") for s in siblings),
        total_gb=round(sum((s.size or 0) for s in siblings) / 1e9, 3),
    )
    # Can this token actually read file contents? (gated repos list files but refuse downloads)
    readme = next((s.rfilename for s in siblings if s.rfilename.lower() == "readme.md"), None)
    probe_file = readme or (siblings[0].rfilename if siblings else None)
    try:
        if probe_file:
            api.hf_hub_download(repo_id, probe_file, repo_type="dataset",
                                local_dir=str(REPO_ROOT / "outputs" / "probe" / repo_id.replace("/", "__")))
        res["access_ok"] = True
    except GatedRepoError:
        res["access_ok"] = False
        res["reason"] = "gated: terms not accepted for this account"
    except Exception as exc:  # noqa: BLE001
        res["access_ok"] = False
        res["reason"] = f"{type(exc).__name__}: {exc}"[:300]

    viewer_info = viewer_get("info", token, dataset=repo_id)
    configs: dict[str, Any] = {}
    if viewer_info and "dataset_info" in viewer_info:
        for cfg, ci in viewer_info["dataset_info"].items():
            configs[cfg] = {
                "features": {k: (v.get("_type") if isinstance(v, dict) else str(v))
                             for k, v in (ci.get("features") or {}).items()},
                "splits": {s: v.get("num_examples") for s, v in (ci.get("splits") or {}).items()},
            }
    else:
        res["viewer_error"] = (viewer_info or {}).get("error")
    # Large multilingual repos: only inspect the configs relevant to us.
    wanted = [c for c in configs if any(t in c.lower() for t in ("hi", "en", "default", "hindi"))] or list(configs)
    for cfg in wanted[:4]:
        split = next(iter(configs[cfg]["splits"]), "train")
        rows = viewer_get("first-rows", token, dataset=repo_id, config=cfg, split=split)
        configs[cfg]["sample_rows"] = summarize_rows(rows) if rows and "rows" in rows else rows
    res["configs"] = {c: configs[c] for c in wanted[:4]} | {"_all_config_names": list(configs)[:60]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data_pipeline" / "source_probe.json")
    ap.add_argument("--only", nargs="*", help="probe only these repo ids")
    args = ap.parse_args()
    token = hf_token("write")
    api = HfApi(token=token)
    results = []
    for repo_id in args.only or CANDIDATES:
        logger.info(f"probing {repo_id}")
        results.append(probe(api, repo_id, token))
        r = results[-1]
        logger.info(f"  exists={r.get('exists')} gated={r.get('gated')} access={r.get('access_ok')} "
                    f"license={r.get('license')} files={r.get('n_files')} gb={r.get('total_gb')}")
    write_json(args.out, results)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
