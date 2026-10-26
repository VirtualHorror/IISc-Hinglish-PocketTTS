"""Curate the human-verified Hinglish TTS dataset from the downloaded sources.

Pass 1 (parallel, nothing written): every utterance of the configured speaker groups gets its
  transcript annotated in three scripts (hinglish_text_processor) and its audio processed
  (audio_processing: 24 kHz mono, 0.10 s lead / 0.18 s end-on-pause tail, -20 dBFS, quality metrics).
  Text and audio filters record a rejection reason or mark the utterance as a candidate.
Pass 2: per-group quotas from an SNR-ranked candidate pool (seeded sampling, per-speaker caps),
  then a stratified train/validation split.
Pass 3 (parallel): re-decode only the selected utterances and write 24 kHz PCM-16 WAVs.

Outputs (under $WORK/data/processed): audio/<utt_id>.wav, selected.jsonl, candidates.jsonl,
rejections.json; summary metrics -> data_pipeline/dataset_stats.json.

    python -m data_pipeline.prepare_dataset [--config configs/data_pipeline.yaml] [--workers 24]
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os
import random
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml

from data_pipeline.audio_processing import AudioConfig, process_utterance
from data_pipeline.common import DATA_ROOT, REPO_ROOT, read_jsonl, setup_logging, write_json, write_jsonl
from data_pipeline.hinglish_text_processor import HinglishTextProcessor, tokenize, clean_text, cmu_pronunciation

logger = setup_logging("prepare_dataset")

INDICTTS_PATH = re.compile(r"train_([a-z]+?)(female|male)_(\d+)\.wav$")
DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# ----------------------------------------------------------------------------------------------
# Worker state (one text processor per process)
# ----------------------------------------------------------------------------------------------

_PROC: HinglishTextProcessor | None = None
_ACFG: AudioConfig | None = None
_TCFG: dict[str, Any] = {}


def _init_worker(audio_cfg: dict, text_cfg: dict, target_sr: int) -> None:
    global _PROC, _ACFG, _TCFG
    _PROC = HinglishTextProcessor()
    _ACFG = AudioConfig.from_yaml(audio_cfg, target_sr)
    _TCFG = text_cfg


def _unpronounceable(text: str) -> bool:
    """Latin tokens a TTS reader cannot pronounce reliably: digits glued to letters (test1) or
    vowel-less non-dictionary strings longer than an acronym (gchmpnt)."""
    for tok in tokenize(clean_text(text)):
        core = tok.core
        if LATIN.search(core) and re.search(r"\d", core):
            return True
        if tok.kind == "latn" and not re.search(r"[aeiouyAEIOUY]", core) and len(core) > 5 \
                and cmu_pronunciation(core.lower()) is None:
            return True
    return False


def _text_checks(item: dict) -> tuple[dict | None, str]:
    text = item["text"]
    if not text or not text.strip():
        return None, "empty_text"
    if item.get("require_both_scripts") and not (DEVA.search(text) and LATIN.search(text)):
        return None, "not_code_switched"
    if item.get("no_digits") and re.search(r"\d", text):
        return None, "digits_in_code_switched_text"
    if _unpronounceable(text):
        return None, "unpronounceable_token"
    try:
        ann = _PROC.annotate(text)
    except Exception as exc:  # noqa: BLE001 -- recorded as a rejection reason
        return None, f"text_error:{type(exc).__name__}"
    if len(ann.spoken_words) < _TCFG.get("min_words", 2):
        return None, "too_few_words"
    return ann.__dict__, ""


def _measure(item: dict, wav: np.ndarray, sr: int, keep_audio: bool) -> dict:
    rec = {k: v for k, v in item.items() if k not in ("text",)}
    rec["text_original"] = item["text"]
    ann, reason = _text_checks(item)
    if ann is None:
        rec.update(ok=False, reason=reason)
        return rec
    rec.update({k: ann[k] for k in ("text_devanagari", "text_roman", "text_code_switched",
                                    "spoken_devanagari", "spoken_roman", "spoken_code_switched",
                                    "language", "n_english_words", "n_hindi_words")})
    rec["n_words"] = len(ann["spoken_words"])
    proc = process_utterance(wav, sr, _ACFG)
    rec.update(proc.metrics())
    if proc.ok:
        letters = len(re.findall(r"[A-Za-z]", ann["spoken_roman"]))
        cps = letters / max(proc.duration_s - _ACFG.lead_silence_s - _ACFG.trail_silence_s, 0.1)
        rec["chars_per_sec"] = round(cps, 2)
        if not (_TCFG.get("min_chars_per_sec", 0) <= cps <= _TCFG.get("max_chars_per_sec", 1e9)):
            rec.update(ok=False, reason="speaking_rate_outlier")
    if keep_audio and rec["ok"]:
        rec["_audio"] = proc.audio
    return rec


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return wav, sr


# ----------------------------------------------------------------------------------------------
# Work units
# ----------------------------------------------------------------------------------------------


def _parquet_unit(unit: dict) -> list[dict]:
    """Measure (pass 1) or write (pass 3) the wanted rows of one parquet row group."""
    pf = pq.ParquetFile(unit["file"])
    table = pf.read_row_group(unit["row_group"], columns=["audio", unit["text_col"]])
    audio = table.column("audio").combine_chunks()
    texts = table.column(unit["text_col"]).to_pylist()
    out = []
    for row, item in unit["rows"].items():
        item = dict(item, text=texts[row])
        wav, sr = _decode(audio[row]["bytes"].as_py())
        out.append(_finish_unit_row(unit, item, wav, sr))
    return out


def _files_unit(unit: dict) -> list[dict]:
    out = []
    for item in unit["items"]:
        wav, sr = sf.read(item["source_file"], dtype="float32", always_2d=False)
        out.append(_finish_unit_row(unit, dict(item), wav, sr))
    return out


def _finish_unit_row(unit: dict, item: dict, wav: np.ndarray, sr: int) -> dict:
    write = unit.get("write_dir")
    rec = _measure(item, wav, sr, keep_audio=bool(write))
    if write and rec.get("ok"):
        path = Path(write) / f"{item['utt_id']}.wav"
        sf.write(path, rec.pop("_audio"), _ACFG.target_sr, subtype="PCM_16")
        rec["audio_path"] = str(path)
    rec.pop("_audio", None)
    return rec


# ----------------------------------------------------------------------------------------------
# Source enumeration (cheap: metadata only)
# ----------------------------------------------------------------------------------------------


def enumerate_parquet_source(key: str, src: dict, files: dict[str, str], rng: random.Random) -> list[dict]:
    """Group rows by (file, row_group) for the configured speaker groups, capping candidates."""
    text_col = "text" if src["repo"].startswith("SPRINGLab") else "transcription"
    per_group: dict[str, list[tuple]] = collections.defaultdict(list)
    for fname, path in sorted(files.items()):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            cols = ["audio.path"] + (["file_name"] if text_col == "transcription" else [])
            t = pf.read_row_group(rg, columns=cols)
            col = t.column(0).combine_chunks()
            paths = col.field("path").to_pylist() if hasattr(col.type, "num_fields") and col.type.num_fields else col.to_pylist()
            names = t.column("file_name").to_pylist() if "file_name" in t.column_names else paths
            for row, (p, name) in enumerate(zip(paths, names)):
                if "speaker" in src:  # single-speaker source (SYSPIN)
                    group = "all"
                    speaker, gender = src["speaker"], ("female" if "female" in src["speaker"] else "male")
                    source_utt_id = name or p
                else:
                    m = INDICTTS_PATH.search(p or "")
                    if not m:
                        continue
                    group = m.group(1) + m.group(2)
                    speaker, gender = f"indictts_{m.group(1)}_{m.group(2)}", m.group(2)
                    source_utt_id = p
                if group not in src["quota"]:
                    continue
                per_group[group].append((path, rg, row, speaker, gender, source_utt_id))
    units: dict[tuple, dict] = {}
    for group, rows in per_group.items():
        cap = max(6 * src["quota"][group], 400)
        if len(rows) > cap:
            rows = rng.sample(rows, cap)
        for path, rg, row, speaker, gender, suid in rows:
            u = units.setdefault((path, rg), {"kind": "parquet", "file": path, "row_group": rg,
                                              "text_col": text_col, "rows": {}})
            lang_tag = src["language"]
            stem = re.sub(r"\W+", "_", Path(str(suid)).stem)
            u["rows"][row] = {
                "utt_id": f"{key}_{stem}", "source_key": key, "quota_group": group,
                "speaker_id": speaker, "gender": gender, "source_utt_id": str(suid),
                "source_dataset": src["repo"], "source_url": src["url"], "source_license": src["license"],
                "source_language": lang_tag, "human_verified": True,
            }
    return list(units.values())


def enumerate_openslr(key: str, src: dict, files: dict[str, str], raw_dir: Path, chunk: int = 64) -> list[dict]:
    extract_dir = raw_dir / "openslr104"
    marker = extract_dir / ".extracted"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"extracting {files['data/train.tar.gz']} -> {extract_dir} (one-time)")
        subprocess.run(["tar", "-xzf", files["data/train.tar.gz"], "-C", str(extract_dir)], check=True)
        marker.touch()
    items = []
    with open(files["data/metadata.csv"], encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].endswith(".wav"):
                continue
            rel, text = row[0], ",".join(row[1:])
            path = extract_dir / rel
            if not path.exists():
                continue
            stem = Path(rel).stem
            recording = stem.rsplit("_", 1)[0]
            items.append({
                "utt_id": f"{key}_{stem}", "source_key": key, "quota_group": "all",
                "speaker_id": f"openslr104_{recording}", "gender": "unknown", "source_utt_id": rel,
                "source_dataset": src["repo"], "source_url": src["url"], "source_license": src["license"],
                "source_language": src["language"], "human_verified": True, "source_file": str(path),
                "text": text, "require_both_scripts": bool(src.get("require_both_scripts")),
                "no_digits": True,
            })
    return [{"kind": "files", "items": items[i:i + chunk]} for i in range(0, len(items), chunk)]


# ----------------------------------------------------------------------------------------------
# Selection and split
# ----------------------------------------------------------------------------------------------


def select(cands: list[dict], cfg: dict, rng: random.Random) -> list[dict]:
    chosen: list[dict] = []
    by_group: dict[tuple, list[dict]] = collections.defaultdict(list)
    for c in cands:
        by_group[(c["source_key"], c["quota_group"])].append(c)
    for (key, group), rows in sorted(by_group.items()):
        src = cfg["sources"][key]
        quota = src["quota"][group]
        cap = src.get("max_per_speaker")
        if cap:  # spread over many speakers: best-SNR utterances of each speaker first
            per_spk: dict[str, list[dict]] = collections.defaultdict(list)
            for r in rows:
                per_spk[r["speaker_id"]].append(r)
            rows = [r for spk in sorted(per_spk) for r in sorted(per_spk[spk], key=lambda r: -r["snr_db"])[:cap]]
        rows = sorted(rows, key=lambda r: -r["snr_db"])
        pool = rows[: max(quota, math.ceil(1.5 * quota))]
        take = rng.sample(pool, min(quota, len(pool)))
        logger.info(f"select {key}/{group}: {len(rows)} candidates -> pool {len(pool)} -> {len(take)}")
        chosen += take
    return chosen


def assign_splits(rows: list[dict], frac: float, min_val: int, rng: random.Random) -> None:
    by_group: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_group[(r["source_key"], r["quota_group"])].append(r)
    for _, grp in sorted(by_group.items()):
        rng.shuffle(grp)
        n_val = min(max(min_val, round(frac * len(grp))), len(grp) // 5)
        for i, r in enumerate(grp):
            r["split"] = "validation" if i < n_val else "train"


# ----------------------------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------------------------


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {"min": round(float(a.min()), 3), "p05": round(float(np.percentile(a, 5)), 3),
            "median": round(float(np.median(a)), 3), "mean": round(float(a.mean()), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}


def build_stats(selected: list[dict], all_recs: list[dict], cfg: dict, timing: dict) -> dict:
    def hours(rows: list[dict]) -> float:
        return round(sum(r["duration_s"] for r in rows) / 3600, 3)

    def counts(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == k]
            out[k] = {"utterances": len(sub), "hours": hours(sub)}
        return out

    rej = collections.Counter((r["source_key"], r["reason"]) for r in all_recs if not r.get("ok"))
    rejections: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for (key, reason), n in sorted(rej.items()):
        rejections[key][reason] = n
    measured = collections.Counter(r["source_key"] for r in all_recs)
    passed = collections.Counter(r["source_key"] for r in all_recs if r.get("ok"))
    cs_mixed = [r for r in selected if DEVA.search(r["text_code_switched"]) and LATIN.search(r["text_code_switched"])]
    return {
        "dataset": "iisc-hinglish-pocket-tts-dataset",
        "generated_by": "data_pipeline/prepare_dataset.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sample_rate": cfg["target_sample_rate"],
        "total": {"utterances": len(selected), "hours": hours(selected),
                  "speakers": len({r["speaker_id"] for r in selected})},
        "splits": counts(selected, "split"),
        "by_language": counts(selected, "language"),
        "by_source": counts(selected, "source_key"),
        "by_speaker": counts(selected, "speaker_id"),
        "by_gender": counts(selected, "gender"),
        "code_switched_text_rows": {"utterances": len(cs_mixed), "hours": hours(cs_mixed),
                                    "note": "text_code_switched contains both Devanagari and Latin words"},
        "duration_s": _dist([r["duration_s"] for r in selected]),
        "snr_db": _dist([r["snr_db"] for r in selected]),
        "active_rms_dbfs": _dist([r["rms_dbfs"] for r in selected]),
        "peak_dbfs": _dist([r["peak_dbfs"] for r in selected]),
        "peak_limited_fraction": round(sum(r["peak_limited"] for r in selected) / max(1, len(selected)), 4),
        "chars_per_sec": _dist([r["chars_per_sec"] for r in selected]),
        "words_per_utterance": _dist([r["n_words"] for r in selected]),
        "source_sample_rates": dict(collections.Counter(str(r["source_sample_rate"]) for r in selected)),
        "max_clipped_samples": max((r["clipped_samples"] for r in selected), default=0),
        "filtering": {"measured": dict(measured), "passed_filters": dict(passed), "rejections": dict(rejections)},
        "config": {"audio": cfg["audio"], "text": cfg["text"],
                   "validation_fraction": cfg["validation_fraction"]},
        "timing_s": timing,
    }


# ----------------------------------------------------------------------------------------------


def run_units(units: list[dict], pool: ProcessPoolExecutor, desc: str) -> list[dict]:
    fn = {"parquet": _parquet_unit, "files": _files_unit}
    out: list[dict] = []
    t0 = time.time()
    futures = [pool.submit(fn[u["kind"]], u) for u in units]
    for i, fut in enumerate(futures, 1):
        out += fut.result()
        if i % max(1, len(futures) // 10) == 0 or i == len(futures):
            logger.info(f"{desc}: {i}/{len(futures)} units, {len(out)} utterances, {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data_pipeline.yaml")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 6))
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "processed")
    ap.add_argument("--stats", type=Path, default=REPO_ROOT / "data_pipeline" / "dataset_stats.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rng = random.Random(cfg["seed"])
    manifest = json.loads((DATA_ROOT / "raw" / "download_manifest.json").read_text())
    timing: dict[str, float] = {}

    t0 = time.time()
    units: list[dict] = []
    for key, src in cfg["sources"].items():
        if src["kind"] == "parquet":
            units += enumerate_parquet_source(key, src, manifest[key], rng)
        else:
            units += enumerate_openslr(key, src, manifest[key], DATA_ROOT / "raw")
    n_items = sum(len(u.get("rows", u.get("items", []))) for u in units)
    timing["enumerate"] = round(time.time() - t0, 1)
    logger.info(f"{n_items} utterances to measure in {len(units)} work units")

    init = (_init_worker, (cfg["audio"], cfg["text"], cfg["target_sample_rate"]))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init[0], initargs=init[1]) as pool:
        t0 = time.time()
        recs = run_units(units, pool, "pass 1 (measure)")
        timing["pass1_measure"] = round(time.time() - t0, 1)
        args.out.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out / "candidates.jsonl", recs)

        cands = [r for r in recs if r.get("ok")]
        selected = select(cands, cfg, rng)
        assign_splits(selected, cfg["validation_fraction"], cfg["min_validation_per_group"], rng)
        chosen = {r["utt_id"]: r for r in selected}

        # pass 3: rebuild work units holding only the chosen rows, and write their audio
        audio_dir = args.out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        write_units = []
        for u in units:
            if u["kind"] == "parquet":
                rows = {k: v for k, v in u["rows"].items() if v["utt_id"] in chosen}
                if rows:
                    write_units.append(dict(u, rows=rows, write_dir=str(audio_dir)))
            else:
                items = [it for it in u["items"] if it["utt_id"] in chosen]
                if items:
                    write_units.append(dict(u, items=items, write_dir=str(audio_dir)))
        t0 = time.time()
        written = run_units(write_units, pool, "pass 3 (write)")
        timing["pass3_write"] = round(time.time() - t0, 1)

    final = []
    for rec in written:
        if not rec.get("ok"):
            raise RuntimeError(f"re-processing changed the verdict for {rec['utt_id']}: {rec.get('reason')}")
        rec["split"] = chosen[rec["utt_id"]]["split"]
        rec.pop("require_both_scripts", None)
        rec.pop("no_digits", None)
        final.append(rec)
    final.sort(key=lambda r: (r["split"], r["source_key"], r["utt_id"]))
    write_jsonl(args.out / "selected.jsonl", final)
    rejections = collections.Counter(f"{r['source_key']}:{r['reason']}" for r in recs if not r.get("ok"))
    write_json(args.out / "rejections.json", dict(sorted(rejections.items())))
    stats = build_stats(final, recs, cfg, timing)
    write_json(args.stats, stats)
    logger.info(f"selected {len(final)} utterances ({stats['total']['hours']} h, "
                f"{stats['total']['speakers']} speakers) -> {args.out / 'selected.jsonl'}")
    logger.info(f"stats -> {args.stats}")


if __name__ == "__main__":
    main()
