"""Word-level forced alignment and Pocket TTS training manifests.

The pocket-tts trainer cuts every utterance at a random word boundary: audio before the cut is the
voice prompt, audio after it is the target, paired with the remaining words (training/dataloader/
loader.py). It therefore needs word timestamps ("words": [{"word", "start", "end"}]).

Alignment reuses the official CTC trellis (pocket-tts training/scripts/align_data.py):
  * hi / hi-en utterances: the all-Devanagari spoken form with a Hindi char-level wav2vec2 CTC
    (Harveenchadha/vakyansh-wav2vec2-hindi-him-4200; its vocabulary has ॉ ॅ and the nukta, so
    English words written in Devanagari align too);
  * en utterances: the English text with facebook/wav2vec2-large-960h-lv60-self.
Word i of the Devanagari, Roman and code-switched spoken forms is the same spoken word
(hinglish_text_processor guarantees it), so one alignment yields word timings for all three.

Writes, under $WORK/data/manifests/:
  {split}_{deva,roman,cs}.jsonl            audio manifests (path, start, duration, transcript, speaker)
  {split}_{deva,roman,cs}_aligned.jsonl    the same with "words"
  {split}_mix_aligned.jsonl                all three scripts (identical renderings kept once)
  alignment_report.json

    python -m data_pipeline.align_words [--batch-size 16]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

from data_pipeline.common import DATA_ROOT, read_jsonl, setup_logging, write_json, write_jsonl

sys.path.insert(0, os.environ.get("POCKET_TTS_REPO", str(Path.home() / "vaani_pocket_tts_work" / "pocket-tts")))
from training.scripts.align_data import _load_ctc_model, _tokens_for, batched_word_spans  # noqa: E402

logger = setup_logging("align_words")

MODELS = {
    "hi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
    "en": "facebook/wav2vec2-large-960h-lv60-self",
}
SCRIPTS = {"deva": "spoken_devanagari", "roman": "spoken_roman", "cs": "spoken_code_switched"}


def align_group(rows: list[dict], text_key: str, model_name: str, device: torch.device,
                batch_size: int) -> dict[str, list[tuple[float, float] | None] | None]:
    """utt_id -> per-word (start, end) seconds (None for words the CTC vocabulary cannot see),
    or None when the whole utterance failed to align."""
    model, vocab, blank, delim, fold, sr, use_bf16 = _load_ctc_model(model_name, device)
    out: dict[str, list | None] = {}
    rows = sorted(rows, key=lambda r: r["duration_s"])
    t0 = time.time()
    for b0 in range(0, len(rows), batch_size):
        chunk = rows[b0:b0 + batch_size]
        prepared = []
        for r in chunk:
            wav, in_sr = sf.read(r["audio_path"], dtype="float32")
            wav = soxr.resample(wav, in_sr, sr, quality="HQ").astype(np.float32)
            words = r[text_key].split()
            norm = ["".join(c for c in fold(w) if c in vocab and c != "|") for w in words]
            tokens, word_of = _tokens_for([w for w in norm if w], vocab, delim)
            prepared.append((r, wav, norm, tokens, word_of))
        ok = [p for p in prepared if p[3]]
        for p in prepared:
            if not p[3]:
                out[p[0]["utt_id"]] = None
        if not ok:
            continue
        lens = [len(p[1]) for p in ok]
        x = torch.zeros(len(ok), max(lens))
        for i, p in enumerate(ok):
            x[i, : lens[i]] = torch.from_numpy(p[1])
        attn = (torch.arange(max(lens))[None, :] < torch.tensor(lens)[:, None]).long()
        with torch.no_grad():
            logits = model(x.to(device, torch.bfloat16 if use_bf16 else torch.float32),
                           attention_mask=attn.to(device)).logits.float()
        emissions = logits.log_softmax(-1)
        T = torch.tensor([model._get_feat_extract_output_lengths(n) for n in lens], device=device)
        spans = batched_word_spans(emissions, T, [p[3] for p in ok], [p[4] for p in ok], blank)
        for p, sp, t_frames, n in zip(ok, spans, T.tolist(), lens):
            r, _, norm, _, _ = p
            if sp is None:
                out[r["utt_id"]] = None
                continue
            sec = (n / sr) / t_frames
            timed, k = [], 0
            for nw in norm:
                if not nw:
                    timed.append(None)
                    continue
                s, e = sp[k]
                k += 1
                timed.append((round(s * sec, 3), round((e + 1) * sec, 3)))
            out[r["utt_id"]] = timed
        if (b0 // batch_size) % 50 == 0:
            logger.info(f"{model_name.split('/')[-1]}: {min(b0 + batch_size, len(rows))}/{len(rows)} "
                        f"({time.time() - t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return out


def manifest_row(r: dict, script: str, timings: list | None) -> dict:
    words = r[SCRIPTS[script]].split()
    row = {
        "path": r["audio_path"], "start": 0.0, "duration": r["duration_s"],
        "transcript": " ".join(words), "speaker": r["speaker_id"],
        "utt_id": r["utt_id"], "script": script, "language": r["language"],
    }
    if timings is not None:
        row["words"] = [{"word": w, "start": t[0] if t else None, "end": t[1] if t else None}
                        for w, t in zip(words, timings)]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path, default=DATA_ROOT / "processed" / "selected.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "manifests")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-lead-s", type=float, default=0.6, help="first word must start before this")
    ap.add_argument("--max-tail-s", type=float, default=0.8, help="last word must end within this of the end")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl(args.selected)
    groups = {"en": [r for r in rows if r["language"] == "en"],
              "hi": [r for r in rows if r["language"] != "en"]}
    timings: dict[str, list | None] = {}
    t0 = time.time()
    for lang, grp in groups.items():
        text_key = "spoken_roman" if lang == "en" else "spoken_devanagari"
        logger.info(f"aligning {len(grp)} {lang} utterances on {text_key} with {MODELS[lang]}")
        timings.update(align_group(grp, text_key, MODELS[lang], device, args.batch_size))

    report = collections.Counter()
    leads, tails, word_cov = [], [], []
    kept: list[dict] = []
    for r in rows:
        t = timings.get(r["utt_id"])
        if t is None:
            report["failed_alignment"] += 1
            continue
        timed = [x for x in t if x is not None]
        if len(timed) < 2:
            report["fewer_than_2_aligned_words"] += 1
            continue
        lead, tail = timed[0][0], r["duration_s"] - timed[-1][1]
        if lead > args.max_lead_s or tail > args.max_tail_s or tail < -0.05:
            report["implausible_word_boundaries"] += 1
            continue
        leads.append(lead)
        tails.append(tail)
        word_cov.append(len(timed) / len(t))
        r["_timings"] = t
        kept.append(r)
    report["kept"] = len(kept)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        sub = [r for r in kept if r["split"] == split]
        tag = "valid" if split == "validation" else "train"
        mix: list[dict] = []
        for script in SCRIPTS:
            write_jsonl(args.out / f"{tag}_{script}.jsonl", [manifest_row(r, script, None) for r in sub])
            write_jsonl(args.out / f"{tag}_{script}_aligned.jsonl", [manifest_row(r, script, r["_timings"]) for r in sub])
        for r in sub:
            seen = set()
            for script in SCRIPTS:
                row = manifest_row(r, script, r["_timings"])
                if row["transcript"] not in seen:
                    seen.add(row["transcript"])
                    mix.append(row)
        write_jsonl(args.out / f"{tag}_mix_aligned.jsonl", mix)
        report[f"{tag}_utterances"] = len(sub)
        report[f"{tag}_mix_rows"] = len(mix)
    q = lambda a, p: round(float(np.percentile(a, p)), 3)  # noqa: E731
    summary = {
        "counts": dict(report), "models": MODELS,
        "first_word_start_s": {"p05": q(leads, 5), "median": q(leads, 50), "p95": q(leads, 95)},
        "last_word_end_to_file_end_s": {"p05": q(tails, 5), "median": q(tails, 50), "p95": q(tails, 95)},
        "aligned_word_fraction": {"mean": round(float(np.mean(word_cov)), 4), "min": round(float(np.min(word_cov)), 4)},
        "seconds": round(time.time() - t0, 1),
    }
    write_json(args.out / "alignment_report.json", summary)
    logger.info(f"alignment summary: {summary}")


if __name__ == "__main__":
    main()
