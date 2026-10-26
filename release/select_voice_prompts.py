"""Choose the released voice prompts by round-trip ASR on the selection set (validation utterances,
not the benchmark), with the exported model.

Per voice (studio speaker), candidates are scored on every distinct rendering of the 30
eval/selection_set.json utterances (70 texts): the speaker's original prompt (a whole sentence), another
whole sentence, and two training-style prefixes: the start of a longer utterance cut at a word
boundary 3.5-5 s in, which is exactly how training prompts are built (training/dataloader). A
candidate whose generations open with the prompt's own last words is flagged (prompt leakage).
Writes the winners to $WORK/data/voice_prompts_s2/<voice>.wav and eval/results/voice_prompt_search.json;
release.export_release then ships them.

    python -m release.select_voice_prompts [--release $WORK/release]
    python -m release.select_voice_prompts --redecide --disqualify indian_male_aarav=original_sentence \
        --reason "..."   # re-pick from the saved candidates (no regeneration)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data_pipeline.common import DATA_ROOT, REPO_ROOT, WORK, read_jsonl, setup_logging, write_json
from eval.asr import HinglishScorer, WhisperASR, asr_language, edit_distance

logger = setup_logging("select_voice_prompts")

VOICE_SPEAKERS = {
    "indian_female_ananya": "indictts_hindifull_female",
    "indian_male_aarav": "indictts_hindifull_male",
    "indian_female_meera": "syspin_hindi_female",
    "indian_male_vikram": "syspin_hindi_male",
}
ORIGINAL_PROMPT = {
    "indian_female_ananya": "indictts_hindifull_female__hi",
    "indian_male_aarav": "indictts_hindifull_male__hi",
    "indian_female_meera": "syspin_hindi_female__hi",
    "indian_male_vikram": "syspin_hindi_male__hi",
}
OUT_DIR = DATA_ROOT / "voice_prompts_s2"


def candidates(voice: str, final: list[dict], manifest: dict[str, dict], tmp: Path) -> list[dict]:
    """Original prompt + one more whole sentence + two word-boundary prefixes of longer utterances."""
    from pocket_tts.data.audio_utils import end_on_pause

    speaker = VOICE_SPEAKERS[voice]
    s1 = json.loads((DATA_ROOT / "voice_prompts" / "index.json").read_text())
    s1_row = next(r for r in s1 if r["name"] == ORIGINAL_PROMPT[voice])
    out = [{"kind": "original_sentence", "wav": s1_row["wav"], "text": s1_row["text"], "source_utt_id": s1_row["source_utt_id"]}]
    rows = [r for r in final if r["speaker_id"] == speaker and r["split"] == "train" and r["language"] == "hi"
            and r["utt_id"] in manifest and r["utt_id"] != s1_row["source_utt_id"]]
    rows.sort(key=lambda r: (-round(r["snr_db"]), abs(r["duration_s"] - 5.5)))
    whole = next(r for r in rows if 4.5 <= r["duration_s"] <= 6.5 and r["text_devanagari"].endswith("."))
    picks = [("whole_sentence", whole, None)]
    for r in rows:
        if len(picks) == 3:
            break
        if r["duration_s"] < 7.0 or r is whole:
            continue
        words = manifest[r["utt_id"]]["words"]
        cuts = [(0.5 * (a["end"] + b["start"]), i) for i, (a, b) in enumerate(zip(words, words[1:]), 1)
                if a["end"] is not None and b["start"] is not None and 3.5 <= 0.5 * (a["end"] + b["start"]) <= 5.0]
        if cuts:
            picks.append(("prefix_at_word_boundary", r, max(cuts)))
    for kind, r, cut in picks:
        wav, sr = sf.read(r["audio_path"], dtype="float32")
        text = r["text_devanagari"]
        if cut is not None:
            wav = wav[: int(cut[0] * sr)]
            text = " ".join(w["word"] for w in manifest[r["utt_id"]]["words"][: cut[1]])
        clip = end_on_pause(torch.from_numpy(wav)[None], sr)[0].numpy()
        path = tmp / f"{voice}__{kind}__{r['utt_id']}.wav"
        sf.write(str(path), clip, sr, subtype="PCM_16")
        out.append({"kind": kind, "wav": str(path), "text": text, "source_utt_id": r["utt_id"]})
    return out


def opens_with_prompt_tail(hyp: str, prompt_text: str, ref: str, scorer: HinglishScorer) -> bool:
    tail = [w for w in scorer.canonical(prompt_text).split()[-3:] if len(w) > 2]
    head, ref_head = scorer.canonical(hyp).split()[:2], scorer.canonical(ref).split()[:4]

    def close(a: str, b: str) -> bool:
        return edit_distance(a, b) / max(len(a), len(b)) <= 0.4

    return any(close(h, t) for h in head for t in tail) and not any(close(h, x) for h in head for x in ref_head)


def pick(results: list[dict], disqualified: set[str]) -> dict:
    ok = [r for r in results if r["kind"] not in disqualified] or results
    return min(ok, key=lambda r: (r["prompt_leaks"] > 0, r["cer"], r["wer"]))


def redecide(disqualify: list[str], reason: str) -> None:
    """Re-pick from the candidates scored by a previous run, e.g. disqualifying a prompt whose
    leakage was observed elsewhere on the selection set."""
    path = REPO_ROOT / "eval" / "results" / "voice_prompt_search.json"
    report = json.loads(path.read_text())
    rules = {}
    for spec in disqualify:
        voice, kind = spec.split("=", 1)
        rules.setdefault(voice, set()).add(kind)
    tmp = WORK / "tmp" / "voice_prompt_search"
    for voice, entry in report["voices"].items():
        best = pick(entry["candidates"], rules.get(voice, set()))
        wav = (DATA_ROOT / "voice_prompts" / f"{ORIGINAL_PROMPT[voice]}.wav" if best["kind"] == "original_sentence"
               else tmp / f"{voice}__{best['kind']}__{best['source_utt_id']}.wav")
        audio, sr = sf.read(str(wav), dtype="float32")
        sf.write(str(OUT_DIR / f"{voice}.wav"), audio, sr, subtype="PCM_16")
        entry.update(chosen=best["kind"], chosen_source_utt_id=best["source_utt_id"], chosen_text=best["text"])
        if voice in rules:
            entry["disqualified"] = sorted(rules[voice])
            entry["disqualified_reason"] = reason
    write_json(path, report)
    logger.info({v: (r["chosen"], r["chosen_source_utt_id"]) for v, r in report["voices"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=WORK / "release")
    ap.add_argument("--selection-set", type=Path, default=REPO_ROOT / "eval" / "selection_set.json")
    ap.add_argument("--redecide", action="store_true", help="re-pick from the saved report, no generation")
    ap.add_argument("--disqualify", action="append", default=[], help="voice=candidate_kind")
    ap.add_argument("--reason", default="")
    args = ap.parse_args()
    if args.redecide:
        redecide(args.disqualify, args.reason)
        return
    from release.inference import HinglishTTS

    final = read_jsonl(DATA_ROOT / "processed" / "final.jsonl")
    manifest = {r["utt_id"]: r for r in read_jsonl(DATA_ROOT / "manifests" / "train_deva_aligned.jsonl")}
    utts = json.loads(args.selection_set.read_text(encoding="utf-8"))
    items = []
    for u in utts:
        seen = set()
        for script in ("deva", "roman", "cs"):
            if u["text"][script] not in seen:
                seen.add(u["text"][script])
                items.append({"text": u["text"][script], "language": u["language"], "script": script})
    tmp = WORK / "tmp" / "voice_prompt_search"
    tmp.mkdir(parents=True, exist_ok=True)
    tts = HinglishTTS.from_pretrained(str(args.release), device="cuda", auto_normalize=False)
    asr, scorer = WhisperASR(), HinglishScorer()
    report: dict = {"selection_set": str(args.selection_set.relative_to(REPO_ROOT)), "n_texts": len(items), "voices": {}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for voice in VOICE_SPEAKERS:
        results = []
        for cand in candidates(voice, final, manifest, tmp):
            state = tts.model.get_state_for_audio_prompt(Path(cand["wav"]))
            wavs = []
            t0 = time.time()
            for k, it in enumerate(items):
                torch.manual_seed(5000 + k)
                audio = tts.model.generate_audio(state, it["text"]).float().cpu().numpy()
                path = tmp / f"gen_{voice}_{cand['kind']}_{k:03d}.wav"
                sf.write(str(path), audio, tts.sample_rate)
                wavs.append(path)
            hyps = asr.transcribe(wavs, language=[asr_language(it["language"]) for it in items])
            s = scorer.score([it["text"] for it in items], hyps)
            leaks = sum(opens_with_prompt_tail(h, cand["text"], it["text"], scorer) for h, it in zip(hyps, items))
            res = {**{k: cand[k] for k in ("kind", "source_utt_id", "text")}, "cer": round(s["cer"], 4),
                   "wer": round(s["wer"], 4), "prompt_leaks": leaks, "duration_s": round(sf.info(cand["wav"]).duration, 2),
                   "gen_seconds": round(time.time() - t0, 1), "wav": cand["wav"]}
            results.append(res)
            logger.info(f"{voice} {cand['kind']} ({cand['source_utt_id']}): CER {res['cer']:.4f} WER {res['wer']:.4f} "
                        f"leaks {leaks}")
        best = pick(results, set())
        wav, sr = sf.read(best["wav"], dtype="float32")
        sf.write(str(OUT_DIR / f"{voice}.wav"), np.asarray(wav), sr, subtype="PCM_16")
        report["voices"][voice] = {"speaker_id": VOICE_SPEAKERS[voice], "chosen": best["kind"],
                                   "chosen_source_utt_id": best["source_utt_id"], "chosen_text": best["text"],
                                   "candidates": [{k: v for k, v in r.items() if k != "wav"} for r in results]}
    write_json(REPO_ROOT / "eval" / "results" / "voice_prompt_search.json", report)
    logger.info({v: (r["chosen"], r["chosen_source_utt_id"]) for v, r in report["voices"].items()})


if __name__ == "__main__":
    main()
