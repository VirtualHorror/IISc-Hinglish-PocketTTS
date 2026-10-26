"""Render the Hugging Face dataset card (README.md body) from dataset_stats.json + the config."""

from __future__ import annotations

import yaml

from data_pipeline.common import REPO_ROOT


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _hours(d: dict) -> str:
    return f"{d['hours']:.2f}"


def render_card(stats: dict, repo: str, examples: list[dict]) -> str:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data_pipeline.yaml").read_text())
    a, t = cfg["audio"], cfg["text"]
    tot, funnel, al = stats["total"], stats["curation_funnel"], stats["alignment"]
    lang_names = {"hi": "Hindi", "en": "Indian English", "hi-en": "Hinglish (code-switched)"}

    src_rows = []
    for key, src in cfg["sources"].items():
        s = stats["by_source"].get(key, {"utterances": 0, "hours": 0})
        src_rows.append([f"[{src['repo']}]({src['url']})", src["origin"], src["language"],
                         s["utterances"], _hours(s), src["license"]])
    speakers = stats["by_speaker"]
    studio = [[k, v["utterances"], _hours(v)] for k, v in sorted(speakers.items()) if not k.startswith("openslr104")]
    n_slr = sum(1 for k in speakers if k.startswith("openslr104"))
    ex_rows = []
    for r in examples:
        ex_rows.append(f"**{r['source_key']}** (`{r['utt_id']}`, {r['language']}, {r['duration_s']:.2f} s)\n\n"
                       f"- original: {r['text_original']}\n- `text_devanagari`: {r['text_devanagari']}\n"
                       f"- `text_roman`: {r['text_roman']}\n- `text_code_switched`: {r['text_code_switched']}\n")
    rej_rows = []
    for key, reasons in stats["filtering"]["rejections"].items():
        rej_rows.append([key, stats["filtering"]["measured"].get(key, 0), stats["filtering"]["passed_filters"].get(key, 0),
                         ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))])
    d, snr, rms, pk = stats["duration_s"], stats["snr_db"], stats["active_rms_dbfs"], stats["peak_dbfs"]

    return f"""
# IISc Hinglish Pocket TTS Dataset

A curated, **human-verified** speech dataset for fine-tuning [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
to speak Indian **Hinglish**: natural Indian-accented Hindi, Indian English and Hindi-English code-switching. Every
utterance carries its transcript in **three aligned scripts** (full Devanagari, Roman Hinglish, and natural
code-switched Devanagari + Latin), so a model trained on it can read any of them.

Built for the IISc Speech AI internship (round 2). Pipeline code: `data_pipeline/` in the project repository
(reproduction commands at the end of this card).

## At a glance

{_table(["", ""], [
    ["Utterances", f"{tot['utterances']:,}"], ["Hours", f"{tot['hours']:.2f}"], ["Speakers", tot["speakers"]],
    ["Splits", f"train {stats['splits']['train']['utterances']:,} ({_hours(stats['splits']['train'])} h) / validation {stats['splits']['validation']['utterances']} ({_hours(stats['splits']['validation'])} h)"],
    ["Audio", f"{stats['sample_rate']} Hz mono (Mimi codec rate), stored as lossless FLAC (16-bit)"],
    ["Duration", f"{d['min']}-{d['max']} s, median {d['median']} s"],
    ["Loudness", f"active-speech RMS {rms['median']} dBFS (median), peaks <= {pk['max']} dBFS, 0 clipped samples"],
    ["Blind SNR", f">= {snr['min']:.1f} dB (median {snr['median']:.1f} dB, capped at 80)"],
])}

{_table(["Language", "Utterances", "Hours"], [[lang_names.get(k, k), v["utterances"], _hours(v)] for k, v in stats["by_language"].items()])}

`language` is derived from the text: `hi-en` rows contain both Hindi words and English(-origin) words.

## Loading

The repository is private; use a token with read access:

```python
from datasets import load_dataset
ds = load_dataset("{repo}", token="hf_...")          # DatasetDict(train, validation)
row = ds["validation"][0]
row["audio"]["array"], row["audio"]["sampling_rate"]  # 24 kHz
row["text_devanagari"], row["text_roman"], row["text_code_switched"]
```

Audio decoding needs `datasets[audio]`; without it, cast with `Audio(decode=False)` and read the FLAC bytes with
`soundfile`.

## Fields

{_table(["Column", "Description"], [
    ["`audio`", "processed 24 kHz mono speech (FLAC)"],
    ["`text_devanagari`", "full Devanagari: English words in Devanagari phonetics, numbers verbalised (e.g. `मीटिंग`, `प्लीज़`, `सात`)"],
    ["`text_roman`", "Roman Hinglish: Hindi in casual romanisation with schwa deletion (`kal`, `subah`, `hain`), English as English"],
    ["`text_code_switched`", "Hindi in Devanagari, English in Latin. Human code-switching (OpenSLR-104) is kept exactly as transcribed; for monolingual sources English loanwords are moved to Latin script conservatively"],
    ["`text_original`", "the source transcript, unmodified"],
    ["`language`", "`hi`, `en` or `hi-en` (see above)"],
    ["`speaker_id`", "stable speaker label; IndicTTS voice artists keep one id across their Hindi and English recordings"],
    ["`gender`", "`female` / `male` from source metadata; `unknown` for OpenSLR-104 narrators"],
    ["`source_dataset`, `source_url`", "the Hugging Face dataset the utterance came from"],
    ["`source_license`", "licence of that source corpus"],
    ["`source_utt_id`", "the utterance's file name in the source corpus (traceability)"],
    ["`source_sample_rate`", "sample rate before resampling (48000 / 44100 / 16000)"],
    ["`human_verified`", "always `true`: real human speech (no synthetic audio) with human-produced transcripts (recording prompts or human transcription, no ASR output)"],
    ["`duration_s`", "duration after trimming"],
    ["`snr_db`", "blind SNR estimate of the source audio (95th minus 10th percentile of 20 ms frame energy)"],
])}

## Sources

All sources are public on the Hugging Face Hub and downloaded without gating. Only human speech with
human-produced transcripts was used.

{_table(["Source (HF)", "Origin", "Lang", "Utterances", "Hours", "Licence"], src_rows)}

**Human-verification provenance.**
- *Indic TTS (IIT Madras)*: studio recordings by native voice artists reading prepared prompts; the transcript is
  the prompt text released with the recordings (`SPRINGLab/IndicTTS-Hindi`, `SPRINGLab/IndicTTS-English`, uploaded
  by IIT Madras's SPRING Lab). The English recordings used are by the Hindi voice artists (shards 07-10), plus
  Gujarati, Marathi and Rajasthani voice artists for accent and speaker variety.
- *SYSPIN (SPIRE Lab, IISc Bangalore)*: studio-quality TTS corpus (one female and one male Hindi voice); the
  transcript is the recording prompt. Taken from a public third-party mirror whose card states no licence; the
  official distribution and terms are at https://ee.iisc.ac.in/limmitsdataset/.
- *OpenSLR SLR104 (MUCS 2021 Hindi-English code-switching)*: narrations from the Spoken Tutorial project with
  human transcripts, segmented by the challenge organisers using sentence timestamps; recorded at 16 kHz (natural,
  non-studio). Only transcripts that genuinely mix Devanagari and Latin words were kept, at most
  {cfg['sources']['openslr104_hinglish']['max_per_speaker']} per narration, ranked by SNR.

**Speakers.** {len(studio)} studio speakers plus {n_slr} OpenSLR-104 narrators:

{_table(["speaker_id", "Utterances", "Hours"], studio)}

**Considered and not used.** `ai4bharat/Rasa` has no Hindi (its languages are as, bn, kn, ml, mr, ne, ta, pa, te,
sa, ur, ks, sd) and is gated; `ai4bharat/indicvoices_r`, `ai4bharat/Svarah`, `ARTPARK-IISc/Vaani-transcription-part`
are gated for this account; `ai4bharat/IndicCMix` audio is synthesized with F5-TTS (not human); `google/fleurs`
has no `en_in` config and its `hi_in` audio is crowd-sourced 16 kHz; `ai4bharat/MANGO` is a TTS rating set;
`HumynLabs/*-hinglish-audio` ships no transcripts. The full probe is in `data_pipeline/source_probe.json`.

## Audio preprocessing

1. Decode, downmix to mono, resample to **24,000 Hz** with soxr (HQ).
2. Voiced frames: 20 ms frames within {a['silence_floor_db']:.0f} dB of the loudest frame (the rule Pocket TTS's own
   `end_on_pause` uses for voice prompts).
3. Trim dead air: keep **{a['lead_silence_s'] * 1000:.0f} ms** before the first voiced frame and exactly
   **{a['trail_silence_s'] * 1000:.0f} ms** after the last one (crisp end-on-pause; zero-padded if the recording
   ends sooner); {a['fade_s'] * 1000:.0f} ms fades at the cut points.
4. Loudness: active-speech RMS normalised to **{a['loudness']['target_dbfs']:.0f} dBFS**, lowered where needed so
   peaks stay <= {a['loudness']['peak_ceiling_dbfs']:.0f} dBFS ({stats['peak_limited_fraction'] * 100:.1f}% of rows
   were peak-limited). RMS rather than literal peak normalisation to -20 dBFS: a -20 dBFS peak would put speech
   ~15 dB below typical TTS corpora and Pocket TTS's own training audio.
5. Filters: {a['min_duration_s']}-{a['max_duration_s']} s after trimming; **zero clipped samples** in the source
   (|x| >= {a['clip_threshold']} FS); blind SNR >= {a['min_snr_db']:.0f} dB; no internal pause longer than
   {a['max_internal_pause_s']} s; {t['min_chars_per_sec']}-{t['max_chars_per_sec']} romanised characters per second
   (catches transcript/audio mismatches); at least {t['min_words']} words.

## Text processing (dual-script normaliser)

`data_pipeline/hinglish_text_processor.py` turns any transcript (Devanagari, Roman Hinglish, English or mixed)
into the three renderings, with identical word segmentation so word timings transfer between them:

- Devanagari to Roman: rule-based romanisation with Hindi schwa deletion (`समझना` to `samajhna`, `ज़िंदगी` to
  `zindagi`), casual conventions (`aa`/`ee`/`oo` medially, short vowels word-finally), and a lexicon for function
  words (`हैं` hain, `में` mein, `नहीं` nahin).
- English to Devanagari: CMUdict pronunciations mapped to Devanagari with Indian-English conventions (retroflex
  t/d, nukta f/z, anusvara for nasal codas, ॉ for the vowel of *doctor*, unstressed *re-* as रि), stem+suffix
  fallback for missing inflections, letter names for acronyms.
- Roman Hinglish to Devanagari: a Hindi-word index built from the 30k most frequent Hindi words, with a two-state
  Viterbi over the sentence to decide which Latin words are Hindi and which are English.
- English loanwords written in Devanagari are detected conservatively (frequency-margin rules plus a
  corpus-audited block list) before being moved to Latin script, so native words like `कम`, `मोर`, `वर`, `मरी` are
  never rewritten as English.
- Numbers are verbalised in the sentence's matrix language (Hindi 0-99 table, Indian lakh/crore grouping;
  English otherwise). Following the brief's examples, `text_roman` / `text_code_switched` keep digits as written,
  while the TTS training manifests use fully verbalised forms in all three scripts.
- Punctuation is kept; the danda `।` (and `|`) becomes `.`, and every transcript ends with `.`, `?` or `!` (Pocket
  TTS is trained on, and at inference appends, sentence-final punctuation).

Examples (validation split):

{chr(10).join(ex_rows)}

## Curation funnel

{_table(["Stage", "Utterances"], [
    ["measured (configured speaker groups)", f"{funnel['measured']:,}"],
    ["passed text + audio filters", f"{funnel['passed_text_and_audio_filters']:,}"],
    ["selected by per-group quotas (SNR-ranked pool, seeded sampling)", f"{funnel['selected_by_quota']:,}"],
    ["passed word alignment (final)", f"{funnel['passed_word_alignment']:,}"],
])}

{_table(["Source", "Measured", "Passed", "Rejections"], rej_rows)}

**Word alignment.** Pocket TTS training cuts each utterance at a word boundary (audio before the cut is the voice
prompt), so every utterance was force-aligned with a char-level wav2vec2 CTC model (`{al['models']['hi']}` on the
Devanagari rendering for Hindi and code-switched speech; `{al['models']['en']}` for English). Utterances whose
alignment failed or whose first/last word sits implausibly far from the trimmed edges were dropped
({al['counts'].get('implausible_word_boundaries', 0)} rows, mostly OpenSLR-104, where the transcript does not match
the audio). Median first-word start {al['first_word_start_s']['median']} s; median gap from last word to file end
{al['last_word_end_to_file_end_s']['median']} s.

## Limitations

- About 10 hours of speech: enough to fine-tune a pre-trained Pocket TTS checkpoint, far below the 1000+ hours
  recommended for training a new language from scratch.
- OpenSLR-104 audio was recorded at 16 kHz, so it has no content above 8 kHz after resampling
  (`source_sample_rate` marks these rows). Pocket TTS copies acoustic quality from the voice prompt, so prompt with
  studio audio at inference.
- Code-switched renderings of IndicTTS/SYSPIN rows are derived by rule (loanword detection); only OpenSLR-104 rows
  are natural human code-switching. Romanisation and English-to-Devanagari transliteration are rule-based and will
  have occasional errors; proper nouns are not capitalised in Roman text.
- All speech is read speech (prompts, narrations), not spontaneous conversation.

## Licence

Mixed: each row's `source_license` names the terms of its source corpus (Indic TTS licence, IIT Madras; SYSPIN
corpus terms, IISc; CC BY-SA 4.0 for OpenSLR SLR104). See `LICENSE_NOTES.md`. The Indic TTS licence PDF
(https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf) could not be fetched while this card was written, so
check its terms before any redistribution beyond private evaluation.

## Citation / acknowledgements

Indic TTS database (Speech Technology Consortium, IIT Madras; Hema A. Murthy, S. Umesh). SYSPIN / LIMMITS (SPIRE
Lab, IISc Bangalore). MUCS 2021 challenge and OpenSLR SLR104 (Hindi-English code-switched subtask, Spoken Tutorial
project). Kyutai Pocket TTS and the CALM paper (arXiv:2509.06926).

## Reproduction

```bash
python -m data_pipeline.probe_sources      # source availability report
python -m data_pipeline.download_sources   # fetch the selected shards
python -m data_pipeline.prepare_dataset    # filter, select, process audio, annotate text
python -m data_pipeline.align_words        # word alignment + Pocket TTS manifests
python -m data_pipeline.finalize_dataset   # keep aligned rows, write dataset_stats.json
python -m data_pipeline.push_to_hub        # this repository
```
"""
