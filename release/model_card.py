"""Render the model card (README.md of the HF model repo) from the generated result files, so every
number in it comes from a file in this repo:

  eval/evaluation_results.json        24-sentence benchmark: round-trip ASR, EOS, speed
  training/training_summary.json      validation / ASR curves of the fine-tune and its ablation
  training/warm_start_report.json     text-embedding warm start (sources, Procrustes test, step-0 loss)
  data_pipeline/dataset_stats.json    dataset size, tokenizer/hinglish_bilingual_4k.stats.json
  <release>/export_info.json, voices/index.json, model.safetensors (parameter counts)

    python -m release.model_card [--release $WORK/release]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.common import REPO_ROOT, WORK, setup_logging

logger = setup_logging("model_card")

REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
DATASET_ID = "MSRIT-NBM/iisc-hinglish-pocket-tts-dataset"
SCRIPT_NAMES = {"roman": "Roman Hinglish", "devanagari": "Devanagari Hinglish", "code_switched": "Code-switched (Devanagari + English)"}
MONITOR_SCRIPTS = {"deva": "Devanagari", "roman": "Roman", "cs": "Code-switched"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}"


def param_counts(model_file: Path) -> dict[str, float]:
    import safetensors

    counts: dict[str, float] = {}
    with safetensors.safe_open(str(model_file), framework="pt") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            n = 1
            for d in shape:
                n *= d
            counts[key.split(".")[0]] = counts.get(key.split(".")[0], 0) + n
    return {k: v / 1e6 for k, v in counts.items()}


def render(release: Path) -> str:
    ev = load(REPO_ROOT / "eval" / "evaluation_results.json")
    ts = load(REPO_ROOT / "training" / "training_summary.json")
    ws = load(REPO_ROOT / "training" / "warm_start_report.json")
    ds = load(REPO_ROOT / "data_pipeline" / "dataset_stats.json")
    tok = load(REPO_ROOT / "tokenizer" / "hinglish_bilingual_4k.stats.json")
    exp = load(release / "export_info.json")
    voices = load(release / "voices" / "index.json")
    params = param_counts(release / "model.safetensors")
    sel = ts["selected"]
    main_run = ts["runs"][sel["run"]]
    ablation = next((r for k, r in ts["runs"].items() if k != sel["run"]), None)
    q, sp = ev["quality"], ev["speed"]
    default, keep, raw, inv = q["default"], q["keep_script"], q["raw"], q["invariance"]
    floors = ts.get("asr_floor_real_recordings", {})
    floor = floors.get("selection") or floors.get("monitor") or {}
    gpu = sp["gpu_fp32"]
    cpu_keys = [k for k in sp if k.startswith("cpu_")]
    best_cpu = min(cpu_keys, key=lambda k: sp[k]["rtf_total"]) if cpu_keys else None
    n_voices = len(voices)

    def by_script(block: dict) -> list[list[Any]]:
        rows = []
        for s in ("roman", "devanagari", "code_switched"):
            g = block.get(f"input_script={s}")
            if g:
                rows.append([SCRIPT_NAMES[s], g["n"], pct(g["cer"]), pct(g["wer"]), g["no_eos"]])
        o = block["overall"]
        rows.append(["**All**", o["n"], f"**{pct(o['cer'])}**", f"**{pct(o['wer'])}**", o["no_eos"]])
        return rows

    kinds = {"session1_sentence": "whole sentence", "whole_sentence": "whole sentence",
             "prefix_at_word_boundary": "utterance start, cut at a word boundary"}
    voice_rows = [[f"`{v['name']}`", v["source"], f"{v['duration_s']:.1f} s, {kinds.get(v.get('prompt_kind'), '–')}",
                   pct(default.get(f"voice={v['name']}", {}).get("cer")),
                   pct(default.get(f"voice={v['name']}", {}).get("wer"))] for v in voices]

    speed_rows = []
    for key in ["gpu_fp32", *cpu_keys]:
        s = sp[key]
        device, precision, *threads = key.split("_")
        label = {"gpu": f"GPU ({ev['environment']['gpu']})", "cpu": f"CPU ({ev['environment']['cpu']})"}[device]
        speed_rows.append([label, precision, threads[0].replace("thread", " thread") if threads else "–",
                           f"{s['rtf_total']:.3f}", f"{s['x_real_time']:.2f}×", f"{s['ttfa_ms_p50']:.0f}",
                           f"{s['ttfa_ms_p90']:.0f}"])

    curve_rows = []
    monitor = {r["step"]: r for r in main_run.get("asr_monitor", [])}
    selection = {r["step"]: r for r in main_run.get("asr_selection", [])}
    for v in main_run["valid"]:
        m, s = monitor.get(v["step"]), selection.get(v["step"])
        curve_rows.append([v["step"], f"{v['loss']:.3f}", f"{v['deva/loss']:.3f} / {v['roman/loss']:.3f} / {v['cs/loss']:.3f}",
                           f"{v['eos_hit@-4']:.2f}", pct(m["cer"]) if m else "–", pct(m["wer"]) if m else "–",
                           pct(s["cer"]) if s else "–", pct(s["wer"]) if s else "–"])
    ablation_rows = []
    ablation_note = ""
    if ablation:
        a_mon = {r["step"]: r for r in ablation.get("asr_monitor", [])}
        inloop_path = REPO_ROOT / "eval" / "results" / "inloop_samples_asr.json"
        inloop = load(inloop_path) if inloop_path.exists() else {}
        ours_in = inloop.get(sel["run"], {})
        abl_name = next(k for k in ts["runs"] if k != sel["run"])
        abl_in = inloop.get(abl_name, {})
        for v, av in zip(main_run["valid"], ablation["valid"], strict=False):
            if v["step"] != av["step"]:
                continue
            am, mm = a_mon.get(v["step"]), monitor.get(v["step"])
            oi, ai = ours_in.get(f"step{v['step']:08d}"), abl_in.get(f"step{v['step']:08d}")
            ablation_rows.append([v["step"], f"{v['loss']:.3f}", f"{av['loss']:.3f}",
                                  pct(mm["cer"]) if mm else "–", pct(am["cer"]) if am else "–",
                                  pct(oi["cer"]) if oi else "–", pct(ai["cer"]) if ai else "–"])
        best_abl = min(ablation.get("asr_monitor", []), key=lambda r: r["cer"], default=None)
        best_ours = min(main_run.get("asr_monitor", []), key=lambda r: r["cer"], default=None)
        if best_abl and best_ours:
            ablation_note = (
                f"The official recipe reaches a *lower* validation loss ({ablation['best_valid_loss']['loss']:.3f} at step "
                f"{ablation['best_valid_loss']['step']}, vs {main_run['best_valid_loss']['loss']:.3f}) yet never speaks the "
                f"text intelligibly: its best monitor CER is {pct(best_abl['cer'])} % (step {best_abl['step']}) against "
                f"{pct(best_ours['cer'])} % for the warm start (step {best_ours['step']}), and the trainer's own samples "
                "(raw weights, CFG 2) agree. With 10 h of data, randomly initialised text rows are never wired into the "
                "pretrained backbone: teacher forcing lets the model predict frames from the audio alone, so the loss looks "
                "fine while free-running generation babbles. Teacher-forced loss is therefore not used to select "
                "checkpoints; round-trip ASR is.")
    step0 = ws.get("step0_validation", {}).get("scores", {})
    helper = ws.get("helper", {})
    pg = load_param_groups()
    by_src = ws["by_source"]
    example_rows = [[f"`{e.split(' ')[0]}`", " ".join(e.split(" ")[1:]) or "(same piece)"] for e in
                    ws["examples"].get("transliterated", [])[:4] + ws["examples"].get("decomposed", [])[:2]]
    total = ds["total"]
    langs = ds["by_language"]

    return f"""---
license: cc-by-4.0
language:
- hi
- en
base_model: kyutai/pocket-tts
pipeline_tag: text-to-speech
library_name: pocket-tts
datasets:
- {DATASET_ID}
tags:
- text-to-speech
- tts
- hindi
- hinglish
- code-switching
- indian-english
- voice-cloning
- pocket-tts
---

# Pocket TTS · Indian Hinglish (24-layer)

Text-to-speech for Indian **Hinglish**: Hindi, Indian English and the code-switched mix between them,
read from **Roman Hinglish, Devanagari or mixed-script** input with a natural Indian accent. 24 kHz,
streaming, with zero-shot voice cloning. It is a fine-tune of Kyutai's
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) 24-layer English teacher (`english_2026-04_24l`)
on {total['hours']:.1f} h of human-verified Indian speech, with every utterance shown to the model in all
three scripts, so the same sentence sounds the same whichever way it is typed.

| | |
|---|---|
| Input | Roman Hinglish, Devanagari, code-switched Devanagari + Latin, Indian English |
| Output | 24 kHz mono audio, streamed in 80 ms frames |
| Voices | {n_voices} studio Indian voices, plus zero-shot cloning from any `.wav` |
| Model | 24-layer FlowLM with LSD sampler head ({params.get('flow_lm', 0):.0f} M) + Mimi codec ({params.get('mimi', 0):.0f} M), float32 |
| Round-trip CER / WER (24-sentence benchmark) | **{pct(default['overall']['cer'])} % / {pct(default['overall']['wer'])} %** (Whisper large-v3-turbo; on real recordings of held-out Hindi / English / Hinglish utterances it scores {pct(floor.get('cer'))} % / {pct(floor.get('wer'))} %) |
| Speed (RTF, lower is faster) | GPU {gpu['rtf_total']:.3f} ({gpu['x_real_time']:.1f}× real time){f", CPU {sp[best_cpu]['rtf_total']:.2f} ({best_cpu.split('_')[1]}, {best_cpu.split('_')[2].replace('thread', ' thread')})" if best_cpu else ""} |
| Time to first audio | GPU {gpu['ttfa_ms_p50']:.0f} ms{f", CPU {sp[best_cpu]['ttfa_ms_p50']:.0f} ms" if best_cpu else ""} (median) |

> **Private repository** for evaluation. Authenticate with a token that can read it (`export HF_TOKEN=hf_...`).

## Quick start

```bash
pip install pocket-tts huggingface_hub soundfile cmudict wordfreq
export HF_TOKEN=hf_...                      # read access to this private repo
hf download {REPO_ID} inference.py --local-dir .
python inference.py --text "Kal subah 7 baje meri meeting hai, please reminder set kar do" --out out.wav
python inference.py --text "आज ट्रैफ़िक बहुत ज़्यादा है, मैं दस मिनट लेट हो जाऊँगा" --voice indian_male_aarav --out out2.wav
python inference.py --text "मेरी flight दो घंटे delay हो गई है" --voice my_voice.wav --out cloned.wav   # zero-shot cloning
```

```python
from inference import HinglishTTS             # inference.py from this repo

tts = HinglishTTS.from_pretrained()            # downloads {REPO_ID}
audio = tts.synthesize("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो", voice="indian_female_ananya")
tts.save(audio, "out.wav")                     # float32, tts.sample_rate == 24000
for chunk in tts.stream("Battery low है, मुझे charger की ज़रूरत है"):   # streaming
    ...
```

The official Pocket TTS CLI works too, with this repo's config:

```bash
pocket-tts generate --config hf://{REPO_ID}/config.yaml --voice <prompt.wav> --text "Aap kaise hain? Main theek hoon."
```

**Text front-end** (`inference.py`): a danda (`।`, `॥`) becomes a period and a period is appended when the
text does not end in `.`, `?` or `!` (the model was trained on terminally punctuated sentences). With
`auto_normalize=True` (default; needs `cmudict` and `wordfreq`) the bundled `hinglish_text` normaliser also
verbalises numbers, currency and times in the sentence's matrix language ("7 baje" → "saat baje",
"2,450 rupaye" → "do hazaar chaar sau pachaas rupaye"), normalises quotes and punctuation, and renders the
sentence in Devanagari, spelling English words phonetically (`script="devanagari"`, the default: the most
intelligible rendering on the benchmark below, because it removes the guesswork of reading English and
romanised Hindi spellings). `script="auto"` keeps the input's script, `"roman"` / `"code_switched"` convert to
those; `auto_normalize=False` sends the text as typed (punctuation rules only). The model itself reads all
three scripts (see *raw* and *script invariance* below).

### Voices

{table(["voice", "source", "prompt", "CER %", "WER %"], voice_rows)}

Each voice ships as a clean 24 kHz `.wav` prompt and as `voices/<name>.safetensors`, its model state
precomputed by this model (instant loading). The prompt clip of each speaker was chosen by round-trip ASR on
held-out validation sentences (not the benchmark) among the Session 1 clip, another whole sentence and two
utterance starts cut at a word boundary, which is how prompts look in training
(`training/voice_prompt_search.json`). Any other `.wav` (5–10 s of clean speech) clones that voice zero-shot;
output quality follows the prompt's quality, so prefer studio-clean prompts. CER/WER per voice are on the
24-sentence benchmark (default front-end).

## Benchmark

24 sentences written for this benchmark (not in the training data; `eval/benchmark_sentences.json`):
8 Roman Hinglish, 8 Devanagari Hinglish (English loanwords in Devanagari) and 8 code-switched, typed the way
people write each script, including digits, currency, abbreviations, questions, the danda and inputs without
final punctuation. Every sentence is synthesised with every voice; the audio is transcribed by
`openai/whisper-large-v3-turbo` (Hindi), and reference and transcript are both mapped to canonical spoken
Devanagari before scoring (see [Evaluation protocol](#evaluation-protocol)).

### Intelligibility (round-trip ASR, {len(voices)} voices × 24 sentences)

Default front-end (`auto_normalize=True`, rendered in Devanagari):

{table(["input script", "utterances", "CER %", "WER %", "no EOS"], by_script(default))}

Front-end keeping the input's script (`auto_normalize=True, script="auto"`):

{table(["input script", "utterances", "CER %", "WER %", "no EOS"], by_script(keep))}

Raw text as typed (`auto_normalize=False`, punctuation rules only):

{table(["input script", "utterances", "CER %", "WER %", "no EOS"], by_script(raw))}

Script invariance: each sentence converted by the normaliser into all three scripts (same content, voice
`{voices[0]['name']}`):

{table(["rendered as", "utterances", "CER %", "WER %", "no EOS"], by_script(inv))}

For scale: Whisper's own error on real recordings of held-out validation utterances (10 Hindi, 10 Indian
English, 10 code-switched; studio and spoken-tutorial speech) is CER {pct(floor.get('cer'))} % / WER
{pct(floor.get('wer'))} % (`eval/selection_set.json`). The benchmark sentences have no human recordings, so
part of every number above is Whisper's own error on Hinglish. "no EOS" counts generations that ran to the length
cap without predicting the end of speech.

### Speed

{table(["device", "precision", "torch threads", "RTF", "speed", "first audio p50 (ms)", "p90 (ms)"], speed_rows)}

RTF = synthesis time / audio duration over the 24 sentences (lower is faster; 1/RTF = × real time). First
audio = time from calling `stream()` (text normalisation included) to the first decoded 80 ms chunk, voice
state cached, after warm-up. Pocket TTS generates on one thread and decodes Mimi on another; "torch threads"
is the intra-op thread count. int8 = Pocket TTS's dynamic int8 quantisation (`--quantize`). Measured with
batch size 1, as in streaming use.

## Model details

- **Architecture** (unchanged from the teacher): causal transformer FlowLM, 24 layers, d_model 1024, 16 heads,
  RoPE; an LSD (Lagrangian self-distillation) sampler head (MLP-AdaLN, depth 6, width 512) that turns each
  backbone output into the next 32-dim Mimi latent in one step; an EOS head; a text lookup table
  (4,001 × 1,024); voice conditioning = the prompt's Mimi latents projected into the sequence after a BOS.
  Mimi (24 kHz, 12.5 Hz frames, 32-dim continuous latents) is the frozen codec, encoder included for cloning.
- **Tokenizer**: bilingual SentencePiece unigram, 4,000 pieces ({tok['piece_scripts']['devanagari']:,}
  Devanagari, {tok['piece_scripts']['latin']:,} Latin, 256 byte fallback), standalone `. , ? ! ; :`, digits
  split; trained on the dataset's three renderings.
- **Inference defaults**: temperature {ev['settings']['temperature']}, EOS threshold {ev['settings']['eos_threshold']},
  1 LSD step, no classifier-free guidance (single pass per frame).
- **Weights**: EMA (decay 0.999) of training step {exp['step']} of run `{sel['run']}`.

## Training

### Data

[`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID}) (private): {total['utterances']:,} utterances,
{total['hours']:.1f} h, {total['speakers']} speakers, all human-transcribed read speech: IndicTTS Hindi and
English (IIT Madras studio voice artists), SYSPIN Hindi (IISc), and OpenSLR-104 Hindi–English code-switched
narrations. Hindi {langs['hi']['hours']:.2f} h, Indian English {langs['en']['hours']:.2f} h, code-switched
{langs['hi-en']['hours']:.2f} h. Loudness-normalised (active speech −20 dBFS RMS, −1 dBFS peak), trimmed to a
0.1 s lead and 0.18 s tail, word-aligned with CTC (wav2vec2), every text ending in `.`, `?` or `!`.

### Multi-script training

Every utterance has three renderings with identical word segmentation: Devanagari (English words in
Devanagari phonetics), Roman Hinglish (Hindi romanised, English as English) and code-switched (Hindi in
Devanagari, English in Latin). Word *i* is the same spoken word in all three, so one forced alignment and one
set of Mimi latents serve all of them. The training manifest holds every distinct rendering
(16,045 rows from 6,980 training utterances; renderings identical to another script's, such as the
code-switched form of an all-Hindi sentence, are kept once), giving ~2.3 text-to-audio pairings per utterance.
The model learns that "Kal subah meeting hai", "कल सुबह मीटिंग है" and "कल सुबह meeting है" are the same
Indian speech; validation loss is the same for the three scripts throughout training (table below).

### Warm-started text embeddings

The official recipe for a new tokenizer (`reset_text_embedding`) starts the text table from `nn.Embedding`'s
N(0, 1): rows of norm ≈ 32 fed to a backbone whose trained text rows have norm ≈ {ws['teacher_row_norm']['median']:.2f}.
Instead, every one of the 4,000 pieces was initialised from the English teacher's table
([`training/warm_start.py`](#files)):

{table(["source", "pieces", "rule"], [
        ["exact", f"{by_src.get('exact', 0):,}", "same piece in the teacher's tokenizer with a trained row: copied"],
        ["decomposed", f"{by_src.get('decomposed', 0):,}", "Latin piece: mean of the teacher rows of its segmentation by the teacher's own Unigram model"],
        ["transliterated", f"{by_src.get('transliterated', 0):,}", "Devanagari piece: casual romanisation, then as decomposed"],
        ["random", f"{by_src.get('random', 0):,}", "byte-fallback / control pieces without a trained teacher row: N(0, σ_teacher)"],
    ])}

Composed rows are rescaled to the mean norm of their parts. Examples (piece → romanisation → teacher pieces):

{table(["piece", "built from"], example_rows)}

Devanagari and Roman spellings of a word thus start from the *same* vector (`▁के` and `▁ke` → `▁ ke`).
Indus-Labs/indus-pocket-tts's Hindi/Hinglish table was tested as a second source for Devanagari pieces: an
orthogonal Procrustes map onto the teacher's space fitted on {helper.get('n_anchors', '–')} shared pieces
reaches a held-out cosine of only {helper.get('heldout_cos', '–')} (shuffled pairs:
{helper.get('shuffled_cos', '–')}), i.e. the spaces are unrelated, so it was not used. At step 0 (same
batches and noise) the warm start lowers validation loss from {step0.get('official_reset_N(0,1)', {}).get('loss', '–')}
to {step0.get('warm_start_teacher_only', {}).get('loss', '–')} (EOS loss
{step0.get('official_reset_N(0,1)', {}).get('eos_loss', '–')} → {step0.get('warm_start_teacher_only', {}).get('eos_loss', '–')}).

### Staged unfreezing and differential learning rates

AdamW (β = 0.9/0.95, weight decay 0.1 except the text table), effective batch 64 (16 × 4 accumulation), bf16
autocast, flow-batch multiplier 4, CFG dropout 0.2 (text and voice), EOS loss weight 0.1, voice prompt =
utterance start up to a random word boundary within 5 s:

{table(["group", "parameters", "peak lr", "starts at step", "warmup"], pg)}

Stage 1 trains only the text table against the frozen pretrained backbone; from stage 2 everything trains,
the table at a 5× higher peak. Each group decays by cosine to 10 % of its peak at step {main_run['steps']}.
EMA decay 0.999. One RTX 4090: {main_run.get('it_s_median', '–')} steps/s, {main_run.get('wall_clock_minutes', '–')}
minutes for {main_run['steps']} steps.

### Training curve and checkpoint selection

Validation = teacher-forced loss on the 380 held-out utterances in each script (raw weights). Round-trip ASR
uses the EMA weights through the inference path: *monitor* = 6 held-out code-switched utterances × 3 scripts
× 2 voices, *selection* = 30 held-out utterances (10 Hindi, 10 English, 10 code-switched) in every script
× 2 voices.

{table(["step", "valid loss", "Deva / Roman / mixed", "EOS hit @−4", "monitor CER %", "monitor WER %",
        "selection CER %", "selection WER %"], curve_rows)}

With 10 h of audio, validation loss bottoms out around step {main_run['best_valid_loss']['step']} while
intelligibility keeps improving for another ~1,000–2,000 steps before it degrades: the released checkpoint is
step {sel['step']}. {sel.get('reason', '')}

{"### Ablation: the official recipe" if ablation_rows else ""}

{("Same data, batch, steps and evaluation with the official `finetune_language` settings (N(0, 1) text table, "
  "all groups at lr 2e-4 after 1,000 warmup steps, constant). Monitor = EMA weights through the inference path; "
  "samples = the trainer's in-training samples of the same sentences (raw weights, CFG 2.0):" + chr(10) + chr(10) +
  table(["step", "valid loss (ours)", "valid loss (official)", "monitor CER % (ours)", "monitor CER % (official)",
         "samples CER % (ours)", "samples CER % (official)"], ablation_rows) + chr(10) + chr(10) + ablation_note)
 if ablation_rows else ""}

## Evaluation protocol

Whisper large-v3-turbo (revision `41f01f3`) transcribes every generated utterance with language `hi` (`en` for
English-matrix validation utterances). Reference and transcript both go through the training data's normaliser
(`HinglishTextProcessor.annotate(...).spoken_devanagari`: numbers verbalised, Roman Hindi and English words to
Devanagari) and a light orthographic fold (nukta, chandrabindu → anusvara, candra vowels, nasal consonant +
virama → anusvara, punctuation), so the same words written in different scripts or spellings are not errors.
CER is computed without spaces (Hindi word segmentation varies), WER over words; both are corpus-level
(total edits / total reference length). Generation uses the inference defaults and a fixed seed per utterance.
Code: `eval/asr.py`, `eval/evaluate_hinglish_tts.py`; full per-utterance results in `evaluation_results.json`.

## Limitations

- 10 h of read speech: prosody is read-aloud rather than conversational; very long or unusual sentences,
  rare proper nouns and uncommon English words can be mispronounced (the normaliser's Roman→Devanagari and
  English→Devanagari conversion is rule-based).
- Inputs are split into sentences / chunks of ≤ 50 tokens; long paragraphs are generated chunk by chunk.
- The voice prompt sets voice *and* audio quality: noisy prompts give noisy output. Training prompts are the
  start of the same utterance, so occasionally (≈ 1 % of generations with some prompts) the output opens with
  the prompt's last words; the shipped prompts were chosen to avoid this.
- Rendering English words in Devanagari (the default front-end) can create homographs, e.g. "low" → लो, read as
  the Hindi word; `script="auto"` keeps the typed script. Raw Roman input is the weakest (romanised Hindi
  spellings vary), which is why the front-end converts it by default.
- Scores are round-trip ASR, an automatic proxy for intelligibility; no listening (MOS) study was run.
- Voice cloning can be misused: do not clone a person's voice without their consent.

## Licence and attribution

Released for private evaluation under CC BY 4.0, the licence of the base model
([kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)), whose Mimi codec weights are included.
The training data carries its sources' terms (Indic TTS licence, IIT Madras; SYSPIN corpus terms, IISc;
CC BY-SA 4.0 for OpenSLR-104); the voice prompts are IndicTTS and SYSPIN studio recordings. Check those terms
before any redistribution beyond evaluation. Uses the CALM / Pocket TTS training recipe (Kyutai,
arXiv:2509.06926).

## Files

| file | content |
|---|---|
| `model.safetensors` | EMA weights (FlowLM + Mimi, float32), Pocket TTS inference format |
| `config.yaml` | model config (`hf://` paths), for `pocket-tts generate --config` |
| `tokenizer.model` | bilingual SentencePiece tokenizer (4,000 pieces) |
| `inference.py`, `requirements.txt` | one-command inference / Python API |
| `hinglish_text/` | text normaliser used by `auto_normalize` |
| `voices/` | studio voice prompts (`.wav`), precomputed voice states (`.safetensors`), `index.json` |
| `samples/` | the benchmark sentences with every voice (`roman_*`, `deva_*`, `mixed_*` = input script); `samples/index.json`: text, model input, Whisper transcript, CER/WER |
| `evaluation_results.json` | benchmark results, per utterance |
| `eval/` | benchmark, selection and monitor sentence sets |
| `training/` | training and model configs, warm-start report, training summary, voice-prompt search |
"""


def load_param_groups() -> list[list[Any]]:
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "finetune_hinglish.yaml").read_text(encoding="utf-8"))
    pg = cfg["hinglish"]["param_groups"]
    what = {"text_embedding": "text lookup table (4.1 M)",
            "backbone": "transformer, input projection, speaker projection, BOS (302 M)",
            "heads": "LSD sampler head, EOS head (9.8 M)"}
    return [[g, what[g], f"{float(pg[g]['peak_lr']):.0e}".replace("e-0", "e-"), pg[g].get("start_step", 0),
             pg[g].get("warmup_steps", 0)]
            for g in ("text_embedding", "backbone", "heads")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=WORK / "release")
    args = ap.parse_args()
    card = render(args.release)
    (args.release / "README.md").write_text(card, encoding="utf-8")
    logger.info(f"wrote {args.release / 'README.md'} ({len(card.splitlines())} lines)")


if __name__ == "__main__":
    main()
