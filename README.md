# Pocket TTS for Indian Hinglish

A fine-tune of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) that speaks **Indian Hinglish**
(Hindi, Indian English and the code-switched mix between them) with a natural Indian accent, from text typed in
**Roman Hinglish, Devanagari or mixed script**. 24 kHz, streaming, zero-shot voice cloning, and real time on a
laptop CPU with the distilled student. Built for the IISc Computational Intelligence Lab speech AI evaluation.

| | |
|---|---|
| Intelligibility | **7.2 % CER / 17.6 % WER** (Whisper large-v3-turbo round trip, 24 held-out sentences × 4 voices). The same ASR scores 5.3 % / 13.7 % on real human recordings: the measurement floor. |
| Script robustness | Roman 9.6 %, Devanagari 5.9 %, mixed 5.9 % CER with the phonetic front-end |
| Speed | RTX 4090: RTF 0.126 (7.9× real time), first audio in 30 ms. i9-13900K CPU, int8, 4 threads: RTF 0.221, 74 ms |
| Model | 24-layer FlowLM + LSD head + Mimi codec, 336 M parameters, 4 studio Indian voices + cloning from any `.wav` |
| CPU student | 12-layer distilled student (`--variant 12l`): 10.7 % CER on the benchmark at half the backbone compute. On a 2-core laptop CPU it runs at RTF 0.71 with first audio in 186 ms, where the 24-layer model needs RTF 1.24 |

## The five deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | **Curated, human-verified Hinglish speech dataset**: 7,360 utterances, 10.2 h, 201 speakers, every transcript in three aligned scripts | private HF dataset [`MSRIT-NBM/iisc-hinglish-pocket-tts-dataset`](https://huggingface.co/datasets/MSRIT-NBM/iisc-hinglish-pocket-tts-dataset); code in `data_pipeline/` |
| 2 | **Fine-tuned model**: weights, bilingual tokenizer, config, voices, `inference.py`, a 12-layer CPU student, model card | private HF model [`MSRIT-NBM/pocket-tts-indian-hinglish`](https://huggingface.co/MSRIT-NBM/pocket-tts-indian-hinglish); code in `training/`, `release/` |
| 3 | **Objective benchmark**: round-trip CER / WER per script, front-end and voice; RTF and time to first audio on GPU, desktop CPU and laptop CPU | `eval/evaluation_results.json`, `eval/results/`; code in `eval/` |
| 4 | **Demo**: an interactive studio and a rendered demo video | `demo_app.py` + `demo/`; `demo_hinglish_pocket_tts.mp4` (also in the model repo) |
| 5 | **Reproducible code and documentation**: data pipeline, training, evaluation, release, tests, notes | this repository: `README.md`, `docs/`, `tests/` |

Both Hugging Face repositories are **private**. Evaluators read them with the read-only token shared with the
submission (`export HF_TOKEN=<read-only token>`); without a token both return 401 / not found
(`release/read_token_check.json` for both repositories, `release/model_access_verification.json` for the full
download-and-speak check of the model repo, `data_pipeline/read_access_verification.json` for the dataset).

## Quick start

```bash
pip install pocket-tts huggingface_hub soundfile cmudict wordfreq
export HF_TOKEN=hf_...                                   # read-only token for the private repos

# one sentence to a .wav: the helper downloads the model on first use
hf download MSRIT-NBM/pocket-tts-indian-hinglish inference.py --local-dir .
python inference.py --text "Kal subah 7 baje meri flight hai, please time par alarm set kar dena." --out flight.wav
python inference.py --text "आज ट्रैफ़िक बहुत ज़्यादा है, मैं दस मिनट लेट हो जाऊँगा।" --voice indian_male_aarav --out traffic.wav
python inference.py --text "मेरी next meeting दोपहर 3 बजे है, कृपया presentation और report रेडी रखना।" --out meeting.wav
python inference.py --text "Weekend पर हम Goa जा रहे हैं!" --voice my_recording.wav --out cloned.wav   # zero-shot cloning
python inference.py --variant 12l --quantize --text "Aap kaise hain?" --out fast.wav              # CPU student, int8
```

```python
from inference import HinglishTTS

tts = HinglishTTS.from_pretrained()                      # or ("artifacts/release") for a local copy
audio = tts.synthesize("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो", voice="indian_female_ananya")
tts.save(audio, "out.wav")                               # float32 mono, tts.sample_rate == 24000
for chunk in tts.stream("Battery low है, मुझे charger की ज़रूरत है"):
    ...                                                  # 80 ms chunks as they are decoded
```

The official Pocket TTS CLI reads the repo's config directly:

```bash
pocket-tts generate --config hf://MSRIT-NBM/pocket-tts-indian-hinglish/config.yaml \
    --voice prompt.wav --text "Aaj Bangalore ka weather kaisa hai?"
```

### Interactive studio

```bash
pip install fastapi uvicorn
python demo_app.py                 # http://127.0.0.1:7860   (--device cuda on a GPU machine)
```

`demo_app.py` serves a single-page studio (`demo/index.html`) on top of the shipped `inference.py`:

- **Script showcase**: preset sentences in Roman Hinglish, Devanagari Hinglish and mixed script (`demo/presets.json`).
- **Voices**: the four studio voices (ordered by benchmark CER: Ananya 6.0 %, Aarav 6.3 %, Vikram 7.4 %, Meera 9.0 %),
  or zero-shot cloning from an uploaded recording.
- **Front-end toggle**: the phonetic Hinglish front-end (default, 7.2 % CER) or raw text straight into the
  4,000-piece multi-script tokenizer (12.4 % CER).
- **Telemetry**: time to first audio and real-time factor measured on the machine, the SentencePiece pieces the
  model reads (coloured by script, byte fallbacks flagged), and the 24 kHz waveform and 80-band mel-spectrogram
  drawn while the audio streams. Each number says where it was measured.
- **Instant mode**: every preset × voice × front-end is pre-rendered with the 24-layer model and the student
  (`demo/cache/`, built by `scripts/build_demo_cache.py` on the RTX 4090). Each take keeps the latency measured
  when it was rendered and Whisper's transcript, so presets play in milliseconds on any machine. Other text is
  synthesised live, and a live take played again is replayed from memory.
- **Model choice**: the studio times each model at start-up and preselects the 24-layer one when it runs
  comfortably faster than real time on the machine (RTF < 0.85), otherwise the CPU student.

The studio looks for model files in `--model DIR`, then next to `demo_app.py` (inside a download of the model repo),
then in `artifacts/release/`, and finally downloads the private repo with `HF_TOKEN`.

### Demo video

`demo_hinglish_pocket_tts.mp4` (51 s, 1920×1080 H.264 at 30 fps with AAC audio at 24 kHz, 3.1 MB) was rendered by
`scripts/render_demo_video.py`. It shows six sentences,
two per script, alternating the female and male studio voices. Each one shows the typed text, the front-end
output, the tokenizer pieces, an animated waveform and mel-spectrogram, and the time to first audio and RTF
measured while rendering on the RTX 4090 (whose audio is the soundtrack) and on the i9 CPU with int8 weights.
The soundtrack is the model's own 24 kHz output; per-sentence numbers are in `eval/results/demo_video.json`.

## Architecture and training method

**Base model.** Pocket TTS is a causal transformer (FlowLM) that predicts continuous 32-dimensional Mimi latents
at 12.5 Hz. A one-step LSD flow head samples each latent, an EOS head decides when to stop, and the frozen Mimi
codec decodes 80 ms of 24 kHz audio per frame. Text enters through a lookup table over a SentencePiece
vocabulary; the voice enters as the Mimi latents of a short prompt. The fine-tune starts from Kyutai's 24-layer
English teacher (`english_2026-04_24l`, d_model 1024, 16 heads) and keeps its architecture. Only the tokenizer,
and therefore the text table, is new: a bilingual 4,000-piece SentencePiece (1,866 Devanagari and 1,859 Latin
pieces, digits split, byte fallback) trained on the dataset's three renderings.

**Why the stock recipe fails on 10 hours.** Pocket TTS's own recipe for a new language (`finetune_language.yaml`:
the text table re-initialised from `nn.Embedding`'s N(0, 1), every parameter at lr 2e-4) was run as an ablation
on the same data, batch and steps. It reached a *lower* teacher-forced validation loss (0.254 vs 0.273) yet never
read the text: round-trip CER stayed at **64–105 % at every checkpoint** (best 64.3 %), and the trainer's own
samples agree. The new rows have norm ≈ 32 while the teacher's trained rows have norm ≈ 0.46. With only 10 h,
teacher forcing lets the model predict the next frame from the audio alone, so the randomly initialised text
rows never get wired into the pretrained backbone and the loss curve does not reveal it.

**What made it work.** Four changes, all in `training/` as hooks on pocket-tts's untouched training loop
(`training/train_hinglish.py`, config `configs/finetune_hinglish.yaml`):

1. **Cross-script warm-started token embeddings** (`training/warm_start.py`). Every piece of the new tokenizer
   gets a row in the teacher's embedding space instead of N(0, 1): 868 pieces exist in the teacher's tokenizer
   and are copied; 1,069 Latin pieces are the mean of the teacher rows of their segmentation by the teacher's own
   Unigram model; 1,863 Devanagari pieces are romanised and then composed the same way; 200 byte and control
   pieces get N(0, σ_teacher). Composed rows are rescaled to their parts' norm. So `▁के` and `▁ke` start from the
   same vector, and Devanagari, Roman and mixed spellings of a word begin in agreement. At step 0 this lowers
   validation loss from 0.454 to 0.421 and EOS loss from 0.174 to 0.123.
2. **Multi-script manifest**. Every utterance is presented in all three scripts: Devanagari, Roman Hinglish and
   code-switched. The three share word segmentation, one forced alignment and one set of Mimi latents. That gives
   16,045 text–audio rows from 6,980 training utterances; renderings identical to another script's are kept once,
   which averages 2.3 renderings per utterance. Validation loss stays equal across the three scripts throughout
   (within 0.003).
3. **400-step lookup-table warm-up**. For the first 400 steps only the text table trains, against the frozen
   pretrained backbone, so the backbone is not disturbed while the new rows find their place.
4. **5× differential learning rate**. Three AdamW groups with their own schedules: text table at peak 5e-4 from
   step 0; backbone and heads at peak 1e-4 from step 400 (400-step warm-up); each decays by cosine to 10 %.
   Effective batch 64, bf16, EMA 0.999, CFG dropout 0.2.

The run took 5,000 steps in 26.8 minutes on one RTX 4090. **Checkpoints are selected by round-trip ASR, not by
validation loss.** Validation loss bottoms out at step 2,000, but intelligibility keeps improving until about
3,000–4,000 steps and then degrades. A rule fixed before scoring (lowest CER on a 140-generation held-out
selection set, WER breaking ties within 0.3 points, then the earlier step) picked the **EMA weights of step
3,500**: selection-set CER 7.99 %, WER 18.76 %.

| step | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 | 4,000 | 4,500 | 5,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| monitor CER %, this method | 62.4 | 36.1 | 15.4 | 8.3 | 7.5 | 5.8 | 6.3 | 5.7 | 7.2 | 7.8 |
| monitor CER %, stock recipe | 104.6 | 89.1 | 84.6 | 83.1 | 77.5 | 67.2 | 65.8 | 64.5 | 64.3 | 69.0 |

**Text front-end.** `inference.py` turns a danda into a period and adds final punctuation; the model was trained on
terminally punctuated sentences. With `auto_normalize=True` (the default), the bundled normaliser verbalises numbers,
currency and times in the sentence's matrix language and renders the sentence in Devanagari, spelling English words
phonetically. This was chosen by benchmark: it is the most intelligible input for all three scripts, and Roman input
improves from 18.8 % CER as typed to 9.6 %.

**CPU student.** On a 2-core laptop CPU the 24-layer model is slower than real time even with int8 weights (RTF
1.24). pocket-tts generates latents on one thread and decodes Mimi on another, and the 24-layer FlowLM step alone
takes 93 ms of every 80 ms frame (`scripts/cpu_speed_probe.py`). So the model was distilled with Pocket TTS's own
depth-distillation recipe (`configs/distill_hinglish_12l.yaml`). A student seeded from the teacher (the Hinglish
text table, the heads, and the bottom and top layers of the backbone) learns to reproduce the teacher's backbone
output on the training data, with the flow and EOS heads frozen. A 6-layer student (8,000 steps) reached at best
15.4 % CER on the selection set and swung up to 44 % between checkpoints, so it is not shipped. A **12-layer
student** (6,000 steps, about 30 minutes on the RTX 4090) reaches 11.1 % CER / 23.2 % WER on the selection set
(24-layer: 8.0 / 18.8) and 10.7 % / 22.2 % on the benchmark (24-layer: 7.2 / 17.6), with half the backbone compute:
RTF 0.71 and first audio in 186 ms on the 2-core laptop. It ships as `student_12l/`. A separate pre-quantised int8
checkpoint would not be faster. Dynamic int8 (`--quantize`) already halves the FlowLM step; quantising at load only
costs about 10 s on this CPU, and torch's packed int8 weights are tied to the torch version.

## Dataset curation and human-verification provenance

Only human speech with human-produced transcripts was used: recording prompts or human transcriptions, never ASR
output or synthetic audio. All sources are public on the Hugging Face Hub and were downloaded without gating:

| source | origin | language | utterances | hours |
|---|---|---|---|---|
| IndicTTS Hindi (`SPRINGLab/IndicTTS-Hindi`) | IIT Madras studio voice artists reading prompts | Hindi | 2,589 | 4.00 |
| IndicTTS English (`SPRINGLab/IndicTTS-English`) | IIT Madras Hindi, Gujarati, Marathi, Rajasthani voice artists | Indian English | 2,395 | 2.56 |
| SYSPIN Hindi (public mirror) | IISc SPIRE Lab studio TTS corpus, one female and one male voice | Hindi | 785 | 1.61 |
| OpenSLR-104 (MUCS 2021) | Spoken Tutorial narrations with human transcripts | Hindi–English code-switched | 1,591 | 2.04 |

Curation funnel: 80,253 utterances measured → 57,918 passed the text and audio filters → 7,600 selected by
per-group quotas → **7,360** kept after word alignment (train 6,980 / validation 380; Hindi 4.59 h, English 2.48 h,
code-switched 3.13 h; 201 speakers). Audio is resampled to 24 kHz and trimmed to a 0.1 s lead and an exact 0.18 s
tail. It is loudness-normalised (active-speech RMS −20 dBFS, peaks ≤ −1 dBFS) and contains zero clipped samples.
Every kept utterance has blind SNR ≥ 42 dB (median 80 dB). Each transcript is rendered in three aligned scripts by a
dual-script normaliser (`data_pipeline/hinglish_text_processor.py`: romanisation with schwa deletion, audited
loanword detection, number verbalisation) and word-aligned with wav2vec2 CTC models; 240 rows with implausible word
boundaries were dropped. The dataset card on the Hub documents every filter, licence and rejected source.

## Objective benchmark

24 sentences written for the benchmark and not in the training data (`eval/benchmark_sentences.json`): 8 Roman
Hinglish, 8 Devanagari Hinglish and 8 mixed-script, with digits, currency, questions, dandas and missing final
punctuation. Every sentence is synthesised with every voice through `inference.py` and transcribed by Whisper
large-v3-turbo. Reference and transcript are mapped to canonical spoken Devanagari before scoring, so the same words
in different scripts are not errors. CER ignores spaces; both CER and WER are corpus-level. Results are in
`eval/evaluation_results.json`; code is in `eval/`.

CER / WER (%) by input script, 4 voices × 8 sentences each:

| front-end | Roman Hinglish | Devanagari | mixed script | all 96 |
|---|---|---|---|---|
| **phonetic, rendered in Devanagari (default)** | 9.6 / 21.8 | 5.9 / 15.1 | 5.9 / 15.4 | **7.2 / 17.6** |
| phonetic, input script kept | 13.6 / 26.1 | 5.2 / 14.3 | 12.1 / 24.7 | 10.4 / 21.6 |
| raw text as typed | 18.8 / 31.9 | 6.2 / 16.5 | 11.8 / 24.0 | 12.4 / 24.2 |
| ASR floor: real recordings of held-out utterances | | | | 5.3 / 13.7 |

Per voice (default front-end): Ananya 6.0 / 15.2, Aarav 6.3 / 16.0, Vikram 7.4 / 16.7, Meera 9.0 / 22.4. No
generation missed its end of speech (0 / 360). Rendering the same sentence in each script with one voice gives
Roman 10.5, Devanagari 8.4 and mixed 8.3 % CER: the model itself reads all three scripts.

**Speed.** RTF = synthesis time / audio duration (below 1 is faster than real time). First audio is measured from
the `stream()` call, text normalisation included, to the first decoded 80 ms chunk, with the voice state cached.
Batch size 1, as in streaming use:

| device | weights | threads | RTF, 24-layer | first audio, 24-layer | RTF, 12-layer student | first audio, student |
|---|---|---|---|---|---|---|
| RTX 4090 GPU | fp32 | – | 0.126 | 30 ms | 0.079 | 25 ms |
| i9-13900K CPU | fp32 | 1 | 0.704 | 237 ms | 0.375 | 128 ms |
| i9-13900K CPU | fp32 | 4 | 0.499 | 129 ms | 0.271 | 79 ms |
| i9-13900K CPU | int8 | 1 | 0.323 | 109 ms | 0.185 | 65 ms |
| i9-13900K CPU | int8 | 4 | 0.221 | 74 ms | 0.136 | 48 ms |
| laptop: Xeon 8259CL, 2 cores | int8 | 1 | 1.24 | 360 ms | 0.71 | 186 ms |
| laptop: Xeon 8259CL, 2 cores | fp32 | 1 | 2.48 | 837 ms | 1.29 | 438 ms |

**Why the 24-layer model ran at about real time on the laptop.** The laptop-class machine is a 2-core / 4-thread
Xeon 8259CL, measured while a remote desktop connection used 0.5–2 of its 4 logical CPUs. pocket-tts generates
latents on one thread and decodes Mimi on another, and calls `torch.set_num_threads(1)` when it is imported. With
two cores both stages are already busy, so extra torch threads only add contention (24-layer int8: 1 thread RTF
1.24, 2 threads 1.52). Per 80 ms frame, the 24-layer FlowLM step takes 93 ms in int8 and 182 ms in fp32, against
Mimi's 38 ms: the backbone is the bottleneck. Dynamic int8 halves it, and the 12-layer student halves it again. The
studio therefore runs int8 with one torch thread per two physical cores, and preselects the student on such
machines. All settings are in `eval/results/cpu_speed_local.json`.

## Repository layout

| path | contents |
|---|---|
| `data_pipeline/` | source probe and download, dual-script normaliser and lexicon, audio processing, curation, CTC word alignment, tokenizer training, Mimi latents, dataset push, card and read-access check |
| `training/` | `train_hinglish.py` (official loop + hooks), `warm_start.py`, `param_groups.py`, `eos_metrics.py`, `asr_monitor.py`, `summarize_runs.py`, warm-start report and training summary |
| `eval/` | `asr.py` (Whisper + script-invariant scoring), `evaluate_hinglish_tts.py` (benchmark), sentence sets, results |
| `release/` | `export_release.py`, `export_student.py`, `inference.py` (shipped helper), `select_voice_prompts.py`, `assemble_repo.py`, `model_card.py`, `push_model.py`, `verify_model_access.py` |
| `configs/` | data pipeline, model configs (24 / 12 / 6 layers), fine-tune, ablation and distillation configs |
| `demo_app.py`, `demo/` | interactive studio and its presets |
| `scripts/` | `build_demo_cache.py`, `render_demo_video.py`, `cpu_speed_probe.py`, `gpu_sync.sh` |
| `tests/` | `pytest tests/`: normaliser, audio processing, training helpers, inference front-end, ASR scoring |
| `docs/` | engineering notes: `data_pipeline_notes.md`, `training_evaluation_notes.md` |
| `artifacts/` (git-ignored) | local copy of the released model folder and generated audio |

## Reproducing

Everything heavy ran on one RTX 4090 workstation. Each step reads and writes files in this repo:

```bash
# data, tokenizer, latents
python -m data_pipeline.probe_sources && python -m data_pipeline.download_sources
python -m data_pipeline.prepare_dataset && python -m data_pipeline.align_words && python -m data_pipeline.finalize_dataset
python -m data_pipeline.train_tokenizer && python -m data_pipeline.extract_mimi_latents
python -m data_pipeline.push_to_hub && python -m data_pipeline.verify_read_access

# fine-tune, monitor, select, export, evaluate
python -m training.train_hinglish configs/finetune_hinglish.yaml
python -m training.asr_monitor --run-dir $WORK/runs/hinglish_24l_ws --monitor-set eval/selection_set.json --tag selection
python -m release.export_release --checkpoint $WORK/runs/hinglish_24l_ws/checkpoint_00003500.pt
python -m eval.evaluate_hinglish_tts --model-dir $WORK/release

# CPU student
python -m training.train_hinglish configs/distill_hinglish_12l.yaml
python -m training.asr_monitor --run-dir $WORK/runs/hinglish_12l_distill --model-config configs/hinglish_pocket_tts_12l.yaml \
    --monitor-set eval/selection_set.json --tag selection
python -m release.export_student --checkpoint $WORK/runs/hinglish_12l_distill/checkpoint_00006000.pt --model-config configs/hinglish_pocket_tts_12l.yaml
python -m eval.evaluate_hinglish_tts --model-dir $WORK/release --variant 12l \
    --out eval/results/evaluation_results_student_12l.json --audio-dir $WORK/eval_audio/benchmark_student_12l

# demo assets, publish, verify with the read-only token
python scripts/build_demo_cache.py --model-dir $WORK/release --variants 24l,12l --asr
python scripts/render_demo_video.py --model-dir $WORK/release --asr
python -m release.assemble_repo && python -m release.model_card && python -m release.push_model
python -m release.verify_model_access
```

Tests need no GPU: install the model's requirements plus `pytest`, then run `python -m pytest tests/` from the
repository root.

## Limitations

- About 10 h of read speech: prosody is read-aloud rather than conversational, and rare proper nouns or uncommon
  English words can be mispronounced. The phonetic front-end's English-to-Devanagari spelling is rule-based
  (for example "neural" → नुरल).
- Scores are automatic round-trip ASR; no listening (MOS) study was run. Part of every CER is Whisper's own error
  on Hinglish (5.3 % on real recordings).
- The voice prompt sets both voice and audio quality. With arbitrary cloning prompts, about 1 % of generations can
  open with the prompt's last words; the shipped prompts were chosen to avoid this (0 / 360 on the benchmark).
- Voice cloning can be misused: do not clone anyone's voice without consent.
- The training data keeps its sources' terms (Indic TTS licence, IIT Madras; SYSPIN terms, IISc; CC BY-SA 4.0 for
  OpenSLR-104). The model is released for private evaluation under CC BY 4.0, the base model's licence.
