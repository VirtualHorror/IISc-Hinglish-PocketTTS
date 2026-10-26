# Data pipeline notes: Pocket TTS inspection, data pipeline, and the fine-tuning starting point

All numbers below come from files in this repo: `data_pipeline/*.json` (generated on the GPU workstation)
and the pocket-tts checkout at commit `2dae420` (2026-10-26, version 4.0.1+2).

## 1. GPU workstation

- 1x RTX 4090 24 GB (driver 595.99, CUDA 13.2), i9-13900K
  (32 threads), 91 GiB RAM, 687 GB free on `/home`. **Shared machine**: other users' jobs appear on the GPU.
- Everything lives in `~/vaani_pocket_tts_work` (`$WORK`); `source $WORK/env.sh` sets the venv, uv-managed
  CPython 3.12, every cache (`HF_HOME`, `UV_CACHE_DIR`, `TORCH_HOME`, ...), `POCKET_TTS_REPO`, and `HF_TOKEN`.
- Local access: `.ssh/gpu_ssh.sh gpu '<cmd>'` (git-ignored; password from `.env` via `SSH_ASKPASS`).
  Code sync: `scripts/gpu_sync.sh push` (code/configs -> `$WORK/repo`) and `scripts/gpu_sync.sh pull`
  (generated `data_pipeline/*.json` + `tokenizer/` -> here).
- Every change to the machine is logged in the local, git-ignored `gpu_state_log.md` (used to restore the workstation afterwards).

## 2. Pocket TTS, as found in the repo

**Models.** `FlowLMModel`: transformer d_model 1024, 16 heads, 6 layers (released students) or 24 layers
(teachers), plus an LSD sampler head (`flow_net`, MLP-AdaLN, depth 6, width 512) that turns the backbone output
into the next 32-dim latent frame, an EOS head (`out_eos`), a text lookup table (`conditioner.embed`
[n_bins+1, 1024]; n_bins 4000, +1 pad row), `speaker_proj_weight` [1024, 32], and `bos_before_voice`.
Latent normalisation buffers `emb_mean` / `emb_std` [32].

**Mimi** (24 kHz): SEANet encoder (strides 6-5-4, 200 Hz), 2-layer transformer (d 512, context 250), conv
downsample x16 to **12.5 Hz**, 32-dim *continuous* latents (`inner_dim`), `DummyQuantizer` projecting 32 -> 512
for decoding. `encode_to_latent([B,1,T]) -> [B, T/1920, 32]`. Mimi is frozen in training; its encoder
weights are zeroed in the "without voice cloning" releases (and in `Indus-Labs/indus-pocket-tts`).

**Voice cloning at inference.** Prompt WAV -> `end_on_pause` (cut trailing frames >35 dB below the loudest
20 ms frame, 20 ms fade, append 80 ms of zeros) -> Mimi latents -> `speaker_proj` -> prefixed after a BOS into
the KV cache. **Text** -> `prepare_text_prompt` (config `replace_characters`, capitalise first letter, append
terminal punctuation from `.!?…`, which does **not** include the danda `।`) -> tokenizer -> long inputs split
at `.`/`!`/`?` tokens, sub-split at `,`/`;`/`:`.

**Training code (`training/`).** Manifests are jsonl `{path, start, duration, transcript, speaker, words}`.
The loader cuts each utterance at a random word boundary inside the first 5 s: audio before the cut is the
voice prompt, audio after it the target, text = `" ".join(words[i:])`. Unaligned rows raise, so **word
alignments are mandatory** (`scripts/align_data.py`, wav2vec2 CTC trellis). The target is trimmed to last word
+ 0.2 s (EOS target); augmentations: 30% final-period dropout, 0-0.5 s prompt-tail trim. `train.py`
precomputes Mimi latents once (`latents/<mimi_hash8>/<stem>_<idx>.safetensors` + `<stem>_latents.jsonl` +
`.meta.json` with `stitch_frames`) and reuses a cache whose Mimi hash matches. Validation always reads audio.
Configs: `scratch`, `finetune` (same tokenizer, lr 2e-5), **`finetune_language`** (24L English teacher,
`reset_text_embedding`, lr 2e-4), `depth_distill` (24L -> 6L student with CFG baked in), `drifting`.
README guidance: >= 100 h for a new language (1000+ h for a strong one); effective batch >= 64; a 24 GB GPU
runs batch 16 x grad-accum 4; the release recipe is fine-tune the 24L teacher, then distil to 6 layers.

## 3. Existing checkpoints (measured: `data_pipeline/checkpoint_inspection.json`)

| | kyutai english_2026-04_24l | saryps-labs/pocket-tts-hindi | Indus-Labs/indus-pocket-tts | ours (tokenizer) |
|---|---|---|---|---|
| layers / params | 24 / 336.1 M (fp32) | 6 / 109.5 M (bf16) | 6 / 109.5 M (fp32) | - |
| Mimi encoder | yes | yes | zeroed (single voice "Inaya") | - |
| tokenizer pieces (Deva / Latin) | 23 / 3,488 | 3,986 / **8** | 2,217 / 1,608 | 1,866 / 1,859 |
| standalone `. , ? !` | yes (no `।`) | **none** | yes (+ `।`) | yes |
| probe: Devanagari | 86 tok, **21 byte-fallback** | 25 tok, 3 unk | 18 tok, 0 | 30 tok, 0 |
| probe: Roman Hinglish | 33 tok, 0 | 67 tok, **24 unk** | 34 tok, 0 | 27 tok, 0 |
| probe: code-switched | 62 tok, 9 fallback | 38 tok, 10 unk | 21 tok, 0 | 28 tok, 0 |
| probe: English | 22 tok, 0 | 61 tok, 21 unk | 22 tok, 0 | 32 tok, 0 |

- **saryps**: trained from scratch on 932 h of IndicVoices + Kathbath Hindi; its README says punctuation was
  "effectively absent from training" and the tokenizer confirms it (no punctuation pieces, 8 Latin pieces):
  Roman Hinglish, English and punctuation are out of vocabulary.
- **Indus**: 24L English teacher fine-tuned on ~90 h of single-voice Hindi/Hinglish call-centre speech, then
  distilled. Its tokenizer covers both scripts, yet its README states "romanised Hindi ("aap kaise hain") is
  read as English": the gap is in what the model learned, not in the tokenizer. No 24L teacher is released and
  the Mimi encoder is zeroed, so it cannot clone voices or serve as the official fine-tuning starting point.
- **Our design answer**: every utterance is trained in all three scripts (Devanagari, Roman Hinglish,
  code-switched) with one bilingual tokenizer, so Roman Hindi is learned as Hindi, English as Indian English,
  and mixed script as both.

## 4. Data decisions (details in the dataset card and `data_pipeline/dataset_stats.json`)

- **Sources** (all ungated): IndicTTS Hindi + English (IIT Madras; the Hindi voice artists' English shards
  give bilingual speakers), SYSPIN Hindi (IISc; third-party HF mirror), OpenSLR-104 (MUCS 2021 human
  code-switched, 16 kHz). `ai4bharat/Rasa` has **no Hindi**; IndicVoices-R/Svarah/Vaani are gated for this
  account; IndicCMix audio is F5-TTS synthetic; FLEURS has no `en_in`. See `data_pipeline/source_probe.json`.
- **Final dataset**: 7,360 utterances, 10.20 h, 201 speakers; train 6,980 / validation 380; Hindi 4.59 h,
  Indian English 2.48 h, code-switched 3.13 h. Private HF repo `MSRIT-NBM/iisc-hinglish-pocket-tts-dataset`
  (verified readable with the evaluators' read-only token; anonymous access refused).
- **Deviation from the brief, deliberate**: loudness is active-speech **RMS -20 dBFS with a -1 dBFS peak
  ceiling**, not literal peak normalisation to -20 dBFS (which would put speech ~15 dB below normal corpora and
  Pocket TTS's training audio). `configs/data_pipeline.yaml: audio.loudness.mode: peak` gives the literal
  version.
- **Digits**: as in the brief's examples, `text_roman` / `text_code_switched` keep digits ("7 baje") while
  `text_devanagari` verbalises them; training manifests use the fully verbalised `spoken_*` forms (word i of
  all three is the same spoken word).
- **Terminal punctuation** is always present (`.`/`?`/`!`; `।` -> `.`), matching what Pocket TTS appends at
  inference; the model config also maps `।` -> `.` via `replace_characters`.
- **Alignment filter**: 240 of 7,600 selected rows (209 from OpenSLR-104) had implausible word boundaries
  (transcript does not match audio) and were dropped from both the dataset and the manifests.

## 5. Artifacts on the GPU workstation (`$WORK = ~/vaani_pocket_tts_work`)

| Path | Content |
|---|---|
| `data/processed/audio/*.wav` | 24 kHz mono PCM-16 (7,600 selected; the 7,360 in `final.jsonl` are used) |
| `data/processed/{candidates,selected,final}.jsonl` | every measured row / quota selection / final dataset |
| `data/manifests/{train,valid}_{deva,roman,cs}[_aligned].jsonl` | pocket-tts manifests per script |
| `data/manifests/{train,valid}_mix_aligned.jsonl` | all three scripts (16,045 train rows / 871 valid rows) |
| `data/manifests/*_latents.jsonl` + `latents/29335a05/` | Mimi latents (hash of the 2026-04 24L weights' Mimi) |
| `data/voice_prompts/` | 22 reference prompts (`index.json`, WAV + latents) |
| `repo/tokenizer/hinglish_bilingual_4k.{model,json}` | bilingual tokenizer (same files as `tokenizer/` here) |

## 6. Starting the fine-tune

```bash
cd ~/vaani_pocket_tts_work/pocket-tts && source ../env.sh
python -m training.train ~/vaani_pocket_tts_work/repo/configs/finetune_hinglish_24l.yaml
```

`data_pipeline/training_readiness.json` records the dry run: config parses, tokenizer = 4000 bins, latent
cache fresh, official DataLoader batches, warm-started 24L model (316 M trainable params) gives a finite loss
(~0.42) using ~2.5 GiB at batch 4. Points to decide for the fine-tune:

- **Data scale**: 10 h is far below the README's 100 h guidance for a new language. Watch validation loss and
  listen to `runs/.../samples/` early; expect to stop well before `max_steps: 20000`.
- **Text-embedding init**: `reset_text_embedding: true` starts the table from random. A cheaper start is
  copying the English teacher's rows for the Latin pieces both tokenizers share (needs a small builder change).
- **Distillation** to 6 layers (`depth_distill.yaml`) for the CPU-friendly release model.
- **Evaluation**: WER with a Hindi/English ASR on the 380 validation rows in each script, speaker similarity
  with the voice prompts, UTMOS; compare against saryps/Indus on Roman and mixed-script inputs.
- The front-end for inference is `HinglishTextProcessor.normalize_for_tts(text, script=...)` (numbers
  verbalised, punctuation normalised, any script in, chosen script out).
