# Training and evaluation notes: fine-tuning, monitoring, evaluation and the private model repo

All numbers come from files in this repo, generated on the GPU workstation: `training/warm_start_report.json`,
`training/training_summary.json`, `eval/results/asr_monitor_*.json`, `eval/evaluation_results.json`,
`release/push_report.json`, `release/model_access_verification.json`.

## 1. Training code (`training/`)

`python -m training.train_hinglish configs/finetune_hinglish.yaml` runs pocket-tts's own loop
(`$POCKET_TTS_REPO/training/train.py`, untouched: data loading, CFG dropout, LSD flow + EOS loss, EMA,
checkpoints, `model.safetensors` export, SIGTERM resume) with five hooks swapped in. Our `training/`
package shares the official package's name and extends its `__path__` with the checkout, so
`training.train`, `training.modules.*`, ... still resolve to pocket-tts while `training.warm_start` etc.
resolve here. The config is an official `TrainArgs` YAML plus a `hinglish:` section that the hooks read
(the official parser never sees it).

| hook | file | what it does |
|---|---|---|
| `build_models` | `warm_start.py` | rebuilds the 4,001 x 1,024 text table from the English teacher's table |
| `build_optimizer` | `param_groups.py` | three AdamW groups: text table / backbone / heads (table without weight decay) |
| `lr_at` | `param_groups.py` | staged unfreeze + per-group linear warmup and cosine decay to 10 % |
| `validate` | `eos_metrics.py` | per-script validation (Devanagari / Roman / code-switched manifests, identical cuts) + EOS accuracy at the inference threshold |
| `write_samples` | `train_hinglish.py` | held-out sentences in all three scripts with two studio voice prompts |

The process ends with `os._exit`: pocket-tts's `SubprocessDataLoader.close()` SIGKILLs its loaders, which
orphans the `torch_shm_manager` processes they spawned; those keep the multiprocessing resource tracker's
pipe open, so a normal interpreter exit waits forever while holding ~14 GB of GPU memory (seen in the dry
run; `gpu_state_log.md` row 18).

### Warm-started text embeddings (`training/warm_start.py`)

`reset_text_embedding: true` (the official new-tokenizer recipe) leaves `nn.Embedding`'s N(0, 1) init: rows
of norm ~32, while the teacher's trained rows have median norm 0.45. Every piece of our tokenizer instead
gets a row in the teacher's space:

- **exact** (868): the piece exists in the teacher's tokenizer with a trained row (norm >= 0.05; 314 teacher
  rows are ~0: byte fallback and foreign-script pieces English data never used).
- **decomposed** (1,069): Latin / digit / punctuation pieces -> the teacher's own Unigram segmentation
  (`tokenizers` model.tokenize, raw, no normaliser) -> mean of the parts' rows.
- **transliterated** (1,863): Devanagari pieces -> casual romanisation (`romanize_hindi`, with leading vowel
  signs / virama / nasal marks given a readable start) -> as decomposed. `▁के` and `▁ke` get the same row.
- **random** (200): byte / control pieces without a trained teacher row -> N(0, sigma_teacher).

Composed rows are rescaled to their parts' mean norm. Indus-Labs/indus-pocket-tts's Hindi table was tested as
a source for Devanagari pieces: a scaled orthogonal Procrustes map fitted on 800 shared pieces reaches a
held-out cosine of 0.053 (shuffled 0.006), so the spaces are unrelated and it was not used. Step-0 validation
on 640 rows (identical batches and noise): official N(0, 1) loss 0.4536 (EOS 0.1736) vs warm start 0.4209
(EOS 0.1230).

### Multi-script data

`train_mix_aligned.jsonl` (built by the data pipeline) already holds every distinct rendering of every training utterance:
16,045 rows from 6,980 utterances (Devanagari 9.67 h, Roman 9.67 h, code-switched 2.96 h of text-audio
pairs). Renderings identical to another script's (code-switched = Devanagari for all-Hindi sentences,
code-switched = Roman for all-English ones) are kept once.

### Schedule (`configs/finetune_hinglish.yaml`)

| group | params | peak lr | start | warmup |
|---|---|---|---|---|
| text_embedding | 4.1 M | 5e-4 | 0 | 100 |
| backbone (transformer, input/speaker projections, BOS) | 302.2 M | 1e-4 | 400 | 400 |
| heads (LSD flow net, EOS head, objective) | 9.8 M | 1e-4 | 400 | 400 |

Cosine decay to 10 % of each peak at step 5,000; batch 16 x 4 accumulation (effective 64), bf16 autocast,
EMA 0.999, latent-stat EMA for 1,000 steps, CFG dropout 0.2/0.2, flow-batch multiplier 4. No
`torch.compile` (3.5 it/s without it on the RTX 4090; compilation costs ~9 min, a wash for a 25 min run).
Peak GPU memory 13.2 GiB.

## 2. Monitoring and evaluation (`eval/`)

- `eval/asr.py`: Whisper large-v3-turbo (fp16, revision 41f01f3) with the language set per utterance (`hi`;
  `en` for English-matrix utterances: forcing `hi` on English speech makes Whisper translate, which gave a
  bogus 52 % "floor" before the fix). Reference and hypothesis -> `HinglishTextProcessor.annotate(...)
  .spoken_devanagari` -> fold (nukta, chandrabindu, candra vowels, nasal+virama, punctuation; keeps vowel
  signs, which Python's `re` `\w` would drop). CER without spaces, WER over words, corpus level.
- `training/asr_monitor.py`: every checkpoint through the release path (`TTSModel`, EMA weights, temperature
  0.3, EOS threshold -4, no CFG), studio prompts `indictts_hindifull_{female,male}__hi`, plus the ASR floor
  (Whisper on the real recordings). Monitor set `eval/monitor_set.json` (6 code-switched validation
  utterances x 3 scripts x 2 voices = 36 per checkpoint); selection set `eval/selection_set.json`
  (30 validation utterances: 10 Hindi, 10 English, 10 code-switched, seeded; 70 distinct renderings x 2
  voices = 140 per checkpoint).
- `eval/evaluate_hinglish_tts.py`: the 24-sentence benchmark (`eval/benchmark_sentences.json`) through
  `release/inference.py` (what evaluators run): default front-end, raw text, script invariance; speed (RTF,
  time to first audio) on the GPU and on the CPU (1 / 4 threads, fp32 / int8).

## 3. Training results (`training/training_summary.json`)

Main run `hinglish_24l_ws` (`configs/finetune_hinglish.yaml`): 5,000 steps in 26.8 min (3.5 it/s). Ablation
`hinglish_24l_official` (`configs/ablation_official_recipe.yaml`): the official `finetune_language` settings
(N(0, 1) text table, all groups lr 2e-4 after 1,000 warmup steps, constant) on the same data / batch / steps.

| step | valid loss ours | valid loss official | monitor CER ours | monitor CER official | selection CER / WER ours |
|---|---|---|---|---|---|
| 500 | 0.362 | 0.385 | 62.4 | 104.6 | |
| 1000 | 0.302 | 0.359 | 36.1 | 89.1 | |
| 1500 | 0.282 | 0.310 | 15.4 | 84.6 | |
| 2000 | **0.273** | 0.269 | 8.3 | 83.1 | |
| 2500 | 0.285 | **0.254** | 7.5 | 77.5 | 8.50 / 19.99 |
| 3000 | 0.303 | 0.268 | 5.8 | 67.2 | 7.99 / 19.07 |
| 3500 | 0.340 | 0.275 | 6.3 | 65.9 | **7.99 / 18.76** |
| 4000 | 0.396 | 0.310 | **5.7** | 64.5 | 8.61 / 19.56 |
| 4500 | 0.437 | 0.362 | 7.2 | 64.3 | 8.56 / 19.25 |
| 5000 | 0.485 | 0.387 | 7.9 | 69.0 | |

ASR floor (Whisper on the real recordings): monitor set CER 5.43 / WER 15.29; selection set CER 5.27 /
WER 13.66 (English 0.9, Hindi 5.9, code-switched 7.3 CER).

- **Warm start + staged differential lr is decisive.** The official recipe reaches a *lower* teacher-forced
  validation loss but never reads the text: monitor CER >= 64 % at every step, confirmed on the trainer's own
  samples (raw weights, CFG 2: 61-83 % vs ours 6.4-8.4 % from step 2,000; `eval/results/inloop_samples_asr.json`).
  Teacher forcing lets a model predict frames from the audio alone, so with 10 h of data the randomly
  initialised text rows never get wired in, and the loss does not show it. Checkpoints are therefore selected
  by round-trip ASR, never by validation loss.
- **Overfitting:** our validation loss bottoms out at step 2,000 (train loss -> ~0 later: memorisation), but
  intelligibility keeps improving until ~3,000-4,000 and then degrades.
- **Script invariance:** validation loss per script is the same within 0.003 at every step (step 2,000:
  Devanagari 0.272, Roman 0.274, code-switched 0.273).
- **EOS:** no generation without EOS in any monitored checkpoint of our run (0 / 36 and 0 / 140); EOS frame
  accuracy at the -4 threshold ~0.989; the head fires on average 0.8-1.5 frames from the true end.
- **Selection rule, fixed before scoring:** lowest selection-set CER, WER within 0.3 pp breaks ties, then the
  earlier step. Steps 3,000 and 3,500 tie at 7.99 % CER; **step 3,500 (EMA)** has the lower WER and is released.
- Remaining errors concentrate on rare English/technical words and names (Integer, bronchial, Esther), short
  Roman first words ("Ve", "Jab", "Mainne"), and Whisper spelling variants; 1 / 140 generations (male prompt)
  starts by repeating the prompt's last words ("...मुस्कुरा देती है").

## 4. Release (`release/`)

- `release.export_release --checkpoint .../checkpoint_00003500.pt` -> `$WORK/release/`: `model.safetensors`
  (EMA, flow_lm + mimi, float32, 336.1 M params, Mimi encoder included), `tokenizer.model` (md5
  2dfe0376...), `config.yaml` with `hf://MSRIT-NBM/pocket-tts-indian-hinglish/...` paths, `voices/`
  (`.wav` prompt + precomputed `.safetensors` state per voice, `index.json`), `hinglish_text/` (the
  normaliser, import rewritten to be package-relative), `inference.py`, `requirements.txt`,
  `export_info.json` (sha256 of every file). It reloads the folder through `inference.py` and speaks once.
- **Voice prompts** (`release.select_voice_prompts`, `eval/results/voice_prompt_search.json`): per speaker the
  original clip, another whole sentence and two training-style prefixes (utterance start cut at a word
  boundary 3.5-5 s in) scored on the selection set (70 texts). ananya keeps the original clip; meera
  (10.5 -> 9.2 % CER) and vikram (9.4 -> 8.6 %) take prefixes; aarav's original clip (best CER, 7.1 %) was
  disqualified because at step 3,500 it made 1 / 70 selection generations open with its own last words
  ("मुस्कुरा देती है"), so aarav takes its best leak-free prefix (7.3 %, lower WER).
- **Front-end default** (`release/inference.py`): danda -> period and a period appended when `. ? !` are
  missing (always); `auto_normalize=True` + `script="devanagari"` by default (numbers verbalised, English words
  spelled phonetically in Devanagari). Chosen from the first benchmark run: rendering a sentence in Devanagari
  beat its Roman / code-switched renderings (invariance CER 7.2 vs 10.3 / 9.9 %), and against the
  keep-the-script front-end it won on 10 of 24 sentences, tied on 10, lost on 4 (mean CER 9.1 -> 7.7 %).
  `script="auto"` keeps the typed script; `auto_normalize=False` sends it as typed.

## 5. Benchmark (`eval/evaluation_results.json`)

24 sentences (`eval/benchmark_sentences.json`, 8 per script, written for the benchmark) x 4 voices, step-3,500
EMA model through `release/inference.py`, Whisper large-v3-turbo (`hi`). No generation without EOS (0 / 360).

| front-end | Roman | Devanagari | code-switched | all: CER / WER % |
|---|---|---|---|---|
| default: `auto_normalize`, rendered in Devanagari | 9.6 | 5.9 | 5.9 | **7.2 / 17.6** |
| `auto_normalize`, script kept (`script="auto"`) | 13.6 | 5.2 | 12.1 | 10.4 / 21.6 |
| raw text as typed | 18.8 | 6.3 | 11.8 | 12.4 / 24.2 |
| invariance: each sentence rendered as ... (1 voice) | 10.5 | 8.4 | 8.3 | 9.1 / 19.5 |

Per voice (default): ananya 6.0 / 15.2, aarav 6.3 / 16.0, vikram 7.4 / 16.7, meera 9.0 / 22.4. Whisper's own
error on real held-out recordings: 5.3 / 13.7. The first benchmark run (original prompts, script kept by
default) gave 11.3 / 22.1; the voice-prompt search and the Devanagari default were then decided on the
selection set and the invariance comparison, and the benchmark was run once more.

Speed (24 sentences, voice state cached, after warm-up; RTF = synthesis / audio time):

| device | RTF | x real time | first audio p50 / p90 (ms) |
|---|---|---|---|
| RTX 4090, fp32 | 0.126 | 7.9 | 30 / 37 |
| i9-13900K, fp32, 1 thread | 0.704 | 1.4 | 237 / 265 |
| i9-13900K, fp32, 4 threads | 0.499 | 2.0 | 129 / 143 |
| i9-13900K, int8, 1 thread | 0.323 | 3.1 | 109 / 121 |
| i9-13900K, int8, 4 threads | 0.221 | 4.5 | 74 / 83 |
| local Xeon 8259CL (2 cores), int8, 1 thread | 1.11 | 0.9 | 292 (single sentence, not the benchmark) |

## 6. Hugging Face model repo

`MSRIT-NBM/pocket-tts-indian-hinglish`, **private**, 126 files / 1.42 GB, latest commit `828e3b6`
(`release/push_report.json`): `model.safetensors`, `config.yaml` (hf:// paths), `tokenizer.model`,
`inference.py`, `requirements.txt`, `hinglish_text/`, `voices/` (4 voices: wav + state), `samples/` (96 wav +
`index.json`), `evaluation_results.json`, `eval/` (sentence sets), `training/` (configs, warm-start report,
training summary, voice-prompt search), `README.md` (model card rendered by `release/model_card.py`).

Read-only token check (`release/model_access_verification.json`): token role `read`; anonymous API refused
(`RepositoryNotFoundError`), anonymous HTTPS 401 for the API, `config.yaml` and `model.safetensors`; read token
200 / 200 / 302. In a fresh `python -I` process with only the read token and an empty HF cache: downloading
only `inference.py` (the card's quick start), importing it, loading the model (bundled normaliser found) and
synthesising "Kal subah 7 baje meri meeting hai, please reminder सेट कर दो" -> 4.1 s of audio, model sha256 =
export; the official `pocket-tts generate --config hf://.../config.yaml` CLI also synthesises.

That check found one front-end bug, fixed before the final upload: in mixed-script text the normaliser treats
every Latin word as English, so Roman Hinglish with a few Devanagari words became "कैल सुबह सेवन बजे" ("Kal",
"7" read as English). `HinglishTTS.normalize` now romanises the Devanagari words when Latin words are the
majority; none of the 264 normalised benchmark inputs changed. Re-exporting gave tensor-identical weights
(only the safetensors header's metadata order differs).

Second upload (commit `e23fef3b`, 294 files / 2.22 GB): adds the 12-layer CPU student (`student_12l/`), the demo
studio (`demo_app.py`, `demo/` with 144 pre-rendered takes), `demo_hinglish_pocket_tts.mp4`, the distillation configs
and results, and the CPU speed probe; `inference.py` gained `variant=`. The read-token check was repeated against it
(`release/model_access_verification.json`, now also loading the student in the fresh process), and
`scripts/verify_read_token.py` checks both private repos with and without the token (`release/read_token_check.json`).

## 7. Local copies and tests

- `scripts/gpu_sync.sh pull` brings every generated record; `scripts/gpu_sync.sh artifacts` mirrors
  `$WORK/release` -> `artifacts/release/` and `$WORK/eval_audio` -> `artifacts/eval_audio/` (git-ignored).
- Local test environment without sudo / venv: `pip3 install --target .test_deps` (CPU torch from the PyTorch
  index, the rest from PyPI; git-ignored, excluded from `gpu_sync.sh push`), pocket-tts from the identical
  source in `outputs/reference/pocket-tts`:
  `PYTHONPATH=.test_deps:outputs/reference/pocket-tts python3 -m pytest -q -p no:cacheprovider tests/`
- Local synthesis from the artifacts works:
  `PYTHONPATH=.test_deps:outputs/reference/pocket-tts python3 artifacts/release/inference.py --repo artifacts/release --text "..."`.
- After the workstation was restored, the git-ignored `artifacts/` keeps the released folder (24-layer model,
  `student_12l/`, demo files), all generated audio, the training checkpoints behind both releases
  (`checkpoints/hinglish_24l_ws/checkpoint_00003500.pt`, `checkpoints/hinglish_12l_distill/checkpoint_00006000.pt`, with
  their run metadata) and the job logs. The studio runs locally with
  `PYTHONPATH=.test_deps:outputs/reference/pocket-tts python3 demo_app.py` (FastAPI and uvicorn are in `.test_deps`).

## 8. CPU speed, distilled students, demo studio, demo video, workstation restore

**Why the 24-layer model ran at RTF ~1.1 on the laptop.** The laptop is a 2-core / 4-thread Xeon 8259CL
(AWS VM, viewed over Chrome Remote Desktop, whose encoder itself takes 0.5-2 cores while the screen changes).
pocket-tts runs latent generation and Mimi decoding in two threads and calls `torch.set_num_threads(1)` when it
is imported. With 2 cores both threads are already busy, so more torch threads only add contention
(`scripts/cpu_speed_probe.py`, int8: 1 thread RTF 1.05, 2 threads 0.96, 4 threads 1.16 on a quiet screen; 1.24 /
1.52 for 1 / 2 threads with the screen in use, `eval/results/cpu_speed_local.json`). The bottleneck is the
24-layer FlowLM step: 75-93 ms (int8) or 155-182 ms (fp32) per 80 ms frame, against 26-40 ms for Mimi. Dynamic int8 halves the FlowLM step and is what `--quantize` / `quantize=True` does at load (~10 s
on this CPU). A pre-quantised `model_int8` checkpoint would run at exactly the same speed and only skip those
seconds, while torch's packed int8 weights are tied to the torch version, so none is shipped. `demo_app.py`
sets one torch thread per two physical cores (1 on this laptop, at most 4) after importing pocket-tts.

**Distilled students** (pocket-tts `depth_distill.yaml` recipe; `configs/distill_hinglish_{6l,12l}.yaml`;
student seeded from the teacher's non-backbone tensors plus its bottom and top layers; pure depth distillation,
no CFG; batch 32 x 2; validation in `train_hinglish.py` also scores the student's own flow / EOS losses through
the frozen heads). Selection-set round-trip ASR (`eval/results/asr_monitor_hinglish_*_distill_selection.json`,
same 140 generations as the teacher's selection, teacher = 7.99 / 18.76):

| student | steps | best CER / WER | at step | note |
|---|---|---|---|---|
| 6 layers | 8,000 | 15.4 / 27.8 | 6,000 | unstable: 44 % at step 5,000 and 29 % at 8,000 (a few English / Roman sentences babble); not shipped |
| 12 layers | 6,000 | 11.1 / 23.2 | 6,000 | released as `student_12l/` (`variant="12l"`) |

Benchmark of the released 12-layer student (`eval/results/evaluation_results_student_12l.json`, protocol of section
5): CER / WER 10.7 / 22.2 % with the default front-end (Roman 12.9, Devanagari 10.1, code-switched 8.8 % CER;
24-layer 7.2 / 17.6), 15.4 / 28.2 % for raw text, no generation without EOS. Speed: RTX 4090 RTF 0.079 (25 ms to
first audio); i9 CPU int8 4 threads 0.136 (48 ms); 2-core laptop int8 1 thread 0.71 (186 ms, vs 1.24 / 360 ms for
the 24-layer model).

**Demo studio** (`demo_app.py`, `demo/`): FastAPI + one page. Each model variant has a worker thread that
owns it (TTSModel is not thread-safe); the newest request stops the one in flight through pocket-tts's stop
event. An earlier version held a lock across the response generator's `yield`s and deadlocked when a client
disconnected mid-stream: Starlette never closed the abandoned generator. Preset takes are pre-rendered on the
GPU (`scripts/build_demo_cache.py`: 9 presets x 4 voices x 2 front-ends x 2 models, one fixed seed each, with
Whisper transcripts) and served in milliseconds; live takes are replayed from memory.

**Demo video** (`scripts/render_demo_video.py`): six showcase sentences synthesised on the RTX 4090 (the
soundtrack) and timed again on the i9 CPU with int8 weights, rendered with Pillow + raqm (correct Devanagari
shaping) and muxed by ffmpeg (H.264 1080p30, AAC 24 kHz); telemetry in `eval/results/demo_video.json`.

**Workstation restore.** After everything was copied back and published, `~/vaani_pocket_tts_work` was deleted
and the machine checked against the pre-project snapshot (`gpu_state_log.md`, git-ignored).
