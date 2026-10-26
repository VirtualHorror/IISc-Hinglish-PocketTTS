# IISc Hinglish Pocket TTS

Fine-tuning [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for Indian **Hinglish**: natural
Indian-accented Hindi and English with smooth code-switching, readable from **Devanagari, Roman Hinglish, or
mixed-script** input (IISc Speech AI internship, round 2).

**Status: Sessions 1–2 of 3 done.**
- Session 1: data pipeline, human-verified dataset (private HF repo `MSRIT-NBM/iisc-hinglish-pocket-tts-dataset`,
  7,360 utterances / 10.2 h), dual-script text normaliser, bilingual 4,000-piece tokenizer, word alignments and
  Mimi latents ([`docs/session1_notes.md`](docs/session1_notes.md)).
- Session 2: warm-started multi-script fine-tune of the 24-layer teacher on one RTX 4090, round-trip ASR
  monitoring, objective benchmark, and the private HF model repo `MSRIT-NBM/pocket-tts-indian-hinglish`
  ([`docs/session2_notes.md`](docs/session2_notes.md)).
- Session 3: demo app, and restoring the GPU workstation.

## Layout

| Path | What |
|---|---|
| `data_pipeline/hinglish_text_processor.py` | Devanagari / Roman Hinglish / code-switched renderings, number verbalisation, script ID, loanword detection (`hinglish_lexicon.py`) |
| `data_pipeline/audio_processing.py` | 24 kHz resampling, dead-air trimming with a 180 ms end-on-pause tail, loudness, SNR/clipping metrics |
| `data_pipeline/probe_sources.py`, `download_sources.py` | which public datasets are usable; fetch the selected shards |
| `data_pipeline/prepare_dataset.py` | filter, select, process and annotate (-> `dataset_stats.json`) |
| `data_pipeline/align_words.py`, `finalize_dataset.py` | CTC word alignment, Pocket TTS manifests, final dataset |
| `data_pipeline/train_tokenizer.py` | bilingual SentencePiece (-> `tokenizer/hinglish_bilingual_4k.*`) |
| `data_pipeline/push_to_hub.py`, `dataset_card.py`, `verify_read_access.py` | private HF dataset, card, read-token check |
| `data_pipeline/extract_mimi_latents.py`, `verify_training_ready.py` | latent cache + voice prompts; training dry run |
| `data_pipeline/inspect_checkpoints.py` | tokenizer/weights comparison of existing Pocket TTS checkpoints |
| `training/train_hinglish.py` | Session 2 entrypoint: the official pocket-tts training loop with the hooks below |
| `training/warm_start.py` | text-embedding warm start from the English teacher (exact / decomposed / transliterated pieces) |
| `training/param_groups.py` | differential learning rates, staged unfreeze, per-group cosine schedule |
| `training/eos_metrics.py` | EOS accuracy at the inference threshold (validation) |
| `training/asr_monitor.py` | round-trip ASR (Whisper) of every checkpoint through the inference path |
| `training/summarize_runs.py` | `training_summary.json`: validation + ASR curves of the runs |
| `eval/asr.py`, `eval/evaluate_hinglish_tts.py` | Whisper + script-invariant CER/WER; the 24-sentence benchmark (`evaluation_results.json`) |
| `release/` | `export_release.py` (checkpoint -> HF repo folder), `inference.py` (shipped), `model_card.py`, `push_model.py`, `verify_model_access.py` |
| `configs/` | pipeline config, Hinglish model config (24L), `finetune_hinglish.yaml` (Session 2), `ablation_official_recipe.yaml` |
| `tests/` | `pytest tests/` (text processor, audio processing, training helpers, inference front-end, ASR scoring) |
| `scripts/gpu_sync.sh` | push code to / pull results and artifacts from the GPU workstation |
| `artifacts/` (git-ignored) | exported model, voices and generated audio (`scripts/gpu_sync.sh artifacts`) |

## Pipeline (on the GPU workstation, after `source ~/vaani_pocket_tts_work/env.sh`)

```bash
# Session 1: data
python -m data_pipeline.probe_sources
python -m data_pipeline.download_sources
python -m data_pipeline.prepare_dataset
python -m data_pipeline.align_words
python -m data_pipeline.finalize_dataset
python -m data_pipeline.train_tokenizer
python -m data_pipeline.extract_mimi_latents
python -m data_pipeline.push_to_hub && python -m data_pipeline.verify_read_access
python -m data_pipeline.verify_training_ready --forward

# Session 2: train, monitor, select, export, evaluate, publish
python -m training.warm_start configs/finetune_hinglish.yaml --compare
python -m training.train_hinglish configs/finetune_hinglish.yaml
python -m training.asr_monitor --run-dir $WORK/runs/hinglish_24l_ws
python -m training.asr_monitor --run-dir $WORK/runs/hinglish_24l_ws --monitor-set eval/selection_set.json --tag selection --steps 2500,3000,3500,4000,4500
python -m release.export_release --checkpoint $WORK/runs/hinglish_24l_ws/checkpoint_<step>.pt
python -m eval.evaluate_hinglish_tts --model-dir $WORK/release
python -m release.model_card && python -m release.push_model && python -m release.verify_model_access
```

Credentials live only in the git-ignored `.env` (`HF_WRITE_TOKEN`, `HF_READ_TOKEN`, GPU SSH details).
