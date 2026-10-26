# IISc Hinglish Pocket TTS

Fine-tuning [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for Indian **Hinglish**: natural
Indian-accented Hindi and English with smooth code-switching, readable from **Devanagari, Roman Hinglish, or
mixed-script** input (IISc Speech AI internship, round 2).

**Status: Session 1 of 3 done**: data pipeline, human-verified dataset (private HF repo
`MSRIT-NBM/iisc-hinglish-pocket-tts-dataset`, 7,360 utterances / 10.2 h), dual-script text normaliser,
bilingual 4,000-piece tokenizer, word alignments and precomputed Mimi latents. Session 2 trains, Session 3
evaluates and publishes. Technical notes: [`docs/session1_notes.md`](docs/session1_notes.md).

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
| `configs/` | pipeline config, Hinglish model config (24L), Session 2 training config |
| `tests/` | `pytest tests/` (text processor + audio processing) |
| `scripts/gpu_sync.sh` | push code to / pull artifacts from the GPU workstation |

## Pipeline (on the GPU workstation, after `source ~/vaani_pocket_tts_work/env.sh`)

```bash
python -m data_pipeline.probe_sources
python -m data_pipeline.download_sources
python -m data_pipeline.prepare_dataset
python -m data_pipeline.align_words
python -m data_pipeline.finalize_dataset
python -m data_pipeline.train_tokenizer
python -m data_pipeline.extract_mimi_latents
python -m data_pipeline.push_to_hub && python -m data_pipeline.verify_read_access
python -m data_pipeline.verify_training_ready --forward
```

Credentials live only in the git-ignored `.env` (`HF_WRITE_TOKEN`, `HF_READ_TOKEN`, GPU SSH details).
