"""Export a training checkpoint as the standalone Pocket TTS Indian Hinglish release folder
(the layout of the Hugging Face model repo):

  model.safetensors   EMA weights in pocket-tts inference format (flow_lm.* + mimi.*, float32),
                      Mimi encoder included, so any .wav can be cloned
  tokenizer.model     the bilingual SentencePiece model (md5 checked against tokenizer/*.json)
  config.yaml         the model config with hf://MSRIT-NBM/pocket-tts-indian-hinglish/ paths, so
                      `pocket-tts generate --config hf://.../config.yaml` works with a read token
  voices/<name>.wav   clean 24 kHz studio prompts (IndicTTS / SYSPIN voice artists; chosen by
                      release.select_voice_prompts when it has run), with
  voices/<name>.safetensors  the prompt's model state precomputed by this model (export-voice
                      format: loads instantly), and voices/index.json (provenance)
  hinglish_text/      the text normaliser inference.py uses for auto_normalize
  inference.py, requirements.txt, export_info.json (source checkpoint, sha256 of every file)

The folder is then reloaded through inference.py and one sentence is synthesised as a check.

    python -m release.export_release --checkpoint $WORK/runs/<run>/checkpoint_<step>.pt [--out $WORK/release]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import safetensors.torch
import soundfile as sf
import torch
import yaml

from data_pipeline.common import REPO_ROOT, WORK, setup_logging, write_json

logger = setup_logging("export_release")

REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
MODEL_CONFIG = REPO_ROOT / "configs" / "hinglish_pocket_tts_24l.yaml"
TOKENIZER = REPO_ROOT / "tokenizer" / "hinglish_bilingual_4k.model"
TOKENIZER_MD5 = "2dfe037602299f52e7da772bc004444c"  # data_pipeline/train_tokenizer.py output used in training
PROMPTS = WORK / "data" / "voice_prompts"
SEARCHED_PROMPTS = WORK / "data" / "voice_prompts_s2"  # release.select_voice_prompts
# release name -> (Session 1 prompt, provenance)
VOICES = {
    "indian_female_ananya": ("indictts_hindifull_female__hi", "IndicTTS Hindi female voice artist (IIT Madras), 48 kHz studio source"),
    "indian_male_aarav": ("indictts_hindifull_male__hi", "IndicTTS Hindi male voice artist (IIT Madras), 48 kHz studio source"),
    "indian_female_meera": ("syspin_hindi_female__hi", "SYSPIN Hindi female voice artist (IISc SPIRE Lab), 44.1 kHz studio source"),
    "indian_male_vikram": ("syspin_hindi_male__hi", "SYSPIN Hindi male voice artist (IISc SPIRE Lab), 44.1 kHz studio source"),
}
REQUIREMENTS = """pocket-tts>=4.0
huggingface_hub
soundfile
pyyaml
# optional, for auto_normalize (numbers, script conversion):
cmudict
wordfreq
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_config(src: Path) -> str:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["weights_path"] = f"hf://{REPO_ID}/model.safetensors"
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = f"hf://{REPO_ID}/tokenizer.model"
    header = ("# Pocket TTS Indian Hinglish (24-layer FlowLM + Mimi, 24 kHz). Fine-tuned from\n"
              "# kyutai english_2026-04_24l on Devanagari, Roman Hinglish and code-switched text.\n"
              f"#   pocket-tts generate --config hf://{REPO_ID}/config.yaml --voice <prompt.wav> --text \"...\"\n")
    return header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def bundle_text_processor(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    src = REPO_ROOT / "data_pipeline"
    shutil.copy2(src / "hinglish_lexicon.py", dst / "hinglish_lexicon.py")
    code = (src / "hinglish_text_processor.py").read_text(encoding="utf-8")
    old = "from data_pipeline import hinglish_lexicon as lex"
    assert old in code, "import line changed: update bundle_text_processor"
    (dst / "hinglish_text_processor.py").write_text(code.replace(old, "from . import hinglish_lexicon as lex"),
                                                    encoding="utf-8")
    (dst / "__init__.py").write_text('"""Hinglish text normaliser (from the training repo\'s data_pipeline/)."""\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=WORK / "release")
    ap.add_argument("--voices", default=",".join(VOICES))
    cli = ap.parse_args()

    from pocket_tts.models.model_state import export_model_state
    from pocket_tts.models.tts_model import TTSModel

    out = cli.out
    if out.exists():
        shutil.rmtree(out)
    (out / "voices").mkdir(parents=True)

    tts = TTSModel.load_model(config=str(MODEL_CONFIG), checkpoint=str(cli.checkpoint))  # EMA weights, CPU
    state = {k: v.detach().float().contiguous().cpu() for k, v in tts.state_dict().items()}
    step = int(torch.load(cli.checkpoint, map_location="cpu", weights_only=True, mmap=True)["step"])
    safetensors.torch.save_file(state, str(out / "model.safetensors"), metadata={
        "format": "pt", "source_checkpoint": cli.checkpoint.name, "step": str(step), "weights": "ema"})
    logger.info(f"model.safetensors: {len(state)} tensors, {sum(v.numel() for v in state.values()) / 1e6:.1f}M params")

    shutil.copy2(TOKENIZER, out / "tokenizer.model")
    md5 = hashlib.md5((out / "tokenizer.model").read_bytes()).hexdigest()
    if md5 != TOKENIZER_MD5:
        raise SystemExit(f"tokenizer md5 {md5} != {TOKENIZER_MD5}, the model the checkpoints were trained with")
    (out / "config.yaml").write_text(release_config(MODEL_CONFIG), encoding="utf-8")

    index = []
    search_path = REPO_ROOT / "eval" / "results" / "voice_prompt_search.json"
    search = json.loads(search_path.read_text())["voices"] if search_path.exists() else {}
    session1 = {r["name"]: r for r in json.loads((PROMPTS / "index.json").read_text())}
    for name in [v for v in cli.voices.split(",") if v]:
        prompt, provenance = VOICES[name]
        chosen = SEARCHED_PROMPTS / f"{name}.wav"
        if name in search and chosen.exists():  # release.select_voice_prompts winner
            src, row = chosen, search[name]
            meta = {"speaker_id": row["speaker_id"], "source_utt_id": row["chosen_source_utt_id"],
                    "prompt_text": row["chosen_text"], "prompt_kind": row["chosen"]}
        else:
            src, row = PROMPTS / f"{prompt}.wav", session1[prompt]
            meta = {"speaker_id": row["speaker_id"], "source_utt_id": row["source_utt_id"],
                    "prompt_text": row["text"], "prompt_kind": "session1_sentence"}
        shutil.copy2(src, out / "voices" / f"{name}.wav")
        export_model_state(tts.get_state_for_audio_prompt(out / "voices" / f"{name}.wav"),
                           out / "voices" / f"{name}.safetensors")
        duration = round(sf.info(str(out / "voices" / f"{name}.wav")).duration, 2)
        index.append({"name": name, "source": provenance, **meta, "duration_s": duration, "sample_rate": 24000})
    write_json(out / "voices" / "index.json", index)

    bundle_text_processor(out / "hinglish_text")
    shutil.copy2(REPO_ROOT / "release" / "inference.py", out / "inference.py")
    (out / "requirements.txt").write_text(REQUIREMENTS)
    del tts

    # reload exactly as an evaluator would (local folder instead of the hub) and speak once
    import sys

    sys.dont_write_bytecode = True  # no __pycache__ inside the release folder
    sys.path.insert(0, str(out))
    from inference import HinglishTTS  # the copy inside the release folder

    check = HinglishTTS.from_pretrained(str(out))
    text = "कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो"
    audio = check.synthesize(text, voice=index[0]["name"])
    check.save(audio, out / "export_check.wav")
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    write_json(out / "export_info.json", {
        "repo_id": REPO_ID, "source_checkpoint": str(cli.checkpoint), "step": step, "weights": "EMA",
        "exported": time.strftime("%Y-%m-%d %H:%M:%S %z"), "tokenizer_md5": md5,
        "check": {"text": text, "model_input": check.normalize(text), "seconds": round(len(audio) / check.sample_rate, 2)},
        "files": files})
    logger.info(f"release folder ready: {out} ({len(files)} files); check audio {len(audio) / check.sample_rate:.2f} s")


if __name__ == "__main__":
    main()
