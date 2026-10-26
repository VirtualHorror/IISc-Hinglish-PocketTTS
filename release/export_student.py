"""Export a distilled CPU student into the release folder, next to the 24-layer model:

  student_<n>l/model.safetensors   EMA weights in pocket-tts inference format (flow_lm.* + mimi.*, float32)
  student_<n>l/config.yaml         the student's model config with hf:// paths (shared tokenizer.model)
  student_<n>l/voices/             the release's studio prompts, with voice states precomputed by the student
  student_<n>l/export_info.json    source checkpoint, sha256 of every file, a reload-and-speak check

inference.py loads it with HinglishTTS.from_pretrained(..., variant="<n>l") (CLI: --variant <n>l); the
front-end, tokenizer and voices are the same as the 24-layer model's.

    python -m release.export_student --checkpoint $WORK/runs/hinglish_6l_distill/checkpoint_<step>.pt \
        --model-config configs/hinglish_pocket_tts_6l.yaml [--release $WORK/release]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import safetensors.torch
import torch
import yaml

from data_pipeline.common import REPO_ROOT, WORK, setup_logging, write_json
from release.export_release import REPO_ID, sha256

logger = setup_logging("export_student")


def student_config(src: Path, folder: str, layers: int) -> str:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["weights_path"] = f"hf://{REPO_ID}/{folder}/model.safetensors"
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = f"hf://{REPO_ID}/tokenizer.model"
    header = (f"# Pocket TTS Indian Hinglish, {layers}-layer CPU student (FlowLM + Mimi, 24 kHz): depth-distilled from\n"
              "# the 24-layer Hinglish model (same tokenizer, front-end and voices; less compute per audio frame).\n"
              f"#   pocket-tts generate --config hf://{REPO_ID}/{folder}/config.yaml --voice <prompt.wav> --text \"...\"\n")
    return header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--release", type=Path, default=WORK / "release")
    cli = ap.parse_args()

    from pocket_tts.models.model_state import export_model_state
    from pocket_tts.models.tts_model import TTSModel

    tts = TTSModel.load_model(config=str(cli.model_config), checkpoint=str(cli.checkpoint))  # EMA weights, CPU
    layers = len(tts.flow_lm.transformer.layers)
    folder = f"student_{layers}l"
    out = cli.release / folder
    if out.exists():
        shutil.rmtree(out)
    (out / "voices").mkdir(parents=True)

    state = {k: v.detach().float().contiguous().cpu() for k, v in tts.state_dict().items()}
    step = int(torch.load(cli.checkpoint, map_location="cpu", weights_only=True, mmap=True)["step"])
    safetensors.torch.save_file(state, str(out / "model.safetensors"), metadata={
        "format": "pt", "source_checkpoint": cli.checkpoint.name, "step": str(step), "weights": "ema",
        "distilled_from": "24-layer Pocket TTS Indian Hinglish (step 3500 EMA)"})
    logger.info(f"{folder}/model.safetensors: {len(state)} tensors, {sum(v.numel() for v in state.values()) / 1e6:.1f}M params")
    (out / "config.yaml").write_text(student_config(cli.model_config, folder, layers), encoding="utf-8")

    index = json.loads((cli.release / "voices" / "index.json").read_text(encoding="utf-8"))
    for v in index:
        wav = out / "voices" / f"{v['name']}.wav"
        shutil.copy2(cli.release / "voices" / f"{v['name']}.wav", wav)
        export_model_state(tts.get_state_for_audio_prompt(wav), out / "voices" / f"{v['name']}.safetensors")
    write_json(out / "voices" / "index.json", index)
    del tts

    # reload through the release's own inference.py, as an evaluator would, and speak once
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(cli.release))
    from inference import HinglishTTS

    variant = f"{layers}l"
    check = HinglishTTS.from_pretrained(str(cli.release), variant=variant)
    text = "कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो"
    audio = check.synthesize(text, voice=index[0]["name"])
    check.save(audio, out / "export_check.wav")
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    write_json(out / "export_info.json", {
        "repo_id": REPO_ID, "folder": folder, "variant": variant, "layers": layers,
        "source_checkpoint": str(cli.checkpoint.relative_to(WORK)) if cli.checkpoint.is_relative_to(WORK) else cli.checkpoint.name,
        "step": step, "weights": "EMA", "model_config": str(cli.model_config.resolve().relative_to(REPO_ROOT)),
        "exported": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "check": {"text": text, "model_input": check.normalize(text), "seconds": round(len(audio) / check.sample_rate, 2)},
        "files": files})
    logger.info(f"{folder} ready: {len(files)} files; check audio {len(audio) / check.sample_rate:.2f} s")


if __name__ == "__main__":
    main()
