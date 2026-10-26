"""Inspect existing Pocket TTS checkpoints: config, tokenizer coverage, safetensors layout.

Measures (instead of assuming) what each tokenizer can represent: script composition of the
vocabulary, standalone punctuation pieces (inference splits sentences on ".", "!", "?" tokens),
and how the same sentence tokenizes in Devanagari, Roman Hinglish and code-switched script
(byte-fallback / unknown pieces mean the model never saw those characters). Weight layouts come
from the safetensors headers only, so nothing large is downloaded.

    python -m data_pipeline.inspect_checkpoints [--ours path/to/tokenizer.model]
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download, parse_safetensors_file_metadata

from data_pipeline.common import REPO_ROOT, hf_token, setup_logging, write_json

logger = setup_logging("inspect_checkpoints")

CHECKPOINTS = {
    "kyutai_english_2026-04_24l (fine-tuning teacher)": {
        "config": ("local", "pocket_tts/config/english_2026-04_24l.yaml", None),
        "tokenizer": ("kyutai/pocket-tts-without-voice-cloning", "languages/english_2026-04_24l/tokenizer.json",
                      "00eac05ed3d16bdc3f6b5d598874019c34a89214"),
        "weights": ("kyutai/pocket-tts", "languages/english_2026-04_24l/model.safetensors",
                    "492522650173a0653b7575cdc25ae09810e5d741"),
    },
    "saryps-labs/pocket-tts-hindi": {
        "config": ("saryps-labs/pocket-tts-hindi", "config.yaml", None),
        "tokenizer": ("saryps-labs/pocket-tts-hindi", "tokenizer.model", None),
        "weights": ("saryps-labs/pocket-tts-hindi", "model.safetensors", None),
    },
    "Indus-Labs/indus-pocket-tts": {
        "config": ("Indus-Labs/indus-pocket-tts", "config.yaml", None),
        "tokenizer": ("Indus-Labs/indus-pocket-tts", "tokenizer.model", None),
        "weights": ("Indus-Labs/indus-pocket-tts", "model.safetensors", None),
    },
}

PROBES = {
    "devanagari": "कल सुबह सात बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो. आप कैसे हैं?",
    "roman": "Kal subah saat baje meri meeting hai, please reminder set kar do. Aap kaise hain?",
    "code_switched": "कल सुबह सात बजे मेरी meeting है, please reminder सेट कर दो. आप कैसे हैं?",
    "english": "Please set a reminder for my meeting at seven tomorrow morning. How are you?",
}
PUNCT = [".", ",", "?", "!", "।", ";", ":"]


def piece_class(piece: str) -> str:
    p = piece.replace("▁", "").replace("Ġ", "")
    if re.fullmatch(r"<0x[0-9A-F]{2}>", piece):
        return "byte"
    if piece in ("<unk>", "<s>", "</s>", "<pad>") or p.startswith("<"):
        return "special"
    if p == "":
        return "whitespace"
    if re.search(r"[ऀ-ॿ]", p):
        return "devanagari"
    if re.search(r"[A-Za-z]", p):
        return "latin"
    if re.fullmatch(r"\d+", p):
        return "digit"
    if re.fullmatch(r"[^\w\s]+", p):
        return "punctuation"
    return "other"


class Tok:
    """Uniform view over a sentencepiece .model or a tokenizers .json file."""

    def __init__(self, path: str):
        self.path = path
        if path.endswith(".json"):
            import tokenizers

            self.t = tokenizers.Tokenizer.from_file(path)
            self.kind = "tokenizers-json"
            self.pieces = [self.t.id_to_token(i) for i in range(self.t.get_vocab_size())]
        else:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=path)
            self.kind = "sentencepiece"
            self.pieces = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

    def encode(self, text: str) -> list[str]:
        if self.kind == "tokenizers-json":
            return self.t.encode(text).tokens
        return self.sp.encode(text, out_type=str)

    def n_unknown(self, text: str) -> int:
        """Pieces the model has no embedding for: <unk> ids (sentencepiece without byte fallback
        returns the raw surface string for these, so ids are the only reliable signal) or bytes."""
        if self.kind == "tokenizers-json":
            enc = self.t.encode(text)
            return sum(piece_class(t) in ("byte", "special") for t in enc.tokens)
        ids = self.sp.encode(text, out_type=int)
        return sum(i == self.sp.unk_id() or self.sp.is_byte(i) for i in ids)


def analyze_tokenizer(path: str) -> dict[str, Any]:
    tok = Tok(path)
    vocab = set(tok.pieces)
    comp = collections.Counter(piece_class(p) for p in tok.pieces)
    out: dict[str, Any] = {
        "format": tok.kind, "vocab_size": len(tok.pieces), "composition": dict(comp),
        "standalone_punctuation": {p: (p in vocab or ("▁" + p) in vocab) for p in PUNCT},
        "probes": {},
    }
    for name, text in PROBES.items():
        toks = tok.encode(text)
        out["probes"][name] = {"n_tokens": len(toks), "n_byte_or_unk": tok.n_unknown(text),
                               "tokens": toks[:40]}
    return out


def safetensors_layout(repo: str, filename: str, revision: str | None, token: str) -> dict[str, Any]:
    tensors = parse_safetensors_file_metadata(repo, filename, revision=revision, token=token).tensors
    groups: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"tensors": 0, "params": 0, "dtypes": set()})
    for name, info in tensors.items():
        top = ".".join(name.split(".")[:2])
        n = 1
        for d in info.shape:
            n *= d
        groups[top]["tensors"] += 1
        groups[top]["params"] += n
        groups[top]["dtypes"].add(info.dtype)
    total = sum(g["params"] for g in groups.values())
    interesting = {k: list(v.shape) for k, v in tensors.items()
                   if any(s in k for s in ("conditioner.embed", "speaker_proj", "bos_before_voice", "emb_mean", "emb_std", "out_eos"))}
    return {
        "n_tensors": len(tensors), "total_params_M": round(total / 1e6, 2),
        "groups": {k: {"tensors": v["tensors"], "params_M": round(v["params"] / 1e6, 3), "dtypes": sorted(v["dtypes"])}
                   for k, v in sorted(groups.items())},
        "key_shapes": interesting,
        "has_mimi_encoder": any(k.startswith("mimi.encoder.") for k in tensors),
        "transformer_layers": len({k.split(".")[3] for k in tensors if k.startswith("flow_lm.transformer.layers.")}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket-tts-repo", type=Path, default=Path.home() / "vaani_pocket_tts_work" / "pocket-tts")
    ap.add_argument("--ours", type=str, default=None, help="our tokenizer.model, analysed alongside")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data_pipeline" / "checkpoint_inspection.json")
    args = ap.parse_args()
    token = hf_token("write")
    report: dict[str, Any] = {}
    for name, spec in CHECKPOINTS.items():
        logger.info(f"inspecting {name}")
        entry: dict[str, Any] = {}
        where, path, _ = spec["config"]
        cfg_path = args.pocket_tts_repo / path if where == "local" else hf_hub_download(where, path, token=token)
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        fl = cfg["flow_lm"]
        entry["config"] = {
            "weights_path": cfg.get("weights_path"), "layers": fl["transformer"]["num_layers"],
            "d_model": fl["transformer"]["d_model"], "heads": fl["transformer"]["num_heads"],
            "flow_head": fl["flow"], "lookup_table": {k: fl["lookup_table"][k] for k in ("n_bins", "tokenizer")},
            "insert_bos_before_voice": fl.get("insert_bos_before_voice"),
            "append_terminal_punctuation": cfg.get("append_terminal_punctuation", True),
            "replace_characters": cfg.get("replace_characters", {}),
            "mimi": {k: cfg["mimi"][k] for k in ("sample_rate", "frame_rate", "inner_dim", "outer_dim")},
        }
        repo, fn, rev = spec["tokenizer"]
        entry["tokenizer"] = analyze_tokenizer(hf_hub_download(repo, fn, revision=rev, token=token))
        repo, fn, rev = spec["weights"]
        entry["weights"] = safetensors_layout(repo, fn, rev, token)
        report[name] = entry
    if args.ours:
        report["ours (data_pipeline/train_tokenizer.py)"] = {"tokenizer": analyze_tokenizer(args.ours)}
    write_json(args.out, report)
    for name, e in report.items():
        t = e["tokenizer"]
        probes = ", ".join(f"{k}: {v['n_tokens']} tok / {v['n_byte_or_unk']} fallback" for k, v in t["probes"].items())
        logger.info(f"{name}: {t['composition']} | punct {t['standalone_punctuation']} | {probes}")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
