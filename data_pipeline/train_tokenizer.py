"""Train the bilingual Devanagari + Latin 4,000-piece SentencePiece tokenizer.

Corpus: the spoken Devanagari, Roman and code-switched renderings of every transcript that passed
the text filters in prepare_dataset (candidates.jsonl), excluding the validation utterances. Using
all three renderings is what lets one table embed Hindi in either script and English words in
either script.

Spec: the one the released Pocket TTS tokenizers use (pocket-tts training/scripts/train_tokenizer.py:
unigram, 4000 pieces = lookup_table.n_bins, identity normalisation, no whitespace squashing, pieces
<= 6 characters, split digits, byte fallback, pad_id 3), plus two deliberate changes:
  * character_coverage 1.0: the cleaned alphabet is small and every character is legitimate;
  * . , ? ! ; : as user-defined symbols, so they are always standalone pieces -- pocket-tts splits
    long inputs at ".", "!", "?" tokens and sub-splits at ",", ";", ":" tokens (text_chunking.py).

    python -m data_pipeline.train_tokenizer [--out tokenizer/hinglish_bilingual_4k]
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import tempfile
from pathlib import Path

import sentencepiece as spm

from data_pipeline.common import DATA_ROOT, REPO_ROOT, read_jsonl, setup_logging, write_json

logger = setup_logging("train_tokenizer")

PUNCT_SYMBOLS = [".", ",", "?", "!", ";", ":"]


def build_corpus(candidates: Path, selected: Path) -> tuple[list[str], dict]:
    val_ids = {r["utt_id"] for r in read_jsonl(selected) if r["split"] == "validation"}
    val_texts = {r["spoken_roman"] for r in read_jsonl(selected) if r["split"] == "validation"}
    lines: list[str] = []
    per_source = collections.Counter()
    seen: set[str] = set()
    for r in read_jsonl(candidates):
        if "spoken_devanagari" not in r or r["utt_id"] in val_ids or r.get("spoken_roman") in val_texts:
            continue
        for key in ("spoken_devanagari", "spoken_roman", "spoken_code_switched"):
            line = r[key]
            if line not in seen:
                seen.add(line)
                lines.append(line)
                per_source[r["source_key"]] += 1
    return lines, {"lines": len(lines), "by_source": dict(per_source), "validation_excluded": len(val_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=DATA_ROOT / "processed" / "candidates.jsonl")
    ap.add_argument("--selected", type=Path, default=DATA_ROOT / "processed" / "selected.jsonl")
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "tokenizer" / "hinglish_bilingual_4k"))
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    lines, corpus_stats = build_corpus(args.candidates, args.selected)
    random.Random(args.seed).shuffle(lines)
    logger.info(f"corpus: {corpus_stats}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Fixed corpus path: sentencepiece records trainer_spec.input inside the .model file, so a random
    # temp name would make every run's .model differ byte-wise even with identical pieces/scores.
    corpus = os.path.join(tempfile.gettempdir(), "hinglish_tokenizer_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if hasattr(spm, "set_random_generator_seed"):
        spm.set_random_generator_seed(args.seed)
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=args.out,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentencepiece_length=6,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        byte_fallback=True,
        pad_id=3,
        user_defined_symbols=PUNCT_SYMBOLS,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        num_threads=16,
        minloglevel=2,
    )
    os.remove(corpus)
    sp = spm.SentencePieceProcessor(model_file=args.out + ".model")
    assert sp.get_piece_size() == args.vocab_size, sp.get_piece_size()

    # coverage report on held-out (validation) transcripts in all three scripts
    val = [r for r in read_jsonl(args.selected) if r["split"] == "validation"]
    report: dict = {"vocab_size": sp.get_piece_size(), "corpus": corpus_stats, "spec": {
        "model_type": "unigram", "character_coverage": 1.0, "normalization": "identity",
        "max_sentencepiece_length": 6, "split_digits": True, "byte_fallback": True, "pad_id": 3,
        "user_defined_symbols": PUNCT_SYMBOLS}}
    for key in ("spoken_devanagari", "spoken_roman", "spoken_code_switched"):
        n_tok = n_unk = n_words = 0
        for r in val:
            ids = sp.encode(r[key], out_type=int)
            n_tok += len(ids)
            n_unk += sum(sp.is_byte(i) or i == sp.unk_id() for i in ids)
            n_words += len(r[key].split())
        report[f"validation_{key}"] = {"tokens_per_word": round(n_tok / max(1, n_words), 3),
                                       "byte_or_unk_fraction": round(n_unk / max(1, n_tok), 5)}
    n = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(n) if not sp.is_byte(i)]  # <0x41> is not "Latin"
    deva = sum(any("ऀ" <= ch <= "ॿ" for ch in p) for p in pieces)
    latin = sum(any(ch.isascii() and ch.isalpha() for ch in p) for p in pieces)
    report["piece_scripts"] = {"devanagari": deva, "latin": latin, "byte": sum(sp.is_byte(i) for i in range(n))}
    report["punctuation_standalone"] = {p: sp.piece_to_id(p) != sp.unk_id() for p in PUNCT_SYMBOLS}
    write_json(Path(args.out + ".stats.json"), report)
    logger.info(f"wrote {args.out}.model / .vocab / .stats.json: {report['piece_scripts']}")
    for key in ("spoken_devanagari", "spoken_roman", "spoken_code_switched"):
        logger.info(f"  {key}: {report['validation_' + key]}")


if __name__ == "__main__":
    main()
