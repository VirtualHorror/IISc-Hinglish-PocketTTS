"""Round-trip ASR scoring of Hinglish speech: Whisper transcribes, and the reference and hypothesis
are compared in one canonical script.

Whisper writes Hindi in Devanagari but often keeps English words in Latin script, while references
come in Devanagari, Roman Hinglish or code-switched form. Both sides therefore go through
HinglishTextProcessor.annotate(...).spoken_devanagari (numbers verbalised, English words in
Devanagari phonetics, Roman Hindi back to Devanagari), then a light orthographic fold so spellings
that sound the same are not errors: nukta dropped, chandrabindu -> anusvara, candra vowels
(ॉ ॅ ऑ ऍ) -> plain, a nasal consonant + virama before a consonant -> anusvara (हिन्दी = हिंदी),
zero-width characters and punctuation removed. WER is over words; CER over characters with spaces
removed (Hindi word segmentation varies: आपका / आप का). Corpus-level rates pool the edits of all
utterances.

    asr = WhisperASR()                       # openai/whisper-large-v3-turbo, fp16 on CUDA
    hyps = asr.transcribe([wav_or_path, ...])
    scorer = HinglishScorer()
    scorer.score(refs, hyps)  -> {"cer": ..., "wer": ..., "per_utt": [...]}
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
ASR_SAMPLE_RATE = 16_000

_FOLD = str.maketrans({"़": "", "ँ": "ं", "ॉ": "ो", "ॅ": "े", "ऑ": "ओ", "ऍ": "ए",
                       "​": "", "‌": "", "‍": "", "﻿": ""})
_NASAL_VIRAMA = re.compile(r"[ङञणनम]्(?=[क-ह])")


def fold_devanagari(text: str) -> str:
    t = unicodedata.normalize("NFD", text).translate(_FOLD)  # NFD splits precomposed nukta letters
    t = unicodedata.normalize("NFC", t)
    t = _NASAL_VIRAMA.sub("ं", t)
    # keep letters, combining marks (matras, virama, anusvara: re's \w would drop them) and digits
    t = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in t)
    return " ".join(t.lower().split())


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


class HinglishScorer:
    """CER / WER in canonical (folded, spoken) Devanagari."""

    def __init__(self, processor: Any | None = None):
        if processor is None:
            from data_pipeline.hinglish_text_processor import HinglishTextProcessor

            processor = HinglishTextProcessor()
        self.processor = processor

    def canonical(self, text: str) -> str:
        try:
            spoken = self.processor.annotate(text).spoken_devanagari
        except ValueError:  # no words at all (e.g. an empty transcript)
            spoken = ""
        return fold_devanagari(spoken)

    def score(self, refs: Sequence[str], hyps: Sequence[str]) -> dict[str, Any]:
        per_utt = []
        totals = {"char_edits": 0, "chars": 0, "word_edits": 0, "words": 0}
        for ref, hyp in zip(refs, hyps, strict=True):
            r, h = self.canonical(ref), self.canonical(hyp)
            rc, hc = r.replace(" ", ""), h.replace(" ", "")
            ce, we = edit_distance(rc, hc), edit_distance(r.split(), h.split())
            totals["char_edits"] += ce
            totals["chars"] += len(rc)
            totals["word_edits"] += we
            totals["words"] += len(r.split())
            per_utt.append({"ref": r, "hyp": h, "cer": ce / max(1, len(rc)), "wer": we / max(1, len(r.split()))})
        return {
            "cer": totals["char_edits"] / max(1, totals["chars"]),
            "wer": totals["word_edits"] / max(1, totals["words"]),
            "n": len(per_utt),
            **totals,
            "per_utt": per_utt,
        }


def load_audio_16k(audio: np.ndarray | str | Path, sample_rate: int | None = None) -> np.ndarray:
    import soundfile as sf
    import soxr

    if not isinstance(audio, np.ndarray):
        audio, sample_rate = sf.read(str(audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate is None:
        raise ValueError("sample_rate is required for raw arrays")
    if sample_rate != ASR_SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, ASR_SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


class WhisperASR:
    def __init__(self, model_id: str = WHISPER_MODEL, revision: str | None = WHISPER_REVISION,
                 device: str = "cuda", dtype: Any = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=self.dtype).to(self.device).eval()
        self.model_id = model_id

    def transcribe(self, audios: Sequence[np.ndarray | str | Path], sample_rates: Sequence[int | None] | None = None,
                   language: str | Sequence[str] = "hi", batch_size: int = 16) -> list[str]:
        """Each audio is a path or a mono array (then give its sample rate); <= 30 s each.
        `language` is one Whisper language for all, or one per audio: "hi" for Hindi-matrix speech
        (Hinglish included), "en" for English-matrix speech. Forcing "hi" on English speech makes
        Whisper translate or transliterate instead of transcribing."""
        torch = self.torch
        rates = list(sample_rates) if sample_rates is not None else [None] * len(audios)
        langs = [language] * len(audios) if isinstance(language, str) else list(language)
        if len(langs) != len(audios):
            raise ValueError(f"{len(langs)} languages for {len(audios)} audios")
        waves = [load_audio_16k(a, sr) for a, sr in zip(audios, rates, strict=True)]
        out: list[str] = [""] * len(waves)
        for lang in sorted(set(langs)):
            idx = [i for i, x in enumerate(langs) if x == lang]
            for b in range(0, len(idx), batch_size):
                chunk = idx[b:b + batch_size]
                feats = self.processor.feature_extractor([waves[i] for i in chunk], sampling_rate=ASR_SAMPLE_RATE,
                                                         return_tensors="pt")
                with torch.inference_mode():
                    ids = self.model.generate(feats.input_features.to(self.device, self.dtype),
                                              language=lang, task="transcribe")
                for i, text in zip(chunk, self.processor.batch_decode(ids, skip_special_tokens=True), strict=True):
                    out[i] = text.strip()
        return out


def asr_language(language: str) -> str:
    """Whisper language for a dataset language tag ("hi", "en", "hi-en")."""
    return "en" if language == "en" else "hi"
