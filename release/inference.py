"""Pocket TTS Indian Hinglish: text-to-speech for Devanagari, Roman Hinglish and mixed-script input.

One command (the repo is private: export a token with read access first):

    pip install pocket-tts huggingface_hub soundfile cmudict wordfreq
    export HF_TOKEN=hf_...
    python inference.py --text "Kal subah 7 baje meri meeting hai, please reminder set kar do" --out out.wav
    python inference.py --variant 12l --quantize --text "..."   # 12-layer CPU student, int8: for laptops

From Python:

    from inference import HinglishTTS
    tts = HinglishTTS.from_pretrained()                      # MSRIT-NBM/pocket-tts-indian-hinglish
    audio = tts.synthesize("आज ट्रैफ़िक बहुत ज़्यादा है, मैं दस मिनट लेट हो जाऊँगा", voice="indian_male_aarav")
    tts.save(audio, "out.wav")
    audio = tts.synthesize("मेरी meeting cancel हो गई", voice="my_voice.wav")   # zero-shot voice cloning

Text front-end (always): danda (। ॥) -> period, and a period is appended when the text does not end
in . ? or !, because the model was trained on terminally punctuated sentences. With
auto_normalize=True (default) the bundled HinglishTextProcessor (hinglish_text/, needs cmudict and
wordfreq) also verbalises numbers ("7 baje" -> "saat baje", "₹2,450" -> "do hazaar chaar sau
pachaas rupaye"), normalises punctuation and quotes, and renders the text in one script:
"devanagari" (default: Hindi as written, English words spelled phonetically in Devanagari via
CMUdict, which gave the most intelligible speech on the benchmark), "roman", "code_switched", or
"auto" to keep the input's script (all-Latin -> Roman Hinglish, all-Devanagari -> Devanagari,
mixed -> code-switched). The model itself reads all three scripts; auto_normalize=False sends the
text as typed. Without cmudict / wordfreq it falls back to the punctuation-only front-end.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
# variant -> folder of its config.yaml / model.safetensors / voices/ (tokenizer and normaliser are shared)
VARIANTS = {"24l": "", "12l": "student_12l"}
DEFAULT_VOICE = "indian_female_ananya"
SCRIPTS = ("devanagari", "auto", "roman", "code_switched")
DEFAULT_SCRIPT = "devanagari"
_DANDA = str.maketrans({"।": ".", "॥": "."})
_TERMINAL = ".?!"
_WEAK = ",;:-–—"
_CLOSERS = "\"'”’)]»"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

logger = logging.getLogger("pocket_tts_hinglish")


def ensure_terminal_punctuation(text: str) -> str:
    """Danda -> period; append a period unless the text ends in . ? or ! (closing quotes and
    brackets allowed after it). A trailing comma, colon or dash becomes the period."""
    t = " ".join(text.translate(_DANDA).split())
    if not t:
        raise ValueError("text is empty")
    core = t.rstrip(_CLOSERS + " ")
    closers = t[len(core):].strip()
    if not core or core[-1] in _TERMINAL:
        return t
    if core[-1] in _WEAK:
        return core.rstrip(_WEAK + " ") + "." + closers
    return t + "."


def detect_script(text: str) -> str:
    has_deva, has_latin = bool(_DEVANAGARI.search(text)), bool(_LATIN.search(text))
    if has_deva and has_latin:
        return "code_switched"
    return "devanagari" if has_deva else "roman"


def load_text_processor(model_dir: Path | None = None) -> Any | None:
    """HinglishTextProcessor from the hinglish_text/ package bundled with the model files (or next
    to this script, or this project's data_pipeline/), or None when it or its dependencies
    (cmudict, wordfreq) are missing."""
    for d in (Path(__file__).resolve().parent, model_dir):
        if d is not None and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    for module in ("hinglish_text.hinglish_text_processor", "data_pipeline.hinglish_text_processor"):
        try:
            mod = __import__(module, fromlist=["HinglishTextProcessor"])
            return mod.HinglishTextProcessor()
        except ImportError as exc:
            last = exc
    logger.warning(f"auto_normalize unavailable ({last}); using the punctuation-only front-end")
    return None


class HinglishTTS:
    default_voice = DEFAULT_VOICE

    def __init__(self, model: Any, voices_dir: Path | None = None, auto_normalize: bool = True,
                 script: str = DEFAULT_SCRIPT, model_dir: Path | None = None):
        if script not in SCRIPTS:
            raise ValueError(f"script must be one of {SCRIPTS}")
        self.model = model
        self.voices_dir = voices_dir
        self.model_dir = model_dir
        self.script = script
        self.processor = load_text_processor(model_dir) if auto_normalize else None
        self._states: dict[str, Any] = {}
        self._quantized = False

    # ----- loading ---------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_or_dir: str = REPO_ID, token: str | None = None, revision: str | None = None,
                        device: str = "cpu", quantize: bool = False, auto_normalize: bool = True,
                        script: str = DEFAULT_SCRIPT, variant: str = "24l") -> HinglishTTS:
        """Download (or reuse) the repo and load the model. The repo's config.yaml points at
        hf:// files; a local copy pointing at the downloaded files is used instead, so loading needs
        the token only once and works offline afterwards. variant="12l" loads the 12-layer distilled
        student (student_12l/): same voices and front-end, half the backbone compute per audio frame."""
        from pocket_tts import TTSModel

        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
        sub = VARIANTS[variant]
        local = Path(repo_or_dir)
        if not local.is_dir():
            from huggingface_hub import snapshot_download

            files = ["config.yaml", "model.safetensors", "voices/*"]
            local = Path(snapshot_download(repo_or_dir, token=token, revision=revision, allow_patterns=[
                *(f"{sub}/{f}" if sub else f for f in files), "tokenizer.model", "hinglish_text/*"]))
        model_dir = local / sub if sub else local
        model = TTSModel.load_model(config=str(cls._local_config(model_dir, local / "tokenizer.model")),
                                    quantize=quantize)
        if device != "cpu":
            if quantize:
                raise ValueError("quantize=True (dynamic int8) runs on CPU only")
            model.to(device)
        tts = cls(model, model_dir / "voices", auto_normalize=auto_normalize, script=script, model_dir=local)
        tts._quantized = quantize
        return tts

    @staticmethod
    def _local_config(local: Path, tokenizer: Path | None = None) -> Path:
        import yaml

        cfg = yaml.safe_load((local / "config.yaml").read_text(encoding="utf-8"))
        cfg["weights_path"] = str(local / "model.safetensors")
        cfg.pop("weights_path_without_voice_cloning", None)
        cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer or local / "tokenizer.model")
        key = hashlib.sha256(str(local.resolve()).encode()).hexdigest()[:12]
        path = Path(tempfile.gettempdir()) / f"pocket_tts_hinglish_{key}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    # ----- voices ----------------------------------------------------------------------------------
    def voices(self) -> list[str]:
        if self.voices_dir is None or not self.voices_dir.is_dir():
            return []
        return sorted({p.stem for p in self.voices_dir.iterdir() if p.suffix in (".safetensors", ".wav")})

    def voice_state(self, voice: str) -> Any:
        """A preset voice name (voices/<name>), or a path to any audio file (zero-shot cloning) or
        to an exported voice state (.safetensors). States are cached per voice."""
        if voice not in self._states:
            preset = self.voices_dir / voice if self.voices_dir is not None else None
            if preset is not None and (preset.with_suffix(".safetensors").exists() or preset.with_suffix(".wav").exists()):
                # precomputed states hold the fp32 model's cache; re-encode the prompt for int8
                use_state = preset.with_suffix(".safetensors").exists() and not self._quantized
                source = preset.with_suffix(".safetensors" if use_state else ".wav")
            else:
                source = Path(voice)
                if not source.exists():
                    raise FileNotFoundError(f"unknown voice {voice!r}: not a preset ({self.voices()}) nor a file")
            self._states[voice] = self.model.get_state_for_audio_prompt(source)
        return self._states[voice]

    # ----- text ------------------------------------------------------------------------------------
    def normalize(self, text: str, auto_normalize: bool | None = None, script: str | None = None) -> str:
        script = script or self.script
        if auto_normalize is False or self.processor is None:
            return ensure_terminal_punctuation(text)
        text = self._romanize_minority_devanagari(text)
        target = detect_script(text) if script == "auto" else script
        return ensure_terminal_punctuation(self.processor.normalize_for_tts(text, script=target))

    def _romanize_minority_devanagari(self, text: str) -> str:
        """Roman Hinglish typed with a few Devanagari words ("Kal 7 baje meeting hai, सेट कर दो"). The
        normaliser reads every Latin word of mixed-script text as English, which is right for Devanagari
        text with English words but wrong here ("Kal" -> कैल, "7" -> seven): when Latin words are the
        majority, romanise the Devanagari ones so it decides Hindi vs English per word."""
        words = text.translate(_DANDA).split()
        n_deva = sum(bool(_DEVANAGARI.search(w)) for w in words)
        n_latin = sum(bool(_LATIN.search(w)) for w in words)
        if not n_deva or n_latin <= n_deva:
            return text
        romanize = self.processor.romanize_hindi_word
        return " ".join(re.sub(r"[\u0900-\u0963\u0966-\u097F]+", lambda m: romanize(m.group(0)), w) for w in words)

    # ----- synthesis -------------------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE, auto_normalize: bool | None = None,
                   script: str | None = None) -> np.ndarray:
        """Float32 mono audio at 24 kHz."""
        prepared = self.normalize(text, auto_normalize, script)
        audio = self.model.generate_audio(self.voice_state(voice), prepared)
        return audio.detach().float().cpu().numpy()

    def stream(self, text: str, voice: str = DEFAULT_VOICE, auto_normalize: bool | None = None,
               script: str | None = None) -> Iterator[np.ndarray]:
        """Audio chunks as soon as they are decoded (first chunk after ~one frame)."""
        prepared = self.normalize(text, auto_normalize, script)
        for chunk in self.model.generate_audio_stream(self.voice_state(voice), prepared):
            yield chunk.detach().float().cpu().numpy()

    def save(self, audio: np.ndarray, path: str | Path) -> None:
        import soundfile as sf

        sf.write(str(path), audio, self.sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hinglish text-to-speech with Pocket TTS")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="preset name or a .wav to clone")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--repo", default=REPO_ID, help="HF repo id or a local directory with the model files")
    ap.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN / cached login)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quantize", action="store_true", help="dynamic int8 on CPU (faster, smaller)")
    ap.add_argument("--no-normalize", action="store_true", help="skip number/script normalisation")
    ap.add_argument("--script", default=DEFAULT_SCRIPT, choices=SCRIPTS,
                    help="script the normaliser renders into (auto = keep the input's)")
    ap.add_argument("--variant", default="24l", choices=tuple(VARIANTS),
                    help="24l = the fine-tuned model, 12l = its distilled 12-layer student (faster on CPU)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tts = HinglishTTS.from_pretrained(args.repo, token=args.token, device=args.device, quantize=args.quantize,
                                      auto_normalize=not args.no_normalize, script=args.script, variant=args.variant)
    prepared = tts.normalize(args.text)
    audio = tts.synthesize(args.text, voice=args.voice)
    tts.save(audio, args.out)
    print(f"model input: {prepared}\nwrote {args.out} ({len(audio) / tts.sample_rate:.2f} s at {tts.sample_rate} Hz)")


if __name__ == "__main__":
    main()
