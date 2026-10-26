#!/usr/bin/env python3
"""Hinglish Pocket TTS studio: an interactive demo of the Indian Hinglish Pocket TTS model.

    pip install fastapi uvicorn            # on top of the model's requirements.txt
    python demo_app.py                     # then open http://127.0.0.1:7860

For any text in Roman Hinglish, Devanagari or mixed script the studio shows
  - the front-end: "phonetic" (default: numbers verbalised, English words spelled phonetically,
    everything rendered in Devanagari; 7.2 % CER on the benchmark) or "raw" (the text as typed, only
    terminal punctuation added, straight into the 4,000-piece multi-script tokenizer)
  - the SentencePiece pieces the model actually reads, chunk by chunk
  - the audio, streamed while it is generated, with the time to first audio (TTFA: stream call to the
    first decoded chunk, text normalisation included, voice state cached; the benchmark's definition)
    and the real-time factor (RTF = synthesis time / audio duration) measured on this machine
  - waveform and mel-spectrogram (drawn in the browser while the audio streams)
  - the studio voices, or zero-shot cloning from an uploaded recording

Speed: on CPU the model runs with dynamic int8 weights (about 2x faster than fp32) and one torch thread
per two physical cores (up to 4; pocket-tts itself pins 1). When the 12-layer distilled student is present
(<model>/student_12l/) it is offered next to the 24-layer model (half the backbone compute per frame), and
it is preselected when the 24-layer model's start-up warm-up is slower than real time on this machine. The preset prompts are also served from
demo/cache/ (takes rendered ahead of time by scripts/build_demo_cache.py, each with the TTFA / RTF
measured when it was rendered), so they play instantly on any machine; other text is synthesised live,
and a live take played again is replayed from memory. Every number on screen says where it was measured.

Model files: --model DIR (an exported release folder: config.yaml, model.safetensors, voices/, ...),
else this script's own folder when it sits in the model repo, else ./artifacts/release, else the
Hugging Face repo (private: export HF_TOKEN=<read token> first).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import platform
import queue
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # the web layer (main() says how to install it when missing)
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except ImportError:
    uvicorn = None

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demo"
REPO_ID = "MSRIT-NBM/pocket-tts-indian-hinglish"
MODES = ("phonetic", "raw")
MAX_TEXT_CHARS = 600
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CLONE_MAX_SECONDS = 15.0
CLONE_MIN_SECONDS = 1.0
CACHE_CHUNK_SECONDS = 0.5  # cached takes are sent in chunks of this length
REPLAY_SLOTS = 32
WARMUP_TEXT = "नमस्ते, आप कैसे हैं?"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_BYTE_PIECE = re.compile(r"<0x[0-9A-Fa-f]{2}>")

logger = logging.getLogger("hinglish_studio")


# ----- model files ------------------------------------------------------------------------------------
def resolve_model_dir(arg: str | None, token: str | None) -> Path:
    if arg:
        if Path(arg).is_dir():
            return Path(arg).resolve()
        repo = arg
    else:
        for d in (HERE, HERE / "artifacts" / "release"):
            if (d / "config.yaml").exists() and (d / "model.safetensors").exists():
                return d
        repo = REPO_ID
    from huggingface_hub import snapshot_download

    logger.info(f"downloading {repo} (needs a token with read access: HF_TOKEN)")
    return Path(snapshot_download(repo, token=token, allow_patterns=[
        "config.yaml", "model.safetensors", "tokenizer.model", "inference.py", "voices/*", "hinglish_text/*",
        "student_*/*", "student_*/voices/*", "evaluation_results.json", "demo/cache/*"]))


def import_inference(model_dir: Path) -> Any:
    """The shipped helper (inference.py: HinglishTTS): this repo's release/ copy, else the model folder's."""
    for d in (HERE / "release", model_dir, HERE):
        if (d / "inference.py").exists():
            sys.path.insert(0, str(d))
            import inference

            return inference
    raise SystemExit(f"inference.py not found next to {model_dir}")


def default_threads() -> int:
    """pocket-tts runs latent generation and Mimi decoding in two threads at once, so each gets half the
    physical cores (logical CPUs / 2 with SMT) as torch intra-op threads, between 1 and 4. On a 2-core laptop
    1 thread is fastest (scripts/cpu_speed_probe.py: RTF 0.71 vs 0.88 with 2 for the 12-layer student)."""
    logical = os.cpu_count() or 2
    physical = logical // 2 if logical >= 4 else logical
    return max(1, min(4, physical // 2))


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return " ".join(line.split(":", 1)[1].split())
    except OSError:
        pass
    return platform.processor() or platform.machine()


# ----- engines ----------------------------------------------------------------------------------------
@dataclass
class Job:
    """One piece of model work, run on its engine's worker thread; `events` carries its stream lines
    (None closes it)."""

    run: Callable[[Job], None]
    stop: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=queue.Queue)
    ended: bool = False

    def emit(self, kind: str, **payload: Any) -> None:
        self.ended = self.ended or kind in ("done", "error")
        self.events.put(event(kind, **payload))


@dataclass
class Engine:
    """One loaded model variant. TTSModel is not thread-safe, so a single worker thread owns the model and
    runs jobs one at a time; request threads only read a job's events. The newest request stops the job in
    flight and skips queued ones (pocket-tts's stop event), so a client that went away never blocks the next."""

    key: str
    label: str
    layers: int
    tts: Any
    device: str
    precision: str
    params_m: float
    jobs: queue.Queue = field(default_factory=queue.Queue)
    current: Job | None = None
    rtf_ema: float | None = None
    chars_per_s: float = 9.8  # model-input characters per second of audio (benchmark mean), refined live

    @property
    def model(self) -> Any:
        return self.tts.model

    def start(self) -> None:
        threading.Thread(target=self._work, name=f"engine-{self.key}", daemon=True).start()

    def _work(self) -> None:
        while True:
            job = self.jobs.get()
            if not job.stop.is_set():
                self.current = job
                try:
                    job.run(job)
                except Exception as exc:  # report it to the page and keep the worker alive
                    logger.exception("job failed")
                    job.emit("error", message=f"Synthesis failed: {exc}")
                finally:
                    self.current = None
            job.events.put(None)

    def submit(self, run: Callable[[Job], None], preempt: bool = True) -> Job:
        job = Job(run)
        if preempt:  # the newest request wins
            current = self.current
            if current is not None:
                current.stop.set()
            with self.jobs.mutex:
                for queued in self.jobs.queue:
                    queued.stop.set()
        self.jobs.put(job)
        return job

    def observe(self, rtf: float, chars: int, audio_s: float) -> None:
        self.rtf_ema = rtf if self.rtf_ema is None else 0.6 * self.rtf_ema + 0.4 * rtf
        if audio_s > 0.5:
            self.chars_per_s = 0.8 * self.chars_per_s + 0.2 * (chars / audio_s)


def count_params(weights: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(weights), "pt") as f:
        return sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())


def load_engines(model_dir: Path, inference: Any, device: str, int8: bool, variants: list[str]) -> list[Engine]:
    engines = []
    for key in variants:
        sub = model_dir / inference.VARIANTS[key] if inference.VARIANTS[key] else model_dir
        if not (sub / "model.safetensors").exists():
            logger.info(f"variant {key}: no model files in {sub}, skipped")
            continue
        t = time.perf_counter()
        tts = inference.HinglishTTS.from_pretrained(str(model_dir), device=device, quantize=int8, variant=key)
        layers = len(tts.model.flow_lm.transformer.layers)
        params = count_params(sub / "model.safetensors") / 1e6  # int8 packing hides weights from .parameters()
        label = f"{layers}-layer" + (" student" if key != "24l" else "")
        engines.append(Engine(key, label, layers, tts, device, "int8" if int8 else "fp32", round(params, 1)))
        logger.info(f"loaded {label} ({params:.0f}M params, {'int8' if int8 else 'fp32'}, {device}) "
                    f"in {time.perf_counter() - t:.1f} s")
    if not engines:
        raise SystemExit(f"no model found in {model_dir}")
    return engines


# ----- text analysis ----------------------------------------------------------------------------------
def classify_piece(piece: str) -> str:
    if _BYTE_PIECE.fullmatch(piece):
        return "byte"
    body = piece.replace("▁", "")
    if not body:
        return "space"
    if _DEVANAGARI.search(body):
        return "deva"
    if _LATIN.search(body):
        return "latin"
    return "digit" if body.isdigit() else "punct"


def input_script(text: str) -> str:
    deva, latin = bool(_DEVANAGARI.search(text)), bool(_LATIN.search(text))
    return "mixed" if deva and latin else "devanagari" if deva else "roman" if latin else "none"


def analyse(engine: Engine, text: str, mode: str) -> dict[str, Any]:
    """Front-end output and the pieces of every chunk pocket-tts will generate, exactly as
    TTSModel.generate_audio_stream prepares them."""
    from pocket_tts.default_parameters import MAX_TOKEN_PER_CHUNK
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences

    t = time.perf_counter()
    model_input = engine.tts.normalize(text, auto_normalize=mode == "phonetic")
    normalize_ms = (time.perf_counter() - t) * 1000
    m = engine.model
    tok = m.flow_lm.conditioner.tokenizer
    opts = (m.pad_with_spaces_for_short_inputs, m.remove_semicolons, m.append_terminal_punctuation,
            m.capitalize_first_letter, m.replace_characters)
    chunks = []
    for chunk in split_into_best_sentences(tok, model_input, MAX_TOKEN_PER_CHUNK, opts[0], remove_semicolons=opts[1],
                                           append_terminal_punctuation=opts[2], capitalize_first_letter=opts[3],
                                           replace_characters=opts[4]):
        prepared, _ = prepare_text_prompt(chunk, *opts)
        ids = tok.encode(prepared)
        pieces = [tok.sp.id_to_piece(i) if hasattr(tok, "sp") else tok.tokenizer.id_to_token(i) for i in ids]
        chunks.append({"text": prepared, "pieces": [{"p": p, "id": i, "k": classify_piece(p)}
                                                     for p, i in zip(pieces, ids, strict=True)]})
    kinds = [p["k"] for c in chunks for p in c["pieces"]]
    return {"text": text, "mode": mode, "input_script": input_script(text), "model_input": model_input,
            "model_input_script": input_script(model_input), "normalize_ms": round(normalize_ms, 1),
            "chunks": chunks, "n_tokens": len(kinds), "vocab_size": tok.sp.vocab_size() if hasattr(tok, "sp") else None,
            "piece_counts": {k: kinds.count(k) for k in ("deva", "latin", "digit", "punct", "space", "byte")}}


# ----- pre-rendered takes and replays --------------------------------------------------------------------
def take_key(engine: str, voice: str, mode: str, text: str) -> str:
    return hashlib.sha1(f"{engine}|{voice}|{mode}|{' '.join(text.split())}".encode()).hexdigest()[:16]


class TakeCache:
    """demo/cache/index.json + one WAV per take (scripts/build_demo_cache.py)."""

    def __init__(self, cache_dir: Path | None):
        self.dir = cache_dir
        self.meta: dict[str, Any] = {}
        self.takes: dict[str, dict[str, Any]] = {}
        index = cache_dir / "index.json" if cache_dir else None
        if index is not None and index.exists():
            data = json.loads(index.read_text(encoding="utf-8"))
            self.meta = {k: v for k, v in data.items() if k != "takes"}
            self.takes = {t["key"]: t for t in data["takes"] if (cache_dir / t["file"]).exists()}
        logger.info(f"pre-rendered takes: {len(self.takes)}" + (f" ({cache_dir})" if cache_dir else ""))

    def get(self, key: str) -> tuple[dict[str, Any], np.ndarray] | None:
        take = self.takes.get(key)
        if take is None:
            return None
        import soundfile as sf

        audio, sr = sf.read(str(self.dir / take["file"]), dtype="float32")
        if sr != 24000:
            return None
        return take, audio


class ReplayMemory:
    """The last live takes, by key: playing the same text, voice and mode again is a replay."""

    def __init__(self, slots: int):
        self.slots = slots
        self.items: OrderedDict[str, tuple[dict[str, Any], np.ndarray]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> tuple[dict[str, Any], np.ndarray] | None:
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key: str, meta: dict[str, Any], audio: np.ndarray) -> None:
        with self.lock:
            self.items[key] = (meta, audio)
            self.items.move_to_end(key)
            while len(self.items) > self.slots:
                self.items.popitem(last=False)


# ----- voices -----------------------------------------------------------------------------------------
class Voices:
    """Studio voices (per engine) and uploaded recordings for zero-shot cloning (shared by engines)."""

    def __init__(self, model_dir: Path, workdir: Path):
        self.workdir = workdir
        self.uploads: dict[str, dict[str, Any]] = {}
        index = model_dir / "voices" / "index.json"
        self.presets = {v["name"]: v for v in json.loads(index.read_text(encoding="utf-8"))} if index.exists() else {}

    def resolve(self, voice: str) -> str:
        """The argument HinglishTTS.voice_state takes: a preset name or an uploaded file's path."""
        if voice in self.uploads:
            return self.uploads[voice]["path"]
        if voice in self.presets:
            return voice
        raise KeyError(f"unknown voice {voice!r}")

    def add_upload(self, data: bytes, filename: str) -> dict[str, Any]:
        import io

        import soundfile as sf

        try:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception as exc:  # soundfile raises its own LibsndfileError / RuntimeError types
            raise ValueError("Could not read that file as audio. Upload a WAV, FLAC, OGG or MP3 recording.") from exc
        audio = audio.mean(axis=1)
        duration = len(audio) / sr
        if duration < CLONE_MIN_SECONDS:
            raise ValueError(f"The recording is {duration:.1f} s long. Upload at least {CLONE_MIN_SECONDS:.0f} s of speech.")
        audio = audio[: int(CLONE_MAX_SECONDS * sr)]
        peak = float(np.abs(audio).max())
        if peak < 1e-4:
            raise ValueError("The recording is silent. Upload a clip with speech in it.")
        digest = hashlib.sha1(data).hexdigest()[:10]
        voice_id = f"upload_{digest}"
        path = self.workdir / f"{voice_id}.wav"
        sf.write(str(path), audio, sr)
        stem = Path(filename or "recording").stem[:40] or "recording"
        self.uploads[voice_id] = {"id": voice_id, "label": stem, "path": str(path),
                                  "duration_s": round(min(duration, CLONE_MAX_SECONDS), 2),
                                  "trimmed": duration > CLONE_MAX_SECONDS, "sample_rate": sr}
        return {k: v for k, v in self.uploads[voice_id].items() if k != "path"}


# ----- the studio -------------------------------------------------------------------------------------
def event(kind: str, **payload: Any) -> bytes:
    return (json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n").encode("utf-8")


def pcm16_b64(audio: np.ndarray) -> str:
    return base64.b64encode((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()).decode("ascii")


class Studio:
    def __init__(self, engines: list[Engine], voices: Voices, cache: TakeCache, presets: dict[str, Any],
                 benchmark: dict[str, Any], hardware: dict[str, Any]):
        self.engines = {e.key: e for e in engines}
        self.voices = voices
        self.cache = cache
        self.presets = presets
        self.benchmark = benchmark
        self.hardware = hardware
        self.replays = ReplayMemory(REPLAY_SLOTS)

    def engine(self, key: str | None) -> Engine:
        return self.engines.get(key or "", next(iter(self.engines.values())))

    def warm_up(self) -> None:
        """One short synthesis per engine (kernels, allocator) with the default voice; seeds the RTF
        estimate the browser uses to size its playback buffer."""
        for e in self.engines.values():  # before the workers start: the model is free
            state = e.tts.voice_state(e.tts.default_voice)
            for _ in range(2):  # the first pass pays one-time costs; the second is timed
                t = time.perf_counter()
                text = e.tts.normalize(WARMUP_TEXT)
                n = sum(len(c) for c in e.model.generate_audio_stream(state, text))
                seconds = time.perf_counter() - t
            e.observe(seconds / max(1e-6, n / 24000), len(text), n / 24000)
            logger.info(f"warm-up {e.label}: {n / 24000:.2f} s of audio in {seconds:.2f} s (RTF {e.rtf_ema:.2f})")

    def start(self) -> None:
        """Start each engine's worker, then queue the other studio voices' states (int8 re-encodes each
        prompt); a request preempts these, and a skipped voice is encoded on first use instead."""
        for e in self.engines.values():
            e.start()
            for name in self.voices.presets:
                e.submit(lambda job, e=e, name=name: e.tts.voice_state(name), preempt=False)

    def info(self) -> dict[str, Any]:
        quality = self.benchmark.get("quality", {}).get("default", {})
        voices = []
        for name, v in self.voices.presets.items():
            q = quality.get(f"voice={name}", {})
            voices.append({"id": name, "label": name.split("_")[-1].capitalize(),
                           "gender": "female" if "_female_" in name else "male", "source": v.get("source", ""),
                           "cer": q.get("cer"), "wer": q.get("wer")})
        voices.sort(key=lambda v: (v["cer"] is None, v["cer"] or 0))
        speed = self.benchmark.get("speed", {})
        return {
            "engines": [{"key": e.key, "label": e.label, "layers": e.layers, "params_m": e.params_m,
                         "device": e.device, "precision": e.precision, "rtf_hint": e.rtf_ema}
                        for e in self.engines.values()],
            "voices": voices, "uploads": [{k: v for k, v in u.items() if k != "path"} for u in self.voices.uploads.values()],
            "presets": self.presets, "hardware": self.hardware,
            "cache": {"takes": len(self.cache.takes), **self.cache.meta},
            "benchmark": {"overall": quality.get("overall"),
                          "by_script": {s: quality.get(f"input_script={s}") for s in ("roman", "devanagari", "code_switched")},
                          "raw_overall": self.benchmark.get("quality", {}).get("raw", {}).get("overall"),
                          "speed": {k: {m: v.get(m) for m in ("rtf_total", "ttfa_ms_p50", "x_real_time")}
                                    for k, v in speed.items()}},
        }

    def synthesize(self, text: str, voice: str, mode: str, engine_key: str | None, source: str) -> Iterator[bytes]:
        engine = self.engine(engine_key)
        try:
            voice_arg = self.voices.resolve(voice)
            info = analyse(engine, text, mode)
        except (KeyError, ValueError) as exc:
            yield event("error", message=str(exc))
            return
        key = take_key(engine.key, voice, mode, text)
        est_audio_s = max(1.0, len(info["model_input"]) / engine.chars_per_s)
        if source != "live":
            hit = self.cache.get(key) if voice in self.voices.presets else None
            if hit is not None:
                yield from self._serve(info, hit[0], hit[1], "cache", engine, key)
                return
            replay = self.replays.get(key)
            if replay is not None:
                yield from self._serve(info, replay[0], replay[1], "replay", engine, key)
                return
        yield event("start", **info, source="live", engine=engine.key, key=key, est_audio_s=round(est_audio_s, 2),
                    rtf_hint=engine.rtf_ema)
        yield from self._live(engine, info, voice, voice_arg, key)

    def _serve(self, info: dict[str, Any], meta: dict[str, Any], audio: np.ndarray, source: str, engine: Engine,
               key: str) -> Iterator[bytes]:
        t = time.perf_counter()
        yield event("start", **info, source=source, engine=engine.key, key=key,
                    est_audio_s=round(len(audio) / 24000, 2), rtf_hint=0.0)
        step = int(CACHE_CHUNK_SECONDS * 24000)
        first_ms = None
        for i in range(0, len(audio), step):
            yield event("audio", pcm=pcm16_b64(audio[i:i + step]))
            if first_ms is None:
                first_ms = (time.perf_counter() - t) * 1000
        yield event("done", source=source, served_ttfa_ms=round(first_ms or 0.0, 2),
                    served_ms=round((time.perf_counter() - t) * 1000, 2), audio_s=round(len(audio) / 24000, 3),
                    take=meta)

    def _live(self, engine: Engine, info: dict[str, Any], voice: str, voice_arg: str, key: str) -> Iterator[bytes]:
        job = engine.submit(lambda job: self._synthesize(job, engine, info, voice, voice_arg, key))
        try:
            while (line := job.events.get()) is not None:
                yield line
            if not job.ended:  # preempted before it started
                yield event("done", source="live", stopped=True, audio_s=0.0)
        finally:
            job.stop.set()  # the stream is over or the client left: stop generating

    def _synthesize(self, job: Job, engine: Engine, info: dict[str, Any], voice: str, voice_arg: str, key: str) -> None:
        """Runs on the engine's worker thread."""
        t = time.perf_counter()
        new_voice = voice_arg not in engine.tts._states
        state = engine.tts.voice_state(voice_arg)
        voice_ms = (time.perf_counter() - t) * 1000
        if new_voice:
            job.emit("status", message=f"Voice prompt encoded in {voice_ms:.0f} ms")
        chunks: list[np.ndarray] = []
        t0 = time.perf_counter()
        # TTFA as in eval/evaluate_hinglish_tts.py: normalisation (already timed) + generation
        ttfa_ms = None
        for chunk in engine.model.generate_audio_stream(state, info["model_input"], stop=job.stop):
            audio = chunk.detach().float().cpu().numpy()
            if ttfa_ms is None:
                ttfa_ms = info["normalize_ms"] + (time.perf_counter() - t0) * 1000
            chunks.append(audio)
            job.emit("audio", pcm=pcm16_b64(audio))
        synth_s = time.perf_counter() - t0 + info["normalize_ms"] / 1000
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio_s = len(audio) / 24000
        if job.stop.is_set() or audio_s == 0:
            job.emit("done", source="live", stopped=True, audio_s=round(audio_s, 3))
            return
        rtf = synth_s / audio_s
        engine.observe(rtf, len(info["model_input"]), audio_s)
        meta = {"engine": engine.key, "voice": voice, "mode": info["mode"], "text": info["text"],
                "model_input": info["model_input"], "audio_s": round(audio_s, 3),
                "ttfa_ms": round(ttfa_ms or 0.0, 1), "rtf": round(rtf, 4), "synth_s": round(synth_s, 3),
                "voice_encode_ms": round(voice_ms, 1) if new_voice else None,
                "device": self.hardware["device_label"], "measured": "live on this machine",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.replays.put(key, meta, audio)
        job.emit("done", source="live", **meta)


def create_app(studio: Studio) -> Any:
    app = FastAPI(title="Hinglish Pocket TTS studio", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DEMO_DIR / "index.html", media_type="text/html")

    @app.get("/api/info")
    def info() -> JSONResponse:
        return JSONResponse(studio.info())

    def _text(body: dict[str, Any]) -> str:
        text = " ".join(str(body.get("text", "")).split())
        if not text:
            raise HTTPException(400, "Type or pick a sentence first.")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(400, f"Keep the text under {MAX_TEXT_CHARS} characters ({len(text)} now).")
        return text

    def _mode(body: dict[str, Any]) -> str:
        mode = body.get("mode", "phonetic")
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        return mode

    @app.post("/api/analyse")
    async def analyse_text(request: Request) -> JSONResponse:
        body = await request.json()
        engine = studio.engine(body.get("engine"))
        try:
            return JSONResponse(analyse(engine, _text(body), _mode(body)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/synthesize")
    async def synthesize(request: Request) -> StreamingResponse:
        body = await request.json()
        source = body.get("source", "auto")
        stream = studio.synthesize(_text(body), str(body.get("voice", "")), _mode(body), body.get("engine"),
                                   "live" if source == "live" else "auto")
        return StreamingResponse(stream, media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/voices")
    async def upload_voice(request: Request) -> JSONResponse:
        data = await request.body()
        if not data:
            raise HTTPException(400, "The upload was empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload a recording under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            return JSONResponse(studio.voices.add_upload(data, request.headers.get("x-filename", "recording")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Hinglish Pocket TTS studio (interactive demo)")
    ap.add_argument("--model", default=None, help="release folder or HF repo id (default: see module docstring)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token for the private repo")
    ap.add_argument("--variants", default="24l,12l", help="model variants to load when present (24l, 12l)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--fp32", action="store_true", help="CPU: keep fp32 weights instead of dynamic int8")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (default: physical cores / 2, 1-4)")
    ap.add_argument("--cache", default=None, help="pre-rendered takes (default: demo/cache here or in the model "
                    "folder; '' disables)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("pocket_tts", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if uvicorn is None:
        raise SystemExit("the studio needs FastAPI and uvicorn: pip install fastapi uvicorn")

    model_dir = resolve_model_dir(args.model, args.token)
    inference = import_inference(model_dir)
    import torch

    threads = args.threads or default_threads()
    torch.set_num_threads(threads)  # after pocket_tts's import-time set_num_threads(1)
    int8 = args.device == "cpu" and not args.fp32
    engines = load_engines(model_dir, inference, args.device, int8,
                           [v for v in args.variants.split(",") if v in inference.VARIANTS])
    device_label = torch.cuda.get_device_name() if args.device.startswith("cuda") else cpu_name()
    hardware = {"device": args.device, "device_label": device_label, "threads": threads,
                "precision": "int8" if int8 else "fp32", "logical_cpus": os.cpu_count(), "torch": torch.__version__}
    bench_path = next((p for p in (HERE / "eval" / "evaluation_results.json", model_dir / "evaluation_results.json")
                       if p.exists()), None)
    benchmark = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path else {}
    presets = json.loads((DEMO_DIR / "presets.json").read_text(encoding="utf-8"))
    if args.cache is None:
        args.cache = next((str(d) for d in (DEMO_DIR / "cache", model_dir / "demo" / "cache")
                           if (d / "index.json").exists()), "")
    with tempfile.TemporaryDirectory(prefix="hinglish_studio_") as workdir:
        studio = Studio(engines, Voices(model_dir, Path(workdir)), TakeCache(Path(args.cache) if args.cache else None),
                        presets, benchmark, hardware)
        studio.warm_up()
        studio.start()
        logger.info(f"studio ready: http://{args.host}:{args.port}  ({device_label}, {hardware['precision']}, "
                    f"{threads} threads)")
        uvicorn.run(create_app(studio), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
