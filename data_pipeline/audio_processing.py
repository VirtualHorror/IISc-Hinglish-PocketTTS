"""Audio preprocessing for TTS training data.

process_utterance(): mono -> 24 kHz (soxr HQ) -> trim dead air to a 0.10 s lead and a crisp 0.18 s
end-on-pause tail -> loudness-normalise -> quality metrics (clipping and blind SNR measured on the
SOURCE audio, internal pauses on the trimmed signal). Voiced/silent frames use the same rule as
pocket_tts.data.audio_utils.end_on_pause: 20 ms frames more than `silence_floor_db` below the
loudest frame are silence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import soxr

FRAME_S = 0.02
DB_FLOOR = -120.0


@dataclass
class AudioConfig:
    target_sr: int = 24000
    min_duration_s: float = 1.5
    max_duration_s: float = 14.0
    lead_silence_s: float = 0.10
    trail_silence_s: float = 0.18
    silence_floor_db: float = 40.0
    fade_s: float = 0.01
    loudness_mode: str = "rms"
    target_dbfs: float = -20.0
    peak_ceiling_dbfs: float = -1.0
    clip_threshold: float = 0.999
    max_clipped_samples: int = 0
    min_snr_db: float = 25.0
    max_internal_pause_s: float = 1.5

    @classmethod
    def from_yaml(cls, cfg: dict, target_sr: int = 24000) -> "AudioConfig":
        a = dict(cfg)
        loud = a.pop("loudness", {})
        a.update(loudness_mode=loud.get("mode", "rms"), target_dbfs=loud.get("target_dbfs", -20.0),
                 peak_ceiling_dbfs=loud.get("peak_ceiling_dbfs", -1.0), target_sr=target_sr)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in a.items() if k in names})


@dataclass
class ProcessedAudio:
    ok: bool
    reason: str
    audio: np.ndarray | None
    sample_rate: int
    duration_s: float = 0.0
    snr_db: float = 0.0
    clipped_samples: int = 0
    max_pause_s: float = 0.0
    rms_dbfs: float = DB_FLOOR
    peak_dbfs: float = DB_FLOOR
    gain_db: float = 0.0
    peak_limited: bool = False
    source_sample_rate: int = 0
    source_duration_s: float = 0.0

    def metrics(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items() if k != "audio"}


def to_mono_float(wav: np.ndarray) -> np.ndarray:
    x = np.asarray(wav)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:  # soundfile gives (frames, channels)
        x = x.mean(axis=1 if x.shape[1] <= 8 else 0)
    return x


def frame_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> tuple[np.ndarray, int]:
    hop = max(1, int(frame_s * sr))
    n = len(x) // hop
    if n == 0:
        return np.array([DB_FLOOR]), hop
    frames = x[: n * hop].astype(np.float64).reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 10 ** (DB_FLOOR / 20))), hop


def count_clipped(x: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(np.abs(x) >= threshold))


def estimate_snr_db(x: np.ndarray, sr: int) -> float:
    """Blind frame-energy SNR: loud-speech level (95th pct of 20 ms frame dB) minus the noise
    floor (10th pct). Conservative on tightly cut clips (little silence -> floor overestimated)."""
    db, _ = frame_db(x, sr)
    if len(db) < 5:
        return 0.0
    return float(min(np.percentile(db, 95) - np.percentile(db, 10), 80.0))


def normalize_loudness(x: np.ndarray, mode: str = "rms", target_dbfs: float = -20.0,
                       peak_ceiling_dbfs: float = -1.0, active: np.ndarray | None = None
                       ) -> tuple[float, bool]:
    """Gain (dB) to apply, and whether the peak ceiling (not the target) set it.

    rms: RMS of `active` (default: x) -> target_dbfs, capped so max|x| <= peak_ceiling_dbfs.
    peak: max|x| -> target_dbfs.
    """
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= 0:
        return 0.0, False
    peak_db = 20 * np.log10(peak)
    if mode == "peak":
        return target_dbfs - peak_db, False
    ref = x if active is None or len(active) == 0 else active
    rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), 1e-9))
    gain = target_dbfs - rms_db
    if peak_db + gain > peak_ceiling_dbfs:
        return peak_ceiling_dbfs - peak_db, True
    return gain, False


def process_utterance(wav: np.ndarray, sr: int, cfg: AudioConfig) -> ProcessedAudio:
    x = to_mono_float(wav)
    res = ProcessedAudio(False, "", None, cfg.target_sr, source_sample_rate=int(sr),
                         source_duration_s=len(x) / sr if sr else 0.0)
    if len(x) < sr * 0.1:
        res.reason = "empty"
        return res
    res.clipped_samples = count_clipped(x, cfg.clip_threshold)
    res.snr_db = estimate_snr_db(x, sr)
    y = x if sr == cfg.target_sr else soxr.resample(x, sr, cfg.target_sr, quality="HQ").astype(np.float32)
    out_sr = cfg.target_sr
    db, hop = frame_db(y, out_sr)
    voiced = np.nonzero(db > db.max() - cfg.silence_floor_db)[0]
    if db.max() <= DB_FLOOR + 1 or len(voiced) == 0:
        res.reason = "silent"
        return res
    gaps = np.diff(voiced) - 1
    res.max_pause_s = float(gaps.max() * hop / out_sr) if len(gaps) else 0.0

    lead, trail = int(cfg.lead_silence_s * out_sr), int(cfg.trail_silence_s * out_sr)
    speech_start, speech_end = int(voiced[0]) * hop, (int(voiced[-1]) + 1) * hop
    start = max(0, speech_start - lead)
    end = min(len(y), speech_end + trail)
    seg = y[start:end].copy()
    fade = max(1, int(cfg.fade_s * out_sr))
    if start > 0:
        seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pad = speech_end + trail - end
    if pad > 0:
        seg = np.concatenate([seg, np.zeros(pad, dtype=np.float32)])
    res.duration_s = len(seg) / out_sr

    active = np.concatenate([y[i * hop:(i + 1) * hop] for i in voiced])
    res.gain_db, res.peak_limited = normalize_loudness(
        seg, cfg.loudness_mode, cfg.target_dbfs, cfg.peak_ceiling_dbfs, active=active)
    g = 10 ** (res.gain_db / 20)
    seg = (seg * g).astype(np.float32)
    res.rms_dbfs = float(20 * np.log10(max(np.sqrt(np.mean((active.astype(np.float64) * g) ** 2)), 1e-9)))
    res.peak_dbfs = float(20 * np.log10(max(float(np.abs(seg).max()), 1e-9)))
    res.audio = seg

    if res.clipped_samples > cfg.max_clipped_samples:
        res.reason = "clipping"
    elif res.duration_s < cfg.min_duration_s:
        res.reason = "too_short"
    elif res.duration_s > cfg.max_duration_s:
        res.reason = "too_long"
    elif res.snr_db < cfg.min_snr_db:
        res.reason = "low_snr"
    elif res.max_pause_s > cfg.max_internal_pause_s:
        res.reason = "long_pause"
    else:
        res.ok = True
    return res
