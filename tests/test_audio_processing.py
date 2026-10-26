"""Behavioural tests for data_pipeline/audio_processing.py on synthetic signals."""

import numpy as np
import pytest

from data_pipeline.audio_processing import (
    AudioConfig,
    count_clipped,
    estimate_snr_db,
    normalize_loudness,
    process_utterance,
)

SR = 24000


def tone(sec: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    # amplitude-modulated tone: speech-like level variation
    return (amp * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def silence(sec: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (noise * rng.standard_normal(int(sec * SR))).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


@pytest.fixture
def cfg() -> AudioConfig:
    return AudioConfig()


def test_trailing_silence_is_exactly_180ms(cfg):
    wav = np.concatenate([silence(0.8, 1e-4), tone(2.0), silence(1.5, 1e-4)])
    out = process_utterance(wav, SR, cfg)
    assert out.ok, out.reason
    # 2.0 s of speech + 0.10 s lead + 0.18 s tail (frame-quantised to 20 ms)
    assert out.duration_s == pytest.approx(2.0 + 0.10 + 0.18, abs=0.03)
    tail = out.audio[-int(0.18 * SR):]
    assert rms_db(tail) < rms_db(out.audio) - 30


def test_short_tail_is_padded_to_180ms(cfg):
    wav = np.concatenate([silence(0.3, 1e-4), tone(2.0)])  # speech runs to the very end
    out = process_utterance(wav, SR, cfg)
    assert out.ok, out.reason
    assert out.duration_s == pytest.approx(2.0 + 0.10 + 0.18, abs=0.03)
    assert np.abs(out.audio[-int(0.1 * SR):]).max() < 1e-3


def test_leading_dead_air_is_trimmed_to_100ms(cfg):
    wav = np.concatenate([silence(2.0, 1e-4), tone(1.8), silence(0.5, 1e-4)])
    out = process_utterance(wav, SR, cfg)
    head = out.audio[: int(0.08 * SR)]
    assert rms_db(head) < rms_db(out.audio) - 30
    assert out.duration_s < 1.8 + 0.4


def test_active_speech_rms_is_normalized_to_minus_20_dbfs(cfg):
    out = process_utterance(np.concatenate([silence(0.3, 1e-4), tone(3.0, amp=0.02), silence(0.5, 1e-4)]), SR, cfg)
    assert out.ok, out.reason
    assert out.rms_dbfs == pytest.approx(-20.0, abs=0.5)
    assert out.peak_dbfs <= -1.0 + 1e-6


def test_peak_ceiling_wins_over_rms_target():
    spiky = np.zeros(SR * 2, dtype=np.float32)
    spiky[::2400] = 0.9  # very high crest factor: RMS -20 dBFS would push peaks past full scale
    spiky += tone(2.0, amp=0.01)
    gain_db, peak_limited = normalize_loudness(spiky, mode="rms", target_dbfs=-20.0, peak_ceiling_dbfs=-1.0)
    assert peak_limited
    assert 20 * np.log10(np.abs(spiky * 10 ** (gain_db / 20)).max()) == pytest.approx(-1.0, abs=0.01)


def test_peak_mode_normalizes_the_peak():
    x = tone(1.0, amp=0.5)
    gain_db, _ = normalize_loudness(x, mode="peak", target_dbfs=-20.0, peak_ceiling_dbfs=-1.0)
    assert 20 * np.log10(np.abs(x * 10 ** (gain_db / 20)).max()) == pytest.approx(-20.0, abs=0.01)


def test_clipped_samples_are_counted():
    x = tone(1.0, amp=0.5)
    assert count_clipped(x, 0.999) == 0
    x[1000:1005] = 1.0
    assert count_clipped(x, 0.999) == 5


def test_clipped_source_is_rejected(cfg):
    x = np.concatenate([silence(0.3, 1e-4), tone(2.0, amp=0.5), silence(0.3, 1e-4)])
    x[5000:5010] = 1.0
    out = process_utterance(x, SR, cfg)
    assert not out.ok and out.reason == "clipping"


def test_snr_estimate_separates_clean_from_noisy():
    clean = np.concatenate([silence(0.5, 1e-4), tone(2.0), silence(0.5, 1e-4)])
    noisy = np.concatenate([silence(0.5, 0.03), tone(2.0) + silence(2.0, 0.03, seed=1), silence(0.5, 0.03, seed=2)])
    assert estimate_snr_db(clean, SR) > 40
    assert estimate_snr_db(noisy, SR) < 25


def test_too_short_and_too_long_are_rejected(cfg):
    assert process_utterance(np.concatenate([silence(0.2, 1e-4), tone(0.9), silence(0.2, 1e-4)]), SR, cfg).reason == "too_short"
    assert process_utterance(np.concatenate([silence(0.2, 1e-4), tone(15.0), silence(0.2, 1e-4)]), SR, cfg).reason == "too_long"


def test_long_internal_pause_is_measured(cfg):
    x = np.concatenate([silence(0.2, 1e-4), tone(1.5), silence(2.0, 1e-4), tone(1.5), silence(0.3, 1e-4)])
    out = process_utterance(x, SR, cfg)
    assert out.max_pause_s == pytest.approx(2.0, abs=0.05)
    assert not out.ok and out.reason == "long_pause"


def test_resamples_48k_to_24k(cfg):
    t = np.arange(int(2.5 * 48000)) / 48000
    x48 = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x48 = np.concatenate([np.zeros(4800, np.float32), x48, np.zeros(9600, np.float32)])
    out = process_utterance(x48, 48000, cfg)
    assert out.ok, out.reason
    assert out.sample_rate == 24000
    assert out.duration_s == pytest.approx(2.5 + 0.10 + 0.18, abs=0.03)
