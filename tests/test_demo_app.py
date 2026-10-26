"""demo_app.py helpers that run without a model: tokenizer-piece classes, script detection, take keys,
the pre-rendered take cache, the replay memory, voice-upload validation and the stream encoding."""

import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

import demo_app
from release.inference import VARIANTS


@pytest.mark.parametrize("piece,kind", [
    ("▁के", "deva"), ("ाइट", "deva"), ("▁ke", "latin"), ("▁meet", "latin"), ("<0xE2>", "byte"),
    ("▁", "space"), (",", "punct"), (".", "punct"), ("7", "digit"),
])
def test_classify_piece(piece, kind):
    assert demo_app.classify_piece(piece) == kind


def test_input_script():
    assert demo_app.input_script("Kal subah 7 baje meri flight hai") == "roman"
    assert demo_app.input_script("कल सुबह सात बजे मेरी फ़्लाइट है।") == "devanagari"
    assert demo_app.input_script("मेरी next meeting दोपहर 3 बजे है") == "mixed"
    assert demo_app.input_script("123 ?") == "none"


def test_take_key_ignores_whitespace_but_not_settings():
    key = demo_app.take_key("24l", "indian_female_ananya", "phonetic", "Kal subah  7 baje")
    assert key == demo_app.take_key("24l", "indian_female_ananya", "phonetic", " Kal subah 7 baje ")
    others = {demo_app.take_key(*args) for args in [
        ("12l", "indian_female_ananya", "phonetic", "Kal subah 7 baje"),
        ("24l", "indian_male_aarav", "phonetic", "Kal subah 7 baje"),
        ("24l", "indian_female_ananya", "raw", "Kal subah 7 baje"),
        ("24l", "indian_female_ananya", "phonetic", "Kal subah 8 baje")]}
    assert key not in others and len(others) == 4


def test_replay_memory_evicts_least_recent():
    memory = demo_app.ReplayMemory(slots=2)
    for k in "abc":
        memory.put(k, {"k": k}, np.zeros(1, dtype=np.float32))
    assert memory.get("a") is None and memory.get("b") is not None
    memory.put("d", {}, np.zeros(1, dtype=np.float32))  # "b" was just used, so "c" goes
    assert memory.get("c") is None and memory.get("b") is not None


def test_take_cache_serves_only_takes_with_audio(tmp_path):
    audio = (0.1 * np.sin(np.linspace(0, 200, 24000))).astype(np.float32)
    sf.write(str(tmp_path / "a.wav"), audio, 24000, subtype="PCM_16")
    takes = [{"key": "a", "file": "a.wav", "ttfa_ms": 30.0}, {"key": "b", "file": "missing.wav"}]
    (tmp_path / "index.json").write_text(json.dumps({"rendered_on": "test", "takes": takes}))
    cache = demo_app.TakeCache(tmp_path)
    assert set(cache.takes) == {"a"} and cache.meta["rendered_on"] == "test"
    meta, got = cache.get("a")
    assert meta["ttfa_ms"] == 30.0 and len(got) == 24000 and np.abs(got - audio).max() < 1e-3
    assert cache.get("b") is None
    assert demo_app.TakeCache(None).takes == {}


def _wav_bytes(seconds: float, amplitude: float = 0.2, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_voice_upload_validation(tmp_path):
    voices = demo_app.Voices(tmp_path, tmp_path)
    added = voices.add_upload(_wav_bytes(20.0), "my voice.wav")
    assert added["id"].startswith("upload_") and added["label"] == "my voice"
    assert added["trimmed"] and added["duration_s"] == demo_app.CLONE_MAX_SECONDS
    assert sf.info(voices.resolve(added["id"])).duration == pytest.approx(demo_app.CLONE_MAX_SECONDS, abs=0.01)
    for data, message in [(b"not audio at all", "Could not read"), (_wav_bytes(0.5), "at least"),
                          (_wav_bytes(2.0, amplitude=0.0), "silent")]:
        with pytest.raises(ValueError, match=message):
            voices.add_upload(data, "x.wav")
    with pytest.raises(KeyError):
        voices.resolve("nobody")


def test_stream_encoding_round_trip():
    audio = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    decoded = np.frombuffer(base64.b64decode(demo_app.pcm16_b64(audio)), dtype="<i2") / 32767.0
    assert np.allclose(decoded, [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-4)  # clipped to [-1, 1]
    line = demo_app.event("done", rtf=0.5, text="मेरी meeting")
    assert line.endswith(b"\n") and json.loads(line) == {"type": "done", "rtf": 0.5, "text": "मेरी meeting"}


def test_threads_and_variants():
    assert 1 <= demo_app.default_threads() <= 4
    assert VARIANTS["24l"] == "" and all(v.startswith("student_") for k, v in VARIANTS.items() if k != "24l")
