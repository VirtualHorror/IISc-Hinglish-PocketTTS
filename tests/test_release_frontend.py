"""release/inference.py text front-end and eval/asr.py scoring."""

import pytest

from data_pipeline.hinglish_text_processor import HinglishTextProcessor
from eval.asr import HinglishScorer, edit_distance, fold_devanagari
from release.inference import HinglishTTS, detect_script, ensure_terminal_punctuation


@pytest.fixture(scope="module")
def proc():
    return HinglishTextProcessor()


@pytest.mark.parametrize("text,expected", [
    ("Kal subah meeting hai", "Kal subah meeting hai."),
    ("आज ट्रैफ़िक बहुत ज़्यादा है।", "आज ट्रैफ़िक बहुत ज़्यादा है."),
    ("क्या आपने मैच देखा?", "क्या आपने मैच देखा?"),
    ("Battery low hai!", "Battery low hai!"),
    ("मेरा फ़ोन हैंग हो रहा है,", "मेरा फ़ोन हैंग हो रहा है."),
    ('He said "wait"', 'He said "wait".'),
    ('He said "wait."', 'He said "wait."'),
    ("  two   spaces\nhere ", "two spaces here."),
    ("श्लोक ॥", "श्लोक ."),
])
def test_ensure_terminal_punctuation(text, expected):
    assert ensure_terminal_punctuation(text) == expected


def test_ensure_terminal_punctuation_rejects_empty():
    with pytest.raises(ValueError):
        ensure_terminal_punctuation("   ")


def test_detect_script():
    assert detect_script("Kal subah meeting hai") == "roman"
    assert detect_script("कल सुबह मीटिंग है") == "devanagari"
    assert detect_script("कल सुबह meeting है") == "code_switched"


def test_normalize_with_and_without_processor(proc):
    tts = HinglishTTS(model=None, auto_normalize=False)
    assert tts.normalize("कल सुबह 7 बजे मेरी meeting है") == "कल सुबह 7 बजे मेरी meeting है."
    tts.processor = proc  # default script: Devanagari, English words spelled phonetically
    assert tts.normalize("कल सुबह 7 बजे मेरी meeting है") == "कल सुबह सात बजे मेरी मीटिंग है."
    assert tts.normalize("Kal subah 7 baje meeting hai") == "कल सुबह सात बजे मीटिंग है."
    assert tts.normalize("कल सुबह 7 बजे मेरी meeting है", script="auto") == "कल सुबह सात बजे मेरी meeting है."
    assert tts.normalize("Kal subah 7 baje meeting hai", script="auto") == "Kal subah saat baje meeting hai."
    assert tts.normalize("कल सुबह 7 बजे", auto_normalize=False) == "कल सुबह 7 बजे."


def test_fold_keeps_vowel_signs():
    # regression: re's \w does not match Devanagari vowel signs, so a \w-based filter dropped them
    assert fold_devanagari("मीटिंग है।") == "मीटिंग है"
    assert fold_devanagari("ज़रूरत, हाँ!") == "जरूरत हां"
    assert fold_devanagari("हिन्दी") == fold_devanagari("हिंदी")
    assert fold_devanagari("डॉक्टर") == fold_devanagari("डोक्टर")


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_scorer_is_script_invariant(proc):
    scorer = HinglishScorer(proc)
    refs = ["Kal subah 7 baje meri meeting hai, please reminder set kar do."]
    hyps = ["कल सुबह सात बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो।"]
    s = scorer.score(refs, hyps)
    assert s["cer"] == 0.0 and s["wer"] == 0.0
    s = scorer.score(["मेरी meeting है।"], ["मेरी मीटिंग नहीं है"])
    assert s["word_edits"] == 1 and s["words"] == 3


def test_release_config_points_at_the_repo():
    import yaml
    from pocket_tts.utils.config import Config

    from release.export_release import MODEL_CONFIG, REPO_ID, release_config

    cfg = yaml.safe_load(release_config(MODEL_CONFIG))
    assert cfg["weights_path"] == f"hf://{REPO_ID}/model.safetensors"
    assert cfg["flow_lm"]["lookup_table"]["tokenizer_path"] == f"hf://{REPO_ID}/tokenizer.model"
    assert "weights_path_without_voice_cloning" not in cfg
    Config(**cfg)  # pocket-tts's own schema


def test_bundled_text_processor_works_standalone(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from release.export_release import bundle_text_processor

    bundle_text_processor(tmp_path / "hinglish_text")
    repo = Path(__file__).resolve().parent.parent
    # dependencies only (absolute), never the repo: the bundle must not need data_pipeline/
    deps = [str(Path(p).resolve()) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(d for d in deps if Path(d) != repo)}
    code = ("import importlib.util; assert importlib.util.find_spec('data_pipeline') is None; "
            "from hinglish_text.hinglish_text_processor import HinglishTextProcessor as P; "
            "print(P().normalize_for_tts('Kal 7 baje milte hain', script='roman'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert out.strip() == "Kal saat baje milte hain."


def test_roman_majority_text_with_devanagari_words(proc):
    tts = HinglishTTS(model=None, auto_normalize=False)
    tts.processor = proc
    roman = tts.normalize("Kal subah 7 baje meri meeting hai, please reminder set kar do")
    assert tts.normalize("Kal subah 7 baje meri meeting hai, please reminder सेट कर दो") == roman
    assert roman == "कल सुबह सात बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो."
    # Devanagari-majority code-switched text keeps "Latin = English"
    assert tts.normalize("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो") == roman
