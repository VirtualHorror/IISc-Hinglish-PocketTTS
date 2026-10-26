"""Behavioural tests for the Hinglish dual-script normalizer (data_pipeline/hinglish_text_processor.py)."""

import unicodedata

import pytest

from data_pipeline.hinglish_text_processor import HinglishTextProcessor


def N(s: str) -> str:
    """Canonical (NFC) form: nukta letters have precomposed and decomposed spellings."""
    return unicodedata.normalize("NFC", s)


@pytest.fixture(scope="module")
def proc() -> HinglishTextProcessor:
    return HinglishTextProcessor()


# --- The three renderings from the assignment brief --------------------------------------------

DEVA = N("कल सुबह सात बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो.")
ROMAN = "Kal subah 7 baje meri meeting hai, please reminder set kar do."
CODE_SWITCHED = N("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो.")


def test_brief_example_from_devanagari_source(proc):
    ann = proc.annotate("कल सुबह 7 बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो")
    assert ann.text_devanagari == DEVA
    assert ann.text_roman == ROMAN
    assert ann.text_code_switched == CODE_SWITCHED


def test_brief_example_from_code_switched_source(proc):
    ann = proc.annotate("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो")
    assert ann.text_devanagari == DEVA
    assert ann.text_roman == ROMAN
    assert ann.text_code_switched == CODE_SWITCHED


def test_brief_example_from_roman_source(proc):
    ann = proc.annotate("kal subah 7 baje meri meeting hai, please reminder set kar do")
    assert ann.text_devanagari == DEVA
    assert ann.text_roman == ROMAN
    assert ann.text_code_switched == CODE_SWITCHED


def test_spoken_forms_verbalize_numbers_with_one_to_one_words(proc):
    ann = proc.annotate("कल सुबह 7 बजे मेरी मीटिंग है")
    assert ann.spoken_roman == "Kal subah saat baje meri meeting hai."
    assert ann.spoken_code_switched == N("कल सुबह सात बजे मेरी meeting है.")
    n = len(ann.spoken_devanagari.split())
    assert n == len(ann.spoken_roman.split()) == len(ann.spoken_code_switched.split())
    assert len(ann.spoken_words) == n


def test_language_tags(proc):
    assert proc.annotate("आज मौसम बहुत अच्छा है").language == "hi"
    assert proc.annotate("the house of the strong wind.").language == "en"
    assert proc.annotate("इस tutorial में हम निम्न के बारे में सीखेंगे").language == "hi-en"


# --- Devanagari -> Roman Hinglish (schwa deletion + casual conventions) -------------------------

@pytest.mark.parametrize(
    "deva, roman",
    [
        ("कल", "kal"), ("सुबह", "subah"), ("बजे", "baje"), ("मेरी", "meri"), ("कर", "kar"),
        ("समझना", "samajhna"), ("कमला", "kamla"), ("अपना", "apna"), ("बदलना", "badalna"),
        ("अच्छा", "accha"), ("बच्चा", "baccha"), ("पत्थर", "patthar"), ("बुद्धि", "buddhi"),
        ("लड़का", "ladka"), ("पढ़ना", "padhna"), ("हिंदी", "hindi"), ("संबंध", "sambandh"),
        ("कृपया", "kripya"), ("क्या", "kya"), ("कहना", "kehna"), ("पहले", "pehle"),
        ("दिल्ली", "dilli"), ("पानी", "paani"), ("काम", "kaam"), ("ज़िंदगी", "zindagi"),
        ("सीखेंगे", "seekhenge"), ("हाँ", "haan"), ("मित्र", "mitra"),
    ],
)
def test_romanize_hindi_word(proc, deva, roman):
    assert proc.romanize_hindi_word(N(deva)) == roman


@pytest.mark.parametrize(
    "deva, roman",
    [("है", "hai"), ("हैं", "hain"), ("में", "mein"), ("मैं", "main"), ("नहीं", "nahin"),
     ("हम", "hum"), ("यह", "yeh"), ("गई", "gayi"), ("गए", "gaye")],
)
def test_romanize_function_words(proc, deva, roman):
    assert proc.romanize_hindi_word(N(deva)) == roman


# --- English -> Devanagari phonetics -------------------------------------------------------------

@pytest.mark.parametrize(
    "en, deva",
    [
        ("meeting", "मीटिंग"), ("please", "प्लीज़"), ("reminder", "रिमाइंडर"), ("set", "सेट"),
        ("phone", "फ़ोन"), ("doctor", "डॉक्टर"), ("school", "स्कूल"), ("station", "स्टेशन"),
        ("time", "टाइम"), ("office", "ऑफ़िस"), ("bank", "बैंक"), ("house", "हाउस"),
        ("wind", "विंड"), ("the", "द"), ("what", "व्हाट"), ("computer", "कंप्यूटर"),
        ("tutorial", "ट्यूटोरियल"), ("data", "डेटा"),
        # regressions from the corpus audit
        ("patterns", "पैटर्न्ज़"), ("sorrowing", "सॉरोइंग"), ("mcq", "एमसीक्यू"),
        ("payments", "पेमेंट्स"), ("development", "डिवेलपमेंट"),
    ],
)
def test_english_to_devanagari(proc, en, deva):
    assert proc.english_to_devanagari(en) == N(deva)


# --- Roman Hinglish -> Devanagari -----------------------------------------------------------------

def test_roman_hinglish_phrase_to_devanagari(proc):
    assert proc.to_devanagari("aap kaise hain") == N("आप कैसे हैं.")


def test_roman_hindi_words_are_not_mistaken_for_english(proc):
    labels = proc.word_languages("kal subah 7 baje meri meeting hai, please reminder set kar do")
    assert labels == ["hi", "hi", "num", "hi", "hi", "en", "hi", "en", "en", "en", "hi", "hi"]


# --- Numbers -------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "number, words",
    [("7", "सात"), ("25", "पच्चीस"), ("99", "निन्यानवे"), ("2800", "दो हज़ार आठ सौ"),
     ("1,50,000", "एक लाख पचास हज़ार"), ("3.5", "तीन दशमलव पाँच")],
)
def test_hindi_number_words(proc, number, words):
    assert " ".join(proc.verbalize_number(number, lang="hi")) == N(words)


@pytest.mark.parametrize(
    "number, words",
    [("7", "seven"), ("25", "twenty five"), ("2800", "two thousand eight hundred"),
     ("101", "one hundred one"), ("3.5", "three point five")],
)
def test_english_number_words(proc, number, words):
    assert " ".join(proc.verbalize_number(number, lang="en")) == words


def test_years_are_read_as_years(proc):
    assert " ".join(proc.verbalize_number("1947", lang="hi", year=True)) == N("उन्नीस सौ सैंतालीस")
    assert " ".join(proc.verbalize_number("1947", lang="en", year=True)) == "nineteen forty seven"


# --- Source-specific behaviour -------------------------------------------------------------------

def test_english_source_sentence(proc):
    ann = proc.annotate("the house of the strong wind.")
    assert ann.text_roman == "The house of the strong wind."
    assert ann.text_code_switched == "The house of the strong wind."
    assert ann.text_devanagari.startswith(N("द हाउस ऑफ़ द "))
    assert ann.text_devanagari.endswith(N(" विंड."))


def test_human_code_switching_is_preserved(proc):
    ann = proc.annotate("इस tutorial में हम निम्न के बारे में सीखेंगे")
    assert ann.text_code_switched == N("इस tutorial में हम निम्न के बारे में सीखेंगे.")
    assert ann.text_devanagari == N("इस ट्यूटोरियल में हम निम्न के बारे में सीखेंगे.")
    assert ann.text_roman == "Is tutorial mein hum nimn ke baare mein seekhenge."


def test_danda_and_pipe_become_periods(proc):
    ann = proc.annotate("यह किताब मेरी है। वह कलम तुम्हारी है|")
    assert ann.text_devanagari == N("यह किताब मेरी है. वह कलम तुम्हारी है.")


def test_question_mark_is_kept(proc):
    assert proc.annotate("आप कैसे हैं?").text_roman == "Aap kaise hain?"


@pytest.mark.parametrize(
    "native", ["से", "बस", "कम", "सब", "पर", "दम", "मन", "जब", "तब", "चीज़", "सुन", "पुल", "कल",
               # converted to English before the corpus audit (more/then/were/some/basin/...)
               "मोर", "देन", "वर", "सम", "मरी", "बेसन", "सेज", "शैल", "एसे", "मेज़", "काज"]
)
def test_native_hindi_words_are_not_converted_to_english(proc, native):
    # Several of these spell a common English word in Devanagari (कम ~ "come", चीज़ ~ "cheese",
    # सुन ~ "soon", पुल ~ "pull"); frequent native words must still stay in Devanagari.
    ann = proc.annotate(f"मैं {native} आज आऊँगा")
    assert N(native) in ann.text_code_switched.split()


@pytest.mark.parametrize("deva, en", [("फेल", "fail"), ("लेटर", "letter"), ("ट्रेंड", "trend"),
                                      ("पेन", "pen"), ("कोट्स", "quotes")])
def test_homophone_loanwords_get_the_intended_spelling(proc, deva, en):
    assert proc.detect_loanword(N(deva)) == en


@pytest.mark.parametrize("native", ["बाईस", "पच्चीस", "परिजन", "राजकुमार", "चूस", "टीस"])
def test_variant_matching_does_not_capture_native_words(proc, native):
    # बाईस (22) ~ "buys", परिजन ~ "parisian", राजकुमार is a CMUdict name entry, चूस ~ "choose"
    assert proc.detect_loanword(N(native)) is None


@pytest.mark.parametrize("deva, en", [("फॉल्स", "false"), ("स्टेटस", "status"), ("फोरम", "forum")])
def test_ties_and_variant_homophones_pick_the_intended_word(proc, deva, en):
    assert proc.detect_loanword(N(deva)) == en


@pytest.mark.parametrize("deva, en", [("गेम्स", "games"), ("कॉमनवेल्थ", "commonwealth"),
                                      ("वेटलिफ्टिंग", "weightlifting")])
def test_loanwords_written_with_hindi_spelling_conventions(proc, deva, en):
    # plural -s written स (CMUdict says Z); syllable-boundary clusters written without virama
    assert proc.detect_loanword(N(deva)) == en


def test_common_loanwords_become_latin_in_code_switched_text(proc):
    ann = proc.annotate("मैं पुलिस स्टेशन जाऊँगा")
    assert ann.text_code_switched == N("मैं police station जाऊँगा.")
    assert ann.text_roman == "Main police station jaaoonga."
