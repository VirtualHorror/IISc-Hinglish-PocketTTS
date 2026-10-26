"""Bidirectional Hinglish script & phonetic normalizer.

Every transcript gets three renderings with identical word segmentation:

  text_devanagari     full Devanagari: English words in Devanagari phonetics, numbers verbalised
                      ("कल सुबह सात बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो.")
  text_roman          Roman Hinglish: Hindi words in casual romanisation, English words as English
                      ("Kal subah 7 baje meri meeting hai, please reminder set kar do.")
  text_code_switched  Hindi words in Devanagari, English words in Latin
                      ("कल सुबह 7 बजे मेरी meeting है, please reminder सेट कर दो.")

The `spoken_*` forms verbalise numbers in all three scripts, so word i of each spoken form is a
rendering of the same spoken word; a word alignment computed on one script therefore holds for the
other two (used to build the word-aligned Pocket TTS training manifests).

Inputs can be Devanagari, Roman Hinglish, English, or human code-switched text. Human script
choices in mixed-script input are kept as-is in the code-switched rendering.

    from data_pipeline.hinglish_text_processor import HinglishTextProcessor
    ann = HinglishTextProcessor().annotate("aap kaise hain")
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field

from data_pipeline import hinglish_lexicon as lex

# ---------------------------------------------------------------------------------------------
# Unicode tables
# ---------------------------------------------------------------------------------------------

VIRAMA, NUKTA = "्", "़"
ANUSVARA, CANDRABINDU, VISARGA = "ं", "ँ", "ः"

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "zh", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f",
                    "य": "y", "र": "r", "न": "n", "ल": "l"}
INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ॠ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o", "ऌ": "li",
}
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "ॄ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o", "ॢ": "li",
}
LABIALS = set("पफबभम")
SHORTEN_FINAL = {"aa": "a", "ee": "i", "oo": "u"}

DEVA_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
NUMBER = re.compile(r"^(?P<cur>₹)?(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>%)?$")
TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$", re.I)
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

PUNCT_MAP = str.maketrans({
    "।": ".", "॥": ".", "|": ".", "‘": "'", "’": "'", "`": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‐": "-", "…": ".", " ": " ", "\t": " ", "\n": " ", "\r": " ",
})
LEADING_PUNCT = "\"'(["
TRAILING_PUNCT = ".,?!;:\"')]"
TERMINAL = ".?!"
WEAK = ",;:-"

# ---------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------


@dataclass
class Token:
    lead: str
    core: str
    trail: str
    kind: str  # "deva" | "latn" | "num" | "other"
    lang: str = ""  # "hi" | "en" | "num"
    english: str | None = None  # English spelling for a Devanagari loanword


@dataclass
class Annotation:
    text_original: str
    text_devanagari: str
    text_roman: str
    text_code_switched: str
    spoken_devanagari: str
    spoken_roman: str
    spoken_code_switched: str
    spoken_words: list[tuple[str, str, str]] = field(default_factory=list)
    language: str = "hi"  # "hi" | "en" | "hi-en"
    n_english_words: int = 0
    n_hindi_words: int = 0


# ---------------------------------------------------------------------------------------------
# Shared resources (loaded once per process)
# ---------------------------------------------------------------------------------------------


@functools.cache
def _cmudict() -> dict[str, list[list[str]]]:
    import cmudict

    return cmudict.dict()


@functools.cache
def _zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, lang)


def clean_text(text: str) -> str:
    """NFC, punctuation/digit normalisation, junk removal, punctuation spacing."""
    t = unicodedata.normalize("NFC", text)
    for zw in ("​", "‌", "‍", "﻿", "­"):
        t = t.replace(zw, "")
    t = t.translate(PUNCT_MAP).translate(DEVA_DIGITS)
    t = t.replace("&", " and ").replace("/", " ").replace("_", " ")
    for abbr, full in lex.SPOKEN_ABBREVIATIONS.items():
        t = re.sub(rf"\b{abbr}\b", full, t, flags=re.IGNORECASE)
    t = re.sub(r"[^ऀ-ॿA-Za-z0-9\s.,?!;:'\"\-()%₹]", " ", t)
    t = re.sub(r"\s+([.,?!;:])", r"\1", t)  # no space before punctuation
    t = re.sub(r"([.?!])[.?!]+", r"\1", t)  # "..", "?!" -> single
    t = re.sub(r"([,;:])[,;:]+", r"\1", t)
    t = re.sub(r"([.,?!;:])(?=[^\s\d.,?!;:'\")\]])", r"\1 ", t)  # space after (not inside 3.5)
    t = re.sub(r"\s+-\s+", " ", t)  # free-standing dashes
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for raw in text.split():
        lead = raw[: len(raw) - len(raw.lstrip(LEADING_PUNCT))]
        rest = raw[len(lead):]
        core = rest.rstrip(TRAILING_PUNCT)
        trail = rest[len(core):]
        if not core:  # bare punctuation: attach to the previous word
            if tokens:
                tokens[-1].trail += lead + trail
            continue
        has_deva, has_latn = bool(DEVA_LETTER.search(core)), bool(LATIN_LETTER.search(core))
        if has_deva and has_latn:  # e.g. "PDFफ़ाइल": split at script boundaries
            parts = re.findall(r"[ऀ-ॿ]+|[^ऀ-ॿ]+", core)
            for i, part in enumerate(parts):
                tokens.append(Token(lead if i == 0 else "", part,
                                    trail if i == len(parts) - 1 else "", _kind(part)))
            continue
        tokens.append(Token(lead, core, trail, _kind(core)))
    return tokens


def _kind(core: str) -> str:
    if DEVA_LETTER.search(core):
        return "deva"
    if LATIN_LETTER.search(core) and not re.search(r"\d", core):
        return "latn"
    if NUMBER.match(core) or TIME.match(core) or ORDINAL.match(core):
        return "num"
    return "latn" if LATIN_LETTER.search(core) else "other"


# ---------------------------------------------------------------------------------------------
# Devanagari -> Roman Hinglish
# ---------------------------------------------------------------------------------------------


@dataclass
class _Unit:
    cons: str | None  # roman consonant
    deva: str | None  # Devanagari consonant (base letter)
    vowel: str | None  # roman vowel; None = halant (bare consonant)
    inherent: bool = False
    nasal: str = ""  # "" | "anusvara" | "candrabindu"
    visarga: bool = False
    independent: bool = False
    deleted: bool = False


def _parse_devanagari(word: str) -> list[_Unit]:
    units: list[_Unit] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            i += 1
            rom = CONSONANTS[ch]
            if i < n and word[i] == NUKTA:
                rom = NUKTA_CONSONANTS.get(ch, rom)
                i += 1
            unit = _Unit(rom, ch, "a", inherent=True)
            if i < n and word[i] == VIRAMA:
                unit.vowel, unit.inherent = None, False
                i += 1
            elif i < n and word[i] in MATRAS:
                unit.vowel, unit.inherent = MATRAS[word[i]], False
                i += 1
            units.append(unit)
        elif ch in INDEPENDENT_VOWELS:
            units.append(_Unit(None, None, INDEPENDENT_VOWELS[ch], independent=True))
            i += 1
        elif ch in (ANUSVARA, CANDRABINDU):
            if units:
                units[-1].nasal = "anusvara" if ch == ANUSVARA else "candrabindu"
            i += 1
        elif ch == VISARGA:
            if units:
                units[-1].visarga = True
            i += 1
        else:  # stray nukta / unsupported sign
            i += 1
    return units


def _has_vowel(u: _Unit) -> bool:
    return u.vowel is not None and not u.deleted


def _delete_schwas(units: list[_Unit]) -> None:
    """Hindi schwa syncope: word-final schwa, then medial schwas in V C _ C V contexts (scanned
    right to left so each deletion feeds the next decision). Closed (nasalised) syllables and
    conjunct-final words keep theirs: ज़िंदगी -> zindagi, मित्र -> mitra."""
    if not units:
        return
    last = units[-1]
    if last.inherent and len(units) > 1 and not last.nasal:
        conjunct_final = units[-2].vowel is None
        if not (conjunct_final and last.deva in ("र", "य", "व")):
            last.deleted = True
    for j in range(len(units) - 2, 0, -1):
        u = units[j]
        if not u.inherent or u.deleted or u.nasal:
            continue
        prev, nxt = units[j - 1], units[j + 1]
        if not _has_vowel(prev) or prev.nasal or prev.visarga:
            continue
        if nxt.cons is None or not _has_vowel(nxt):
            continue
        u.deleted = True


def _romanize_units(units: list[_Unit]) -> str:
    _delete_schwas(units)
    out: list[str] = []
    n = len(units)
    for j, u in enumerate(units):
        nxt = units[j + 1] if j + 1 < n else None
        # --- consonant ---
        if u.cons is not None:
            cons = u.cons
            prev = units[j - 1] if j else None
            if prev is not None and prev.vowel is None and prev.deva in ("च",) and u.deva in ("छ", "च"):
                cons = "ch"  # च्छ / च्च -> "cch" (prev already emitted "c" below)
            if u.deva == "ञ" and prev is not None and prev.deva == "ज" and prev.vowel is None:
                cons = "y"  # ज्ञ -> gy
            out.append(cons)
        # --- vowel ---
        if u.vowel is None:
            if u.deva == "च" and nxt is not None and nxt.deva in ("छ", "च"):
                out[-1] = "c"
            if u.deva == "ज" and nxt is not None and nxt.deva == "ञ":
                out[-1] = "g"
            continue
        if u.deleted:
            continue
        vowel = u.vowel
        # "ah" before a consonant in the next syllable reads "eh": कहना kehna, पहले pehle.
        if (u.inherent and u.cons is not None and nxt is not None and nxt.deva == "ह"
                and not _has_vowel(nxt) and j + 2 < n):
            vowel = "e"
        # vowel hiatus: गई gayi, गए gaye, आए aaye
        if u.independent and j > 0 and units[j - 1].vowel in ("a", "aa") and not units[j - 1].deleted:
            if vowel in ("ee", "e"):
                vowel = "y" + ("i" if vowel == "ee" else "e")
        is_last = j == n - 1
        if is_last and (n > 1 or (u.cons is not None and not u.nasal)):  # हाँ keeps "haan"
            vowel = SHORTEN_FINAL.get(vowel, vowel)
        out.append(vowel)
        if u.nasal == "anusvara" and nxt is not None and nxt.cons is not None:
            out.append("m" if nxt.deva in LABIALS else "n")
        elif u.nasal:
            out.append("n")
        if u.visarga:
            out.append("h")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# English -> Devanagari (CMUdict ARPAbet with Indian-English conventions)
# ---------------------------------------------------------------------------------------------

ARPA_CONS = {
    "P": "प", "B": "ब", "T": "ट", "D": "ड", "K": "क", "G": "ग", "CH": "च", "JH": "ज",
    "F": "फ़", "V": "व", "TH": "थ", "DH": "द", "S": "स", "Z": "ज़", "SH": "श", "ZH": "ज़",
    "HH": "ह", "M": "म", "N": "न", "NG": "ङ", "L": "ल", "R": "र", "W": "व", "Y": "य",
}
ARPA_VOWEL = {  # (independent, matra)
    "AA": ("आ", "ा"), "AE": ("ऐ", "ै"), "AH": ("अ", ""), "AO": ("ऑ", "ॉ"), "AW": ("आउ", "ाउ"),
    "AY": ("आइ", "ाइ"), "EH": ("ए", "े"), "ER": ("अ", ""), "EY": ("ए", "े"), "IH": ("इ", "ि"),
    "IY": ("ई", "ी"), "OW": ("ओ", "ो"), "OY": ("ऑइ", "ॉइ"), "UH": ("उ", "ु"), "UW": ("ऊ", "ू"),
}
ANUSVARA_BEFORE = {"P", "B", "T", "D", "K", "G", "CH", "JH", "F", "V", "TH", "DH", "S", "Z", "SH"}


VOICELESS = {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"}
SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}


def _suffix_phones(suffix: str, stem_phones: list[str]) -> list[str]:
    last = re.sub(r"\d", "", stem_phones[-1])
    if suffix == "ed":
        return ["IH0", "D"] if last in ("T", "D") else (["T"] if last in VOICELESS else ["D"])
    if suffix in ("s", "es"):
        return ["IH0", "Z"] if last in SIBILANTS else (["S"] if last in VOICELESS else ["Z"])
    return {"ing": ["IH0", "NG"], "er": ["ER0"], "ers": ["ER0", "Z"], "est": ["IH0", "S", "T"],
            "ly": ["L", "IY0"], "ness": ["N", "AH0", "S"], "ment": ["M", "AH0", "N", "T"],
            "ments": ["M", "AH0", "N", "T", "S"]}[suffix]


def cmu_pronunciation(word: str) -> list[str] | None:
    """CMUdict lookup, falling back to stem + regular suffix for inflections it lacks
    (sorrowing = sorrow + ing, hoped = hope + d, stopped = stop + ed)."""
    cmu = _cmudict()
    if word in cmu:
        return cmu[word][0]
    for suffix in ("ments", "ment", "ness", "ing", "ers", "est", "ed", "es", "er", "ly", "s"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        stem = word[: -len(suffix)]
        doubled = stem[:-1] if len(stem) > 3 and stem[-1] == stem[-2] else None
        for cand in (stem, stem + "e", doubled, stem[:-1] + "y" if stem.endswith("i") else None):
            if cand and cand in cmu:
                phones = cmu[cand][0]
                return phones + _suffix_phones(suffix, phones)
    return None


def _vowel_letter_groups(word: str) -> list[str]:
    w = word.lower()
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]  # silent final e
    return re.findall(r"[aeiouy]+", w)


def arpabet_to_devanagari(phones: list[str], spelling: str = "") -> str:
    """ARPAbet (CMUdict) -> Devanagari: retroflex T/D, nukta F/Z, anusvara for nasal codas,
    ॉ for the 'o' of doctor/college (US AA), final schwa as ा (डेटा)."""
    base = [re.sub(r"\d", "", p) for p in phones]
    unstressed = [p.endswith("0") for p in phones]
    vowel_idx = [i for i, p in enumerate(base) if p in ARPA_VOWEL]
    groups = _vowel_letter_groups(spelling)
    letter_for = dict(zip(vowel_idx, groups)) if groups and len(groups) == len(vowel_idx) else {}
    lower = spelling.lower()
    n = len(base)
    # Indian English says -ment(s) with a full e: payment पेमेंट, development डेवलपमेंट
    ment_vowel = -1
    if lower.endswith(("ment", "ments")):
        tail = ["M", "AH", "N", "T"] + (["S"] if lower.endswith("s") else [])
        if base[-len(tail):] == tail:
            ment_vowel = n - len(tail) + 1
    out: list[str] = []
    pending: list[str] = []  # consonants waiting for their vowel
    for i, p in enumerate(base):
        nxt = base[i + 1] if i + 1 < n else None
        prev_vowel = i > 0 and base[i - 1] in ARPA_VOWEL
        if p in ARPA_VOWEL:
            indep, matra = ARPA_VOWEL[p]
            if p == "AA" and (letter_for.get(i, "").startswith("o")
                              or (not letter_for and "o" in lower and "a" not in lower)):
                indep, matra = "ऑ", "ॉ"
            elif p == "AO" and nxt == "R":
                indep, matra = "ओ", "ो"
            elif p == "AY" and nxt is None:
                indep, matra = "आय", "ाय"
            elif p == "OY" and nxt is None:
                indep, matra = "ऑय", "ॉय"
            elif p == "AH" and nxt is None:
                indep, matra = "आ", "ा"
            elif p == "IY" and unstressed[i] and nxt is not None:
                indep, matra = "इ", "ि"  # reminder -> रिमाइंडर (Indian English shortens it)
            elif p == "UW" and unstressed[i] and nxt is not None:
                indep, matra = "उ", "ु"
            elif i == ment_vowel:
                indep, matra = "ए", "े"
            if pending:
                out.append(VIRAMA.join(ARPA_CONS[c] for c in pending) + matra)
            else:
                out.append(indep)
            # ER = schwa + r: the r opens the next cluster (बर्ड), takes the next vowel, or ends the word (बटर)
            pending = ["R"] if p == "ER" else []
            continue
        if p == "NG" and not pending and i > 0:
            out.append(ANUSVARA)  # -ing -> िंग, bank -> बैंक, singer -> सिंगर
            if nxt not in ("K", "G"):
                pending = ["G"]
            continue
        if p in ("N", "M") and prev_vowel and not pending and nxt is not None and nxt not in ARPA_VOWEL and (
                (p == "N" and nxt in ANUSVARA_BEFORE) or (p == "M" and nxt in ("P", "B"))):
            out.append(ANUSVARA)
            continue
        # a stop closing one syllable before a nasal opening the next stays unjoined (फ़िटनेस)
        if pending and len(pending) == 1 and pending[0] in ("T", "D", "K", "P", "G", "B") \
                and p in ("M", "N") and out:
            out.append(ARPA_CONS[pending[0]])
            pending = []
        pending.append(p)
    if pending:
        out.append(VIRAMA.join(ARPA_CONS[c] for c in pending))
    return unicodedata.normalize("NFC", "".join(out))


# ---------------------------------------------------------------------------------------------
# Roman Hinglish -> Devanagari (rule fallback; the lexicon index handles common words)
# ---------------------------------------------------------------------------------------------

ROMAN_CONS = [
    ("chh", "छ"), ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
    ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("c", "क"),
    ("j", "ज"), ("t", "त"), ("d", "द"), ("n", "न"), ("p", "प"), ("f", "फ़"), ("b", "ब"), ("m", "म"),
    ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"),
    ("z", "ज़"), ("q", "क़"), ("x", "क्स"),
]
ROMAN_VOWELS = [
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"), ("ii", "ई", "ी"),
    ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"), ("ei", "ए", "े"), ("a", "अ", ""), ("i", "इ", "ि"),
    ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो"),
]


NASAL_CODA_BEFORE = ("k", "g", "c", "j", "t", "d", "p", "b", "s", "z", "f", "q", "x")


def roman_rules_to_devanagari(word: str) -> str:
    w = word.lower()
    out: list[str] = []
    i, n = 0, len(w)
    prev_cons = False
    while i < n:
        for rom, deva in ROMAN_CONS:
            if w.startswith(rom, i):
                after = w[i + len(rom):]
                nasal_coda = rom in ("n", "m") and not prev_cons and bool(out) and (
                    after.startswith(("p", "b")) if rom == "m" else after.startswith(NASAL_CODA_BEFORE))
                if nasal_coda:
                    out.append(ANUSVARA)
                    prev_cons = False
                else:
                    if prev_cons:
                        out.append(VIRAMA)
                    out.append(deva)
                    prev_cons = True
                i += len(rom)
                break
        else:
            for rom, indep, matra in ROMAN_VOWELS:
                if w.startswith(rom, i):
                    final = i + len(rom) == n
                    if prev_cons:
                        if rom == "a" and final:
                            matra = "ा"
                        elif rom == "i" and final:
                            matra = "ी"
                        elif rom == "u" and final:
                            matra = "ू"
                        out.append(matra)
                    else:
                        out.append(indep)
                    prev_cons = False
                    i += len(rom)
                    break
            else:
                out.append(w[i])
                prev_cons = False
                i += 1
    return unicodedata.normalize("NFC", "".join(out))


def _spelling_variants(deva: str) -> set[str]:
    """Ways Hindi writers spell the same English loan: plural -s as स (CMUdict has Z), any subset
    of non-initial viramas dropped (कॉमनवेल्थ, not कॉमन्वेल्थ), without nukta, ाई for ाइ."""
    bases = {deva}
    if deva.endswith("ज" + NUKTA):
        bases.add(deva[:-2] + "स")
    out: set[str] = set()
    for w in bases:
        initial: set[int] = set()  # viramas inside the word-initial cluster (स्ट, प्र) always stay
        i = 0
        while i < len(w) and w[i] in CONSONANTS:
            j = i + 1 + (i + 1 < len(w) and w[i + 1] == NUKTA)
            if j < len(w) and w[j] == VIRAMA:
                initial.add(j)
                i = j + 1
            else:
                break
        medial = [k for k, ch in enumerate(w) if ch == VIRAMA and k not in initial][:4]
        for mask in range(1 << len(medial)):
            drop = {medial[b] for b in range(len(medial)) if mask >> b & 1}
            v = "".join(ch for k, ch in enumerate(w) if k not in drop)
            out |= {v, v.replace(NUKTA, ""), v.replace("ाइ", "ाई")}
    return out


def _loose_roman_key(word: str) -> str:
    w = word.lower()
    w = w.replace("w", "v").replace("z", "j").replace("ph", "f").replace("q", "k")
    for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ey", "e")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


# ---------------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------------


def hindi_int(n: int) -> list[str]:
    if n < 100:
        return [lex.HINDI_0_99[n]]
    words: list[str] = []
    for value, name in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            q, n = divmod(n, value)
            words += hindi_int(q) + [name]
    if n:
        words.append(lex.HINDI_0_99[n])
    return words


def english_int(n: int) -> list[str]:
    if n < 20:
        return [lex.ENGLISH_ONES[n]]
    if n < 100:
        return [lex.ENGLISH_TENS[n // 10]] + ([lex.ENGLISH_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [lex.ENGLISH_ONES[n // 100], "hundred"] + (english_int(n % 100) if n % 100 else [])
    for value, name in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= value:
            q, r = divmod(n, value)
            return english_int(q) + [name] + (english_int(r) if r else [])
    raise AssertionError(n)


def _english_ordinal(words: list[str]) -> list[str]:
    last = words[-1]
    if last in lex.ENGLISH_ORDINALS:
        last = lex.ENGLISH_ORDINALS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return words[:-1] + [last]


# ---------------------------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------------------------


HINDI_NUMBER_WORDS = set(lex.HINDI_0_99) | {unicodedata.normalize("NFC", w) for w in ("सौ", "हज़ार", "लाख", "करोड़", "अरब")}


class HinglishTextProcessor:
    """Stateless after construction; building the lexicon indexes takes a few seconds."""

    def __init__(self, hindi_vocab_size: int = 30000, english_vocab_size: int = 120000):
        self.hindi_vocab_size = hindi_vocab_size
        self.english_vocab_size = english_vocab_size
        self._deva_from_english = self._build_reverse_english_index()
        self._roman_exact, self._roman_loose = self._build_roman_index()

    # ----- indexes ---------------------------------------------------------------------------
    def _build_reverse_english_index(self) -> dict[str, list[tuple[str, bool]]]:
        """Devanagari spelling (and its nukta-less variant) -> English words that produce it."""
        from wordfreq import top_n_list

        index: dict[str, list[str]] = {}
        cmu = _cmudict()
        for w in top_n_list("en", self.english_vocab_size):
            if not w.isalpha() or not w.isascii() or (w not in cmu and w not in lex.EN_TO_DEVA):
                continue
            deva = self.english_to_devanagari(w)
            primary = {deva, deva.replace(NUKTA, "")}
            for key in _spelling_variants(deva):
                index.setdefault(key, []).append((w, key in primary))
        return index

    def _build_roman_index(self) -> tuple[dict[str, str], dict[str, str]]:
        """Romanised Hindi word -> Devanagari, from the most frequent native Hindi words."""
        from wordfreq import top_n_list

        exact: dict[str, str] = {}
        loose: dict[str, str] = {}
        for w in top_n_list("hi", self.hindi_vocab_size):
            if not DEVA_LETTER.search(w) or LATIN_LETTER.search(w):
                continue
            w = unicodedata.normalize("NFC", w)
            if self.detect_loanword(w, min_len=2) is not None:
                continue  # English loans in Devanagari (सेट, मीटिंग) must not make "set" Hindi
            rom = self.romanize_hindi_word(w)
            exact.setdefault(rom, w)
            loose.setdefault(_loose_roman_key(rom), w)
        for deva, rom in lex.DEVA_TO_ROMAN.items():
            exact[rom] = deva
            loose.setdefault(_loose_roman_key(rom), deva)
        return exact, loose

    # ----- word-level converters ---------------------------------------------------------------
    def romanize_hindi_word(self, word: str) -> str:
        word = unicodedata.normalize("NFC", word)
        if word in lex.DEVA_TO_ROMAN:
            return lex.DEVA_TO_ROMAN[word]
        return _romanize_units(_parse_devanagari(word))

    def english_to_devanagari(self, word: str) -> str:
        if "-" in word.strip("-"):
            return "-".join(self.english_to_devanagari(p) for p in word.split("-") if p)
        w = word.lower().strip("'")
        if w in lex.EN_TO_DEVA:
            return lex.EN_TO_DEVA[w]
        if word.isupper() and 2 <= len(word) <= 5 and (len(word) <= 3 or w not in _cmudict()):
            return "".join(lex.LETTER_NAMES_DEVA[c] for c in w if c in lex.LETTER_NAMES_DEVA)
        phones = cmu_pronunciation(w) or cmu_pronunciation(w.replace("'", ""))
        if phones:
            return arpabet_to_devanagari(phones, spelling=w)
        if 2 <= len(w) <= 5 and not re.search(r"[aeiouy]", w):  # mcq, pdf: spelled out
            return "".join(lex.LETTER_NAMES_DEVA[c] for c in w if c in lex.LETTER_NAMES_DEVA)
        return roman_rules_to_devanagari(w)  # OOV: names and tech words read phonetically

    def roman_hindi_to_devanagari(self, word: str) -> str:
        w = word.lower()
        if w in self._roman_exact:
            return self._roman_exact[w]
        key = _loose_roman_key(w)
        if key in self._roman_loose:
            return self._roman_loose[key]
        return roman_rules_to_devanagari(w)

    def detect_loanword(self, deva: str, min_len: int = 4) -> str | None:
        """English spelling if a Devanagari word is an English loan, else None (conservative:
        a wrong positive writes a Hindi word as English, a miss just keeps Devanagari)."""
        if deva in lex.LOANWORD_WHITELIST:
            return lex.LOANWORD_WHITELIST[deva]
        if deva in lex.LOANWORD_BLOCKLIST:
            return None
        if deva in HINDI_NUMBER_WORDS:
            return None
        cands = self._deva_from_english.get(deva) if hasattr(self, "_deva_from_english") else None
        if not cands:
            return None
        # most frequent English word, preferring words whose own transliteration is this spelling
        # (false -> फॉल्स) over words that reach it only through a spelling variant (falls -> फॉल्ज़)
        en = max(cands, key=lambda c: _zipf(c[0], "en") + (0.3 if c[1] else 0.0))[0]
        hi_z, en_z = _zipf(deva, "hi"), _zipf(en, "en")
        if len(en) < min_len or hi_z >= 4.8:
            return None
        if _loose_roman_key(self.romanize_hindi_word(deva)) == _loose_roman_key(en):
            return None  # the "English" word is this Hindi word's own romanisation (राजकुमार, rajkumar)
        if len(en) >= 8:  # long words do not collide with native Hindi words by accident
            return en if en_z >= 2.0 else None
        # en_z >= 6: English function words (more, then, were, some) whose Devanagari spelling is
        # usually a native word (मोर peacock, देन gift, वर groom, सम equal)
        if en_z < 3.5 or en_z >= 6.0 or en_z - hi_z < 0.5:
            return None
        return en

    # ----- numbers ------------------------------------------------------------------------------
    def verbalize_number(self, token: str, lang: str = "hi", year: bool | None = None) -> list[str]:
        m = ORDINAL.match(token)
        if m:
            words = english_int(int(m.group(1)))
            words = _english_ordinal(words)
            return words if lang == "en" else [self.english_to_devanagari(w) for w in words]
        m = TIME.match(token)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            if lang == "en":
                return english_int(h) + (["o", "clock"] if mi == 0 else english_int(mi) if mi >= 10 else ["oh"] + english_int(mi))
            return hindi_int(h) + ["बजे"] if mi == 0 else hindi_int(h) + ["बजकर"] + hindi_int(mi) + ["मिनट"]
        m = NUMBER.match(token)
        if not m:
            return [token]
        digits = m.group("num").replace(",", "")
        int_part, _, frac = digits.partition(".")
        words: list[str]
        if len(int_part) >= 6 and "," not in m.group("num") and not frac:
            # identifiers (OTPs, phone numbers): digit by digit
            words = [(lex.ENGLISH_ONES if lang == "en" else lex.HINDI_0_99)[int(d)] for d in int_part]
        else:
            n = int(int_part)
            is_year = year if year is not None else (len(int_part) == 4 and 1100 <= n <= 1999 and not frac and not m.group("cur") and not m.group("pct"))
            if lang == "en":
                if is_year and 1100 <= n <= 2099 and n % 100 != 0 and not (2000 <= n <= 2009):
                    words = english_int(n // 100) + english_int(n % 100)
                else:
                    words = english_int(n)
            else:
                if is_year and 1100 <= n <= 1999:
                    words = hindi_int(n // 100) + ["सौ"] + (hindi_int(n % 100) if n % 100 else [])
                else:
                    words = hindi_int(n)
            if frac:
                point = "point" if lang == "en" else "दशमलव"
                table = lex.ENGLISH_ONES if lang == "en" else lex.HINDI_0_99
                words += [point] + [table[int(d)] for d in frac]
        if m.group("pct"):
            words.append("percent" if lang == "en" else "प्रतिशत")
        if m.group("cur"):
            words.append("rupees" if lang == "en" else "रुपये")
        return words

    # ----- sentence-level -------------------------------------------------------------------------
    def word_languages(self, text: str) -> list[str]:
        tokens = tokenize(clean_text(text))
        self._label(tokens)
        return [t.lang for t in tokens]

    SWITCH_PENALTY = 1.0  # zipf units (one decade of frequency) per language switch

    def _latin_scores(self, core: str) -> tuple[float, float]:
        """(Hindi, English) evidence for a Latin token, on the zipf scale of each language."""
        w = core.lower().strip("'")
        if not w.isascii() or (core.isupper() and len(core) >= 2):
            return 0.0, 9.0  # acronyms / non-ASCII Latin are English
        hi: float | None = None
        deva = self._roman_exact.get(w)
        if deva is not None:
            hi = _zipf(deva, "hi")
        else:
            deva = self._roman_loose.get(_loose_roman_key(w))
            if deva is not None:
                hi = _zipf(deva, "hi") - 0.5  # spelling variant: slightly weaker evidence
        en = _zipf(w, "en")
        if w in lex.EN_TO_DEVA:
            en = max(en, 4.0)
        if hi is None and en == 0.0:  # unknown to both: fall back on spelling shape
            hindi_shape = re.search(r"(aa|ee|oo|kh|gh|chh|bh|dh|jh|iye|aiy)", w) or \
                w.endswith(("ta", "na", "ga", "ke", "ki", "ne", "ko"))
            return (1.5, 1.0) if hindi_shape else (1.0, 1.5)
        return (hi if hi is not None else 1.0), (en if en > 0 else 1.0)

    def _viterbi_latin(self, tokens: list[Token]) -> None:
        """Label an all-Latin sentence word by word as Hindi or English (2-state Viterbi)."""
        idx = [i for i, t in enumerate(tokens) if t.kind == "latn"]
        if not idx:
            return
        states = ("hi", "en")
        scores = [self._latin_scores(tokens[i].core) for i in idx]
        best = [{s: (scores[0][k], [s]) for k, s in enumerate(states)}]
        for pos in range(1, len(idx)):
            row = {}
            for k, s in enumerate(states):
                cands = [(best[-1][p][0] + scores[pos][k] - (self.SWITCH_PENALTY if p != s else 0.0),
                          best[-1][p][1] + [s]) for p in states]
                row[s] = max(cands, key=lambda c: c[0])
            best.append(row)
        path = max(best[-1].values(), key=lambda c: c[0])[1]
        for i, lang in zip(idx, path):
            tokens[i].lang = lang

    def _label(self, tokens: list[Token]) -> str:
        """Fill token.lang / token.english; return the matrix language ("hi" or "en")."""
        has_deva = any(t.kind == "deva" for t in tokens)
        for t in tokens:
            if t.kind == "num":
                t.lang = "num"
            elif t.kind == "deva":
                t.english = self.detect_loanword(t.core)
                t.lang = "en" if t.english else "hi"
            elif t.kind == "latn":
                t.lang = "en"  # in mixed-script text, Latin words are English
            else:
                t.lang = "other"
        if not has_deva:
            self._viterbi_latin(tokens)
        n_hi = sum(t.lang == "hi" for t in tokens)
        n_en = sum(t.lang == "en" for t in tokens)
        return "hi" if n_hi >= n_en and n_hi > 0 else "en"

    def _render(self, t: Token, matrix: str, all_latin: bool, mixed_script: bool,
                spoken: bool) -> tuple[list[str], list[str], list[str]]:
        """(deva words, roman words, code-switched words) for one token, punctuation attached."""
        if t.kind == "num" or t.kind == "other":
            if t.kind == "other":
                words = [t.core]
                deva = roman = cs = words
            else:
                num_lang = matrix
                words = self.verbalize_number(t.core, lang=num_lang)
                if num_lang == "hi":
                    deva = words
                    roman = [self.romanize_hindi_word(w) for w in words]
                    cs = words
                else:
                    deva = [self.english_to_devanagari(w) for w in words]
                    roman = cs = words
                if not spoken:
                    roman = cs = [t.core]
            return self._punct(t, deva), self._punct(t, roman), self._punct(t, cs)
        core = t.core
        if t.kind == "deva":
            deva = core
            if t.english:
                roman = t.english
                cs = t.english if not mixed_script else core
            else:
                roman = self.romanize_hindi_word(core)
                cs = core
        else:  # Latin
            if t.lang == "hi":
                deva = self.roman_hindi_to_devanagari(core)
                roman = core
                cs = deva
            else:
                deva = self.english_to_devanagari(core)
                roman = core
                if mixed_script or matrix == "en":
                    cs = core
                else:  # embedded English in a romanised Hindi sentence: short loans assimilate
                    cs = core if len(core) >= 4 or core.isupper() else deva
        return self._punct(t, [deva]), self._punct(t, [roman]), self._punct(t, [cs])

    @staticmethod
    def _punct(t: Token, words: list[str]) -> list[str]:
        words = list(words)
        words[0] = t.lead + words[0]
        words[-1] = words[-1] + t.trail
        return words

    @staticmethod
    def _finish(words: list[str], latin_case: bool) -> str:
        text = " ".join(words)
        core = text.rstrip("\"')] ")
        closers = text[len(core):].strip()
        if core and core[-1] not in TERMINAL:
            core = core.rstrip(WEAK + " ") + "."
        text = core + closers
        if latin_case:
            text = re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
            text = re.sub(r"\bi\b", "I", text)
        return text

    def annotate(self, text: str) -> Annotation:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        if not tokens:
            raise ValueError(f"no words in transcript: {text!r}")
        matrix = self._label(tokens)
        has_deva = any(t.kind == "deva" for t in tokens)
        has_latn = any(t.kind == "latn" for t in tokens)
        mixed_script = has_deva and has_latn
        all_latin = has_latn and not has_deva
        display = ([], [], [])
        spoken = ([], [], [])
        spoken_words: list[tuple[str, str, str]] = []
        for t in tokens:
            for dst, sp in ((display, False), (spoken, True)):
                d, r, c = self._render(t, matrix, all_latin, mixed_script, sp)
                dst[0].extend(d)
                dst[1].extend(r)
                dst[2].extend(c)
                if sp:
                    assert len(d) == len(r) == len(c), (t, d, r, c)
                    spoken_words.extend(zip(d, r, c))
        finish = self._finish
        sd, sr, sc = finish(spoken[0], False), finish(spoken[1], True), finish(spoken[2], True)
        # keep the per-word view identical to the finished strings (terminal period, casing)
        spoken_words = list(zip(sd.split(), sr.split(), sc.split()))
        n_en = sum(t.lang == "en" for t in tokens)
        n_hi = sum(t.lang == "hi" for t in tokens)
        language = "hi-en" if n_en and n_hi else ("en" if n_en else "hi")
        return Annotation(
            text_original=text,
            text_devanagari=finish(display[0], False),
            text_roman=finish(display[1], True),
            text_code_switched=finish(display[2], True),
            spoken_devanagari=sd,
            spoken_roman=sr,
            spoken_code_switched=sc,
            spoken_words=spoken_words,
            language=language,
            n_english_words=n_en,
            n_hindi_words=n_hi,
        )

    # ----- convenience (inference front-end) --------------------------------------------------------
    def to_devanagari(self, text: str) -> str:
        return self.annotate(text).spoken_devanagari

    def to_roman(self, text: str) -> str:
        return self.annotate(text).spoken_roman

    def to_code_switched(self, text: str) -> str:
        return self.annotate(text).spoken_code_switched

    def normalize_for_tts(self, text: str, script: str = "code_switched") -> str:
        """Model input for inference: numbers verbalised, punctuation normalised, rendered in the
        requested script ("devanagari" | "roman" | "code_switched")."""
        ann = self.annotate(text)
        return {"devanagari": ann.spoken_devanagari, "roman": ann.spoken_roman,
                "code_switched": ann.spoken_code_switched}[script]


if __name__ == "__main__":
    import sys

    p = HinglishTextProcessor()
    for line in sys.argv[1:] or ["कल सुबह 7 बजे मेरी मीटिंग है, प्लीज़ रिमाइंडर सेट कर दो"]:
        a = p.annotate(line)
        print(f"deva : {a.text_devanagari}\nroman: {a.text_roman}\ncs   : {a.text_code_switched}\n"
              f"lang : {a.language}\nspoken roman: {a.spoken_roman}\n")
