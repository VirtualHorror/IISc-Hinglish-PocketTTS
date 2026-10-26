"""Curated lexicons for the Hinglish normalizer (hinglish_text_processor.py).

Rules handle the regular cases; these tables pin the frequent words where casual Hinglish
spelling or standard Hindi spelling of an English loan departs from what the rules produce.
"""

# Devanagari Hindi word -> casual Roman Hinglish spelling (overrides the rule-based romanizer).
DEVA_TO_ROMAN = {
    "है": "hai", "हैं": "hain", "में": "mein", "मैं": "main", "नहीं": "nahin", "नही": "nahi",
    "हम": "hum", "यह": "yeh", "वह": "woh", "ये": "ye", "वो": "wo", "हूँ": "hoon", "हूं": "hoon",
    "क्यों": "kyon", "क्यूँ": "kyun", "क्यूं": "kyun", "बहन": "behen", "शहर": "shehar",
    "वाला": "wala", "वाली": "wali", "वाले": "wale", "वहाँ": "wahan", "वहां": "wahan",
    "यहाँ": "yahan", "यहां": "yahan", "कहाँ": "kahan", "कहां": "kahan", "वही": "wahi",
    "वैसे": "waise", "वैसा": "waisa", "वैसी": "waisi", "वजह": "wajah", "वक़्त": "waqt",
    "वक्त": "waqt", "वापस": "wapas", "गई": "gayi", "गए": "gaye", "गये": "gaye", "नई": "nayi",
    "नए": "naye", "नये": "naye", "कई": "kayi", "लिए": "liye", "लिये": "liye", "चाहिए": "chahiye",
    "चाहिये": "chahiye", "कीजिए": "kijiye", "दीजिए": "dijiye", "जाइए": "jaiye", "आइए": "aaiye",
    "और": "aur", "एक": "ek", "अच्छा": "accha", "अच्छी": "acchi", "अच्छे": "acche", "ठीक": "theek",
    "बहुत": "bahut", "मुझे": "mujhe", "तुम्हें": "tumhen", "उन्हें": "unhen", "इन्हें": "inhen",
    "हमें": "hamen", "कोई": "koi", "कुछ": "kuch", "सबसे": "sabse", "मुँह": "munh", "पूछ": "poochh",
    "हुआ": "hua", "हुई": "hui", "हुए": "hue", "दुनिया": "duniya", "ज़्यादा": "zyada",
    "ज्यादा": "zyada", "रहे": "rahe", "रहा": "raha", "रही": "rahi", "कहा": "kaha", "पहला": "pehla",
    "पहली": "pehli", "आज": "aaj", "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke",
}

# English word -> standard Hindi (Devanagari) spelling, where CMUdict's US pronunciation would
# give an unusual spelling (function words, wh-words, -ment, Indian pronunciations).
EN_TO_DEVA = {
    "the": "द", "a": "अ", "an": "ऐन", "of": "ऑफ़", "and": "एंड", "to": "टू", "is": "इज़",
    "are": "आर", "was": "वॉज़", "were": "वर", "be": "बी", "been": "बीन", "it": "इट", "in": "इन",
    "on": "ऑन", "at": "ऐट", "for": "फ़ॉर", "from": "फ़्रॉम", "with": "विद", "by": "बाय", "or": "ऑर",
    "not": "नॉट", "no": "नो", "yes": "यस", "this": "दिस", "that": "दैट", "these": "दीज़",
    "those": "दोज़", "what": "व्हाट", "which": "व्हिच", "who": "हू", "whom": "हूम", "whose": "हूज़",
    "where": "व्हेयर", "when": "व्हेन", "why": "व्हाय", "how": "हाउ", "all": "ऑल", "any": "एनी",
    "can": "कैन", "could": "कुड", "will": "विल", "would": "वुड", "should": "शुड", "shall": "शैल",
    "may": "मे", "might": "माइट", "must": "मस्ट", "do": "डू", "does": "डज़", "did": "डिड",
    "have": "हैव", "has": "हैज़", "had": "हैड", "i": "आई", "you": "यू", "he": "ही", "she": "शी",
    "we": "वी", "they": "दे", "me": "मी", "my": "माय", "your": "योर", "our": "आवर", "their": "देयर",
    "his": "हिज़", "her": "हर", "them": "देम", "us": "अस", "there": "देयर", "here": "हियर",
    "very": "वेरी", "also": "ऑलसो", "only": "ओनली", "just": "जस्ट", "okay": "ओके", "ok": "ओके",
    "hello": "हेलो", "hi": "हाय", "bye": "बाय", "thanks": "थैंक्स", "thank": "थैंक", "sorry": "सॉरी",
    "please": "प्लीज़", "office": "ऑफ़िस", "tutorial": "ट्यूटोरियल", "tutorials": "ट्यूटोरियल्स",
    "version": "वर्ज़न", "menu": "मेन्यू", "window": "विंडो", "file": "फ़ाइल", "files": "फ़ाइल्स",
    "new": "न्यू", "news": "न्यूज़", "mobile": "मोबाइल", "internet": "इंटरनेट", "online": "ऑनलाइन",
    "software": "सॉफ़्टवेयर", "hardware": "हार्डवेयर", "video": "वीडियो", "audio": "ऑडियो",
    "linux": "लिनक्स", "ubuntu": "उबंटू", "google": "गूगल", "email": "ईमेल", "police": "पुलिस",
    "college": "कॉलेज", "government": "गवर्नमेंट", "management": "मैनेजमेंट", "payment": "पेमेंट",
    "statement": "स्टेटमेंट", "department": "डिपार्टमेंट", "apartment": "अपार्टमेंट",
    "moment": "मोमेंट", "comment": "कमेंट", "document": "डॉक्यूमेंट", "student": "स्टूडेंट",
    "students": "स्टूडेंट्स", "president": "प्रेसिडेंट", "doctor": "डॉक्टर", "data": "डेटा",
    "status": "स्टेटस", "due": "ड्यू", "date": "डेट", "number": "नंबर", "user": "यूज़र",
    "use": "यूज़", "used": "यूज़्ड", "using": "यूज़िंग", "music": "म्यूज़िक", "mister": "मिस्टर",
    "mr": "मिस्टर", "mrs": "मिसेज़", "dr": "डॉक्टर", "india": "इंडिया", "indian": "इंडियन",
    "delhi": "दिल्ली", "mumbai": "मुंबई", "bangalore": "बैंगलोर", "bengaluru": "बेंगलुरु",
    "minute": "मिनट", "minutes": "मिनट्स", "second": "सेकंड", "seconds": "सेकंड्स",
    "hour": "आवर", "hours": "आवर्स", "percent": "परसेंट", "rupees": "रुपीज़", "rupee": "रुपी",
    # heteronyms: CMUdict lists the less common reading first for these
    "wind": "विंड", "live": "लिव", "read": "रीड", "lead": "लीड", "close": "क्लोज़",
    "record": "रिकॉर्ड", "present": "प्रेज़ेंट", "project": "प्रोजेक्ट", "content": "कंटेंट",
    "object": "ऑब्जेक्ट", "minute": "मिनट", "wound": "वूंड", "tear": "टियर", "bass": "बेस",
}

# Latin abbreviations that are read out as a full word (Spoken Tutorial narrators say "control").
SPOKEN_ABBREVIATIONS = {"ctrl": "control"}

# Letter names for spelling out acronyms (OTP, EMI, PDF ...).
LETTER_NAMES_DEVA = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ़", "g": "जी", "h": "एच",
    "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी",
    "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}

# Devanagari loanwords that should always map to their English spelling (frequent enough in
# Hindi that the frequency-margin rule would otherwise keep them in Devanagari).
LOANWORD_WHITELIST = {
    "पुलिस": "police", "स्कूल": "school", "स्टेशन": "station", "डॉक्टर": "doctor", "टीम": "team",
    "ऑफ़िस": "office", "ऑफिस": "office", "फ़ोन": "phone", "फोन": "phone", "मोबाइल": "mobile",
    "कंप्यूटर": "computer", "कम्प्यूटर": "computer", "इंटरनेट": "internet", "बैंक": "bank",
    "होटल": "hotel", "ट्रेन": "train", "टिकट": "ticket", "कंपनी": "company", "मैनेजर": "manager",
    "प्रोग्राम": "program", "प्रोजेक्ट": "project", "सिस्टम": "system", "मीटिंग": "meeting",
    "प्लीज़": "please", "प्लीज": "please", "रिमाइंडर": "reminder", "पेमेंट": "payment",
    "ट्यूटोरियल": "tutorial", "कॉलेज": "college", "क्रिकेट": "cricket", "फ़िल्म": "film",
    "फिल्म": "film", "वीडियो": "video", "ईमेल": "email", "मैसेज": "message", "मेसेज": "message",
    "लिनक्स": "linux", "सॉफ़्टवेयर": "software", "सॉफ्टवेयर": "software", "विंडो": "window",
    "फ़ाइल": "file", "फाइल": "file", "फ़ोल्डर": "folder", "फोल्डर": "folder", "क्लिक": "click",
    "ऑपरेटिंग": "operating", "ऑनलाइन": "online", "डेटा": "data", "रेडियो": "radio",
    "टेलीविज़न": "television", "टेलीविजन": "television", "ऑर्डर": "order", "एजेंट": "agent",
    "स्टेटमेंट": "statement", "प्रेजेंटेशन": "presentation", "स्लाइड": "slide",
    "डॉक्यूमेंट": "document", "फ़ॉन्ट": "font", "फॉन्ट": "font", "टेक्स्ट": "text",
    # homophones where frequency alone picks the wrong English spelling (audited on the corpus)
    "फेल": "fail", "पेन": "pen", "होल": "hole", "ट्रेंड": "trend", "लेटर": "letter",
    "गेट्स": "gates", "डेम": "dam", "वुड": "wood", "फिल्ड": "field", "टीन": "tin", "सीन": "scene",
    "बाइट": "byte", "बाइट्स": "bytes", "पर्ल": "perl", "कोट्स": "quotes",
    "फोरम": "forum", "फ़ोरम": "forum", "स्टेटस": "status", "हंगरी": "hungary", "मेसी": "messi",
}

# Native Hindi words that happen to spell a common English word in Devanagari; never convert.
LOANWORD_BLOCKLIST = {
    "चीज़", "चीज", "मील", "मेल", "माल", "पुल", "कम", "कल", "बस", "सब", "सुन", "बेल", "बीन",
    "सीधा", "रोल", "सर", "मन", "दम", "जब", "तब", "पर", "से", "रेल", "सारी", "बार", "बात",
    "कार", "पल", "दिल", "तेल", "मेरा", "नाम", "राम", "शाम", "जान", "दान", "पान", "बाल",
    # found by auditing every conversion on the source corpora
    "मेज़", "मेज", "मोर", "देन", "हाय", "मच", "वर", "सम", "टेक", "फेर", "हेर", "काज", "सेंट",
    "एट", "पेड", "मेड", "वेट", "केर", "मरी", "टेल", "सेज", "ओज", "शैल", "बेसन", "टेक्स", "एसे",
    "परिजन", "सेस", "पोस", "टीस", "चूस", "जैस",
}

# Hindi number words 0..99.
HINDI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

ENGLISH_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ENGLISH_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}

# Canonicalise everything to NFC (nukta letters decompose under NFC), matching processor input.
import unicodedata as _ud  # noqa: E402


def _nfc(s: str) -> str:
    return _ud.normalize("NFC", s)


DEVA_TO_ROMAN = {_nfc(k): v for k, v in DEVA_TO_ROMAN.items()}
EN_TO_DEVA = {k: _nfc(v) for k, v in EN_TO_DEVA.items()}
LETTER_NAMES_DEVA = {k: _nfc(v) for k, v in LETTER_NAMES_DEVA.items()}
LOANWORD_WHITELIST = {_nfc(k): v for k, v in LOANWORD_WHITELIST.items()}
LOANWORD_BLOCKLIST = {_nfc(k) for k in LOANWORD_BLOCKLIST}
HINDI_0_99 = [_nfc(w) for w in HINDI_0_99]
