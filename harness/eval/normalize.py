"""Text normalization and scoring units, applied identically to references and hypotheses.

    en  CHiME-8 Whisper-style normalizer (src/utils/text_norm.py, same as the error-analysis phase)
    zh  traditional->simplified, NFKC, punctuation/whitespace removed; scored per character
    ar  N1 of the Arabic benchmark (Open Universal Arabic ASR Leaderboard rules), copied verbatim from
        ArabicASR_Benchmark/code/src/arabic_metrics.py so the harness and the benchmark agree

The normalization of a session follows the session's *reference* language, whatever language the system
detected: Latin or CJK output in an Arabic session stays as it is and counts as errors.
"""
import re
import unicodedata

from src.utils.text_norm import normalize_en, normalize_zh

# ---- Arabic N1 (verbatim from the benchmark) ----
_AR_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|#\s*غير[\s_-]*واضح")
_HARAKAT = re.compile("[\u064B-\u0652\u0653-\u065F\u0670\u0640\u06D6-\u06ED]")  # escaped: bidi-safe
_AR_MAP = str.maketrans({"پ": "ب", "ڤ": "ف", "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي",
                         **{chr(0x0660 + i): str(i) for i in range(10)},
                         **{chr(0x06F0 + i): str(i) for i in range(10)}})


def _ws(t):
    return re.sub(r"\s+", " ", t).strip()


def normalize_ar(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = _AR_TAGS.sub(" ", t).lower()
    t = _HARAKAT.sub("", t).translate(_AR_MAP).replace("ء", "")
    t = "".join(c for c in t if unicodedata.category(c)[0] not in "PS")
    t = re.sub(r"(^|\s)و\s+(?=\S)", r"\1و", t)
    return _ws(t)


LANGS = ("en", "zh", "ar")


def normalize(text: str, lang: str) -> str:
    if lang == "zh":
        return normalize_zh(text or "")
    if lang == "ar":
        return normalize_ar(text)
    if lang == "en":
        return normalize_en(text or "")
    raise ValueError(f"unknown language {lang!r}")


def words(norm: str, lang: str) -> list:
    """Primary scoring units: characters for Mandarin (CER), whitespace words otherwise (WER)."""
    return [c for c in norm if not c.isspace()] if lang == "zh" else norm.split()


def chars(norm: str) -> list:
    """Secondary units for Arabic CER (spaces dropped so the unit stream does not depend on word splitting)."""
    return [c for c in norm if not c.isspace()]


# ---- scripts (foreign-script output, e.g. Audar translating English speech into Arabic) ----
def script(token: str) -> str:
    for c in token:
        o = ord(c)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            return "arabic"
        if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF:
            return "cjk"
    return "latin" if any("a" <= c <= "z" for c in token.lower()) else "other"


# Scripts that never belong in a session of a given language. Latin inside Arabic or Chinese speech is ordinary
# code-switching (Mixat, ASCEND), so it is not "foreign".
FOREIGN = {"en": {"arabic", "cjk"}, "zh": {"arabic"}, "ar": {"cjk"}}
