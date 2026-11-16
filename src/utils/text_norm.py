"""Text normalization applied identically to references and hypotheses before scoring.

English (AMI, NOTSOFAR-1, CHiME-6, DiPCo):
    1. strip corpus annotation markup ([noise], [laughs], <ST/>, {...}, etc.)
    2. Whisper-style EnglishTextNormalizer as used by the CHiME-8 DASR challenge (chime_utils "chime8"):
       lower-case, remove punctuation, spell-standardize (British->American), normalize numbers/currency,
       remove filler words (um, uh, hmm, ...).
Mandarin (AliMeeting):
    traditional->simplified, full-width->half-width, lower-case latin, remove punctuation and all
    whitespace; scored per character (CER).
"""
import re
import unicodedata
from functools import lru_cache

# corpus markup: [noise] [laughs] [inaudible ...] <ST/> <FILL/> {laugh} (( )) etc.
_MARKUP = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}|\(\([^)]*\)\)")


def strip_markup(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKUP.sub(" ", text)).strip()


@lru_cache(maxsize=1)
def _english_normalizer():
    from chime_utils.text_norm import get_txt_norm

    return get_txt_norm("chime8")


def normalize_en(text: str) -> str:
    return re.sub(r"\s+", " ", _english_normalizer()(strip_markup(text))).strip()


@lru_cache(maxsize=1)
def _zhconv():
    import zhconv

    return zhconv


def normalize_zh(text: str) -> str:
    text = strip_markup(text)
    text = unicodedata.normalize("NFKC", text)  # full-width -> half-width
    text = _zhconv().convert(text, "zh-cn").lower()
    # drop punctuation / symbols (unicode categories P*, S*) and whitespace
    text = "".join(c for c in text if not unicodedata.category(c)[0] in "PSZ" and not c.isspace())
    return text


def normalize(text: str, lang: str) -> str:
    return normalize_zh(text) if lang == "zh" else normalize_en(text)


def tokenize(norm_text: str, lang: str) -> list:
    """Scoring units: characters for Mandarin (CER), whitespace words otherwise (WER)."""
    return list(norm_text) if lang == "zh" else norm_text.split()
