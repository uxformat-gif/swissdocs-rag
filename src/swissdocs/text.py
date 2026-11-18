"""Tokenisation, language detection and whitespace cleanup."""

import hashlib
import math
import re

_TOKEN = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+")

# Closed-class words used both to detect the question language and to keep
# BM25 focused on content words. Detection lexicons are the language-specific
# subsets below; BM25 drops the union.
_FR = frozenset(
    "les des une dans pour est que qui par sur avec cette aux son ses pas plus "
    "dans une est sont été être ou où dont lui leur elle ils vous nous du au "
    "de la le un ce se ne en et si sa mon ton à l d qu n y on aux quelles quelle "
    "quels quel comment pourquoi quand".split()
)
_DE = frozenset(
    "die der das und ein eine ist nicht von mit für auf den dem des sich auch "
    "im in zu zum zur ein einer einem einen oder wie was welche welcher welches "
    "sind wird wurde dass als am an bei nach über unter vom vom".split()
)
_EN = frozenset(
    "the and for with that this are not from what does how which was were been "
    "have has had you your into over under about than then them they their its "
    "can may will would should could a an of to in on by or if as at is it be".split()
)

DETECT = {"fr": _FR, "de": _DE, "en": _EN}
STOPWORDS = _FR | _DE | _EN

LANGUAGE_NAMES = {"fr": "French", "de": "German", "en": "English"}


def clean_inline(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_block(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    kept: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and kept:
                kept.append("")
            blank = True
            continue
        kept.append(line)
        blank = False
    return "\n".join(kept).strip()


def tokenize(text: str, *, drop_stops: bool = True) -> list[str]:
    tokens = [tok.lower() for tok in _TOKEN.findall(text)]
    if drop_stops:
        tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return tokens


def detect_language(text: str) -> str:
    """Guess fr/de/en from function words. Defaults to French, the first official language."""
    words = [tok.lower() for tok in _TOKEN.findall(text)]
    if not words:
        return "fr"
    scores = {lang: sum(1 for word in words if word in vocab) for lang, vocab in DETECT.items()}
    best = max(scores, key=lambda lang: scores[lang])
    if scores[best] == 0:
        if re.search(r"[äöüÄÖÜß]", text):
            return "de"
        if re.search(r"[éèêëàâùûçôîï]", text, re.IGNORECASE):
            return "fr"
        return "en"
    return best


def squash(text: str) -> str:
    text = (
        text.replace("\u00ad", "")
        .replace("\xa0", " ")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("’", "'")
        .lower()
    )
    return re.sub(r"\s+", " ", text).strip()


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        zero = [0.0] * len(vector)
        zero[0] = 1.0
        return zero
    return [value / norm for value in vector]


def stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim
