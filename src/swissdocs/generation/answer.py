"""Turn retrieved passages into a cited answer, or an explicit abstention."""

import re

from swissdocs.models import Answer, Citation, Hit
from swissdocs.text import LANGUAGE_NAMES

_FLAG = re.compile(r"INSUFFICIENT_CONTEXT:\s*(true|false)", re.IGNORECASE)
_CITE = re.compile(r"\[(\d+)\]")

_TEMPLATES = {
    "fr": {
        "intro": "D'après les dispositions indexées :",
        "insufficient": (
            "Les textes indexés ne permettent pas de répondre de façon fiable. "
            "La question sort du corpus, ou les passages retrouvés ne contiennent "
            "pas l'élément demandé. Les sources les plus proches sont citées "
            "pour vérification sur Fedlex."
        ),
    },
    "de": {
        "intro": "Nach den indexierten Bestimmungen:",
        "insufficient": (
            "Die indexierten Texte erlauben keine verlässliche Antwort. "
            "Die Frage liegt ausserhalb des Korpus, oder die gefundenen Stellen "
            "enthalten das Gefragte nicht. Die nächsten Quellen sind zur "
            "Prüfung auf Fedlex aufgeführt."
        ),
    },
    "en": {
        "intro": (
            "Fedlex does not publish an English consolidation of these acts. "
            "The closest official French or German provisions are:"
        ),
        "insufficient": (
            "The indexed texts do not support a reliable answer. The question "
            "is outside the corpus, or the retrieved passages do not contain "
            "the fact you asked for. The closest official sources are listed "
            "so you can check them on Fedlex."
        ),
    },
}


def context_is_insufficient(hits: list[Hit], min_dense_score: float) -> bool:
    """Abstain only when both the lexical and the dense signals are weak.

    A paraphrase can have BM25 = 0 and still be a good dense hit. An off-topic
    question tends to miss on both. ``min_dense_score <= 0`` disables the gate.
    """
    if not hits:
        return True
    if min_dense_score <= 0:
        return False
    best_dense = max((hit.dense_score or 0.0) for hit in hits)
    best_bm25 = max((hit.bm25_score or 0.0) for hit in hits)
    return best_dense < min_dense_score and best_bm25 <= 0.0


def to_citation(number: int, hit: Hit) -> Citation:
    chunk = hit.chunk
    snippet = chunk.text.strip()
    if len(snippet) > 360:
        snippet = snippet[:360].rstrip() + "…"
    return Citation(
        n=number,
        chunk_id=chunk.id,
        citation=chunk.label(),
        law_sr=chunk.law_sr,
        law_abbr=chunk.law_abbr,
        article=chunk.article,
        article_title=chunk.article_title,
        language=chunk.language,
        page=chunk.page,
        version_date=chunk.version_date,
        binding=chunk.binding,
        source_url=chunk.source_url,
        snippet=snippet,
        heading_path=chunk.heading_path,
    )


def build_context(hits: list[Hit], *, max_chars: int = 12000) -> str:
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        block = f"[{index}] {hit.chunk.label()}\n{hit.chunk.text.strip()}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def system_prompt(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, "French")
    return (
        "You are SwissDocs, a careful assistant over Swiss federal insurance "
        "legislation (LAMal/KVG and LCA/VVG) published on Fedlex. "
        f"Answer in {name}. Use only the numbered context. "
        "Cite every factual statement with the context number, like [1]. "
        "Do not invent article numbers, amounts, deadlines or conditions. "
        "If the context does not contain the answer, say so and do not guess. "
        "This is not legal advice. "
        "Start your reply with exactly one of these lines:\n"
        "INSUFFICIENT_CONTEXT: true\n"
        "INSUFFICIENT_CONTEXT: false"
    )


def user_prompt(question: str, context: str) -> str:
    return f"Question:\n{question}\n\nContext:\n{context}"


def parse_model_output(text: str, hits: list[Hit]) -> tuple[str, bool, list[Citation]]:
    flag = _FLAG.search(text)
    insufficient = bool(flag and flag.group(1).lower() == "true")
    answer = _FLAG.sub("", text).strip()
    cited_numbers = []
    for raw in _CITE.findall(answer):
        number = int(raw)
        if 1 <= number <= len(hits) and number not in cited_numbers:
            cited_numbers.append(number)
    if not cited_numbers and hits and not insufficient:
        cited_numbers = list(range(1, min(len(hits), 3) + 1))
        answer = answer.rstrip() + "\n\n" + " ".join(f"[{n}]" for n in cited_numbers)
    citations = [to_citation(number, hits[number - 1]) for number in cited_numbers]
    if insufficient and not citations:
        citations = [to_citation(i, hit) for i, hit in enumerate(hits[:3], start=1)]
    return answer, insufficient, citations


def extractive_answer(
    *,
    question: str,
    language: str,
    hits: list[Hit],
    insufficient: bool,
    backend: str,
    disclaimer: str,
) -> Answer:
    language = language if language in _TEMPLATES else "en"
    template = _TEMPLATES[language]
    shown: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        key = hit.chunk.article_key()
        if key in seen:
            continue
        seen.add(key)
        shown.append(hit)
        if len(shown) == 3:
            break
    if insufficient:
        body = template["insufficient"]
    else:
        quotes = []
        for index, hit in enumerate(shown, start=1):
            snippet = hit.chunk.text.strip()
            if len(snippet) > 700:
                snippet = snippet[:700].rstrip() + "…"
            quotes.append(f"[{index}] {snippet}")
        body = template["intro"] + "\n\n" + "\n\n".join(quotes)
    citations = [to_citation(i, hit) for i, hit in enumerate(shown if shown else hits, start=1)]
    return Answer(
        question=question,
        language=language,
        answer=body,
        insufficient_context=insufficient,
        backend=backend,
        citations=citations,
        disclaimer=disclaimer,
    )
