"""Split long articles into overlapping passages that keep article metadata."""

import re

from swissdocs.models import Chunk, ParsedArticle


def _snap_forward(text: str, index: int) -> int:
    """Move an overlap start to the next word so a passage does not open mid-token."""
    if index <= 0:
        return 0
    space = text.find(" ", index)
    if space == -1 or space >= len(text) - 1:
        return index
    return space + 1


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = _snap_forward(text, max(0, end - overlap))
        if start <= (end - size):
            start = end
    return chunks


def split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    if not paragraphs:
        return []
    packed: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_windows(paragraph, max_chars, overlap))
            continue
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if buffer and len(candidate) > max_chars:
            packed.append(buffer)
            tail = buffer[-overlap:]
            if " " in tail:
                tail = tail.split(" ", 1)[1]
            buffer = f"{tail.strip()}\n{paragraph}".strip()
        else:
            buffer = candidate
    if buffer:
        packed.append(buffer)
    return packed


def chunk_articles(
    articles: list[ParsedArticle],
    *,
    max_chars: int = 1100,
    overlap: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        parts = split_text(article.text, max_chars=max_chars, overlap=overlap) or [article.text]
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"{article.law_sr}:{article.language}:{article.eId}:{index}",
                    text=part,
                    law_sr=article.law_sr,
                    law_abbr=article.law_abbr,
                    law_title=article.law_title,
                    language=article.language,
                    article=article.article,
                    article_title=article.article_title,
                    page=article.page,
                    version_date=article.version_date,
                    binding=article.binding,
                    source_url=article.source_url,
                    eli=article.eli,
                    heading_path=article.heading_path,
                )
            )
    return chunks
