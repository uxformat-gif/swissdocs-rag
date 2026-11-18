"""Load and store the JSONL corpus."""

import hashlib
from pathlib import Path

from swissdocs.models import Chunk


def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run `make download` to build it from Fedlex."
        )
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        chunk = Chunk.model_validate_json(line)
        if chunk.id in seen:
            raise ValueError(f"Duplicate chunk id {chunk.id} at {path}:{line_no}")
        seen.add(chunk.id)
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Corpus {path} is empty")
    return chunks


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(chunk.model_dump_json() for chunk in chunks)
    path.write_text(payload + "\n", encoding="utf-8")


def merge_corpora(base: list[Chunk], extra: list[Chunk]) -> list[Chunk]:
    seen = {chunk.id for chunk in base}
    merged = list(base)
    for chunk in extra:
        if chunk.id in seen:
            raise ValueError(f"Custom corpus reuses id {chunk.id}")
        seen.add(chunk.id)
        merged.append(chunk)
    return merged


def fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.page).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def passage_text(chunk: Chunk) -> str:
    """Text actually embedded: statute and article heading plus the body."""
    heading = f"{chunk.law_abbr} — art. {chunk.article}"
    if chunk.article_title:
        heading = f"{heading} {chunk.article_title}"
    return f"{heading}\n{chunk.text}"
