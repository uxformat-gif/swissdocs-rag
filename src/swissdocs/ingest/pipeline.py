"""Build the sample corpus from Fedlex and optional private JSONL."""

import json
import logging
import re
from datetime import date
from pathlib import Path

from swissdocs.config import LANGUAGES, LAWS, LawSpec, Settings
from swissdocs.corpus import load_chunks, merge_corpora, save_chunks
from swissdocs.ingest.akn import parse_akn
from swissdocs.ingest.chunking import chunk_articles
from swissdocs.ingest.fedlex import (
    download_bytes,
    human_url,
    polite_client,
    resolve_manifestation,
)
from swissdocs.ingest.pages import assign_pages, pages_from_pdf
from swissdocs.models import Chunk

logger = logging.getLogger(__name__)


def _article_sort_key(chunk: Chunk) -> tuple[str, str, int, str, int]:
    match = re.match(r"(\d+)", chunk.article)
    number = int(match.group(1)) if match else 10**9
    tail = chunk.id.rsplit(":", 1)[-1]
    part = int(tail) if tail.isdigit() else 0
    return (chunk.law_sr, chunk.language, number, chunk.article, part)


def build_corpus(
    settings: Settings,
    *,
    today: date | None = None,
    languages: tuple[str, ...] = LANGUAGES,
    laws: tuple[LawSpec, ...] = LAWS,
) -> tuple[list[Chunk], dict[str, object]]:
    """Download the latest applicable FR/DE/EN consolidations and chunk them.

    Raw XML and PDF are cached under ``settings.raw_dir`` and are not part of
    the git tree. The returned chunks are what ``make download`` commits as the
    sample corpus.
    """
    raw_dir = Path(settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    chunks: list[Chunk] = []
    manifest_laws: list[dict[str, object]] = []

    with polite_client() as client:
        for law in laws:
            for language in languages:
                logger.info("Resolving %s %s", law.sr, language)
                try:
                    manifest = resolve_manifestation(law, language, client=client, today=today)
                except Exception as exc:
                    logger.warning("Skipping %s %s: %s", law.sr, language, exc)
                    manifest_laws.append(
                        {
                            "sr": law.sr,
                            "language": language,
                            "status": "missing",
                            "error": str(exc),
                        }
                    )
                    continue
                xml = download_bytes(client, manifest.xml_url)
                xml_path = raw_dir / f"{law.sr}-{language}-{manifest.version_date}.xml"
                xml_path.write_bytes(xml)
                articles = parse_akn(
                    xml,
                    language=language,
                    law_sr=law.sr,
                    law_abbr=law.abbr.get(language, law.sr),
                    law_title=law.title.get(language, ""),
                    eli=law.eli,
                    source_url=human_url(law.eli, language),
                    binding=language != "en",
                )
                mapped = 0
                if manifest.pdf_url:
                    pdf = download_bytes(client, manifest.pdf_url)
                    pdf_path = raw_dir / f"{law.sr}-{language}-{manifest.version_date}.pdf"
                    pdf_path.write_bytes(pdf)
                    mapped = assign_pages(pages_from_pdf(pdf), articles)
                law_chunks = chunk_articles(
                    articles,
                    max_chars=settings.chunk_chars,
                    overlap=settings.chunk_overlap,
                )
                chunks.extend(law_chunks)
                manifest_laws.append(
                    {
                        "sr": law.sr,
                        "language": language,
                        "status": "ok",
                        "version_date": manifest.version_date,
                        "binding": language != "en",
                        "xml_url": manifest.xml_url,
                        "pdf_url": manifest.pdf_url,
                        "source_url": human_url(law.eli, language),
                        "articles": len(articles),
                        "chunks": len(law_chunks),
                        "pages_mapped": mapped,
                    }
                )
                logger.info(
                    "Indexed %s %s version %s: %s articles, %s chunks, %s pages mapped",
                    law.sr,
                    language,
                    manifest.version_date,
                    len(articles),
                    len(law_chunks),
                    mapped,
                )

    chunks.sort(key=_article_sort_key)
    report = {
        "generated_on": today.isoformat(),
        "note": (
            "Official FR/DE texts are federal enactments (not copyright-protected, "
            "art. 5 CopA/URG). Fedlex currently has no downloadable English "
            "consolidation for these two acts, so English questions are answered "
            "from the official French and German texts. Insurer general terms "
            "are not included; they are copyrighted. Point CUSTOM_CORPUS_PATH "
            "at a private JSONL to add them locally."
        ),
        "laws": manifest_laws,
        "chunks": len(chunks),
    }
    return chunks, report


def write_corpus(settings: Settings, chunks: list[Chunk], report: dict[str, object]) -> None:
    destination = settings.corpus_file()
    save_chunks(destination, chunks)
    manifest_path = destination.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_active_corpus(settings: Settings) -> list[Chunk]:
    chunks = load_chunks(settings.corpus_file())
    custom = settings.custom_corpus_file()
    if custom is not None:
        chunks = merge_corpora(chunks, load_chunks(custom))
    return chunks
