#!/usr/bin/env python3
"""Download public Fedlex consolidations and write data/sample/chunks.jsonl."""

import logging

from swissdocs.config import get_settings
from swissdocs.ingest.pipeline import build_corpus, write_corpus


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    chunks, report = build_corpus(settings)
    write_corpus(settings, chunks, report)
    print(f"Wrote {len(chunks)} chunks to {settings.corpus_path}")


if __name__ == "__main__":
    main()
