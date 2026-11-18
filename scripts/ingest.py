#!/usr/bin/env python3
"""Embed the corpus into Qdrant and build the in-memory BM25 index."""

import logging

from swissdocs.config import get_settings
from swissdocs.service import RagService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    service = RagService(get_settings())
    service.prepare()
    print(service.health())


if __name__ == "__main__":
    main()
