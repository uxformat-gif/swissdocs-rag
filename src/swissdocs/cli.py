"""Command line: ``python -m swissdocs.cli ask "..."``."""

import argparse
import logging

from swissdocs.config import get_settings
from swissdocs.service import RagService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ask SwissDocs from the terminal")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Answer one question")
    ask.add_argument("question")
    ask.add_argument("--language", choices=["fr", "de", "en"])
    ask.add_argument("--cross-lingual", action="store_true")
    args = parser.parse_args()

    service = RagService(get_settings())
    service.prepare()
    result = service.ask(
        args.question,
        language=args.language,
        cross_lingual=args.cross_lingual,
    )
    print(result.answer)
    print()
    if result.insufficient_context:
        print("insufficient_context: true")
    print(f"backend: {result.backend}")
    for citation in result.citations:
        print(f"[{citation.n}] {citation.citation}")
        print(f"    {citation.source_url}")


if __name__ == "__main__":
    main()
