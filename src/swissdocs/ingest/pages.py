"""Map an article to the PDF page that contains its operative text."""

import io

from pypdf import PdfReader

from swissdocs.models import ParsedArticle
from swissdocs.text import squash


def pages_from_pdf(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def assign_pages(pages: list[str], articles: list[ParsedArticle]) -> int:
    """Set ``article.page`` (1-based). Returns how many articles were mapped.

    Fedlex PDFs often draw the article number in a way ``pypdf`` does not
    extract, while cross-references ("l'art. 64") are plain text. Matching the
    opening words of the article avoids those citations.
    """
    normalised = [squash(page) for page in pages]
    mapped = 0
    for article in articles:
        words = squash(article.text).split(" ")
        if words and words[0].isdigit():
            words = words[1:]
        words = words[:18]
        found: int | None = None
        for length in range(len(words), 5, -1):
            phrase = " ".join(words[:length])
            if len(phrase) < 28:
                continue
            for index, page in enumerate(normalised, start=1):
                if phrase in page:
                    found = index
                    break
            if found is not None:
                break
        if found is not None:
            article.page = found
            mapped += 1
    return mapped
