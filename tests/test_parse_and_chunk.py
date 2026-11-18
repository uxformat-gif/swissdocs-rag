from pathlib import Path

from swissdocs.ingest.akn import parse_akn
from swissdocs.ingest.chunking import chunk_articles, split_text
from swissdocs.ingest.pages import assign_pages

FIXTURE = Path(__file__).parent / "fixtures" / "mini_akn.xml"


def test_parser_strips_notes_and_skips_repealed_articles() -> None:
    articles = parse_akn(
        FIXTURE.read_bytes(),
        language="fr",
        law_sr="832.10",
        law_abbr="LAMal",
        law_title="fallback",
        eli="cc/1995/1328_1328_1328",
        source_url="https://www.fedlex.admin.ch/eli/cc/1995/1328_1328_1328/fr",
        binding=True,
    )
    assert [article.article for article in articles] == ["1a", "64"]
    article = articles[1]
    assert article.article == "64"
    assert article.law_abbr == "LAMal"
    assert article.version_date == "2026-07-01"
    assert article.heading_path == ["Dispositions générales"]
    assert "franchise" in article.text
    assert "Nouvelle teneur" not in article.text
    assert "2022" not in article.text


def test_long_article_is_split_without_losing_the_edges() -> None:
    text = "\n".join(f"Paragraphe {index} " + ("mot " * 40) for index in range(12))
    parts = split_text(text, max_chars=500, overlap=80)
    assert len(parts) > 1
    assert "Paragraphe 0" in parts[0]
    assert "Paragraphe 11" in parts[-1]
    chunks = chunk_articles(
        parse_akn(
            FIXTURE.read_bytes(),
            language="fr",
            law_sr="832.10",
            law_abbr="LAMal",
            law_title="t",
            eli="cc/1995/1328_1328_1328",
            source_url="https://example.test",
            binding=True,
        ),
        max_chars=80,
        overlap=20,
    )
    assert len(chunks) >= 2
    assert all(chunk.article == "64" for chunk in chunks if "art_64" in chunk.id)
    assert any(chunk.id.endswith(":0") for chunk in chunks)


def test_page_mapping_uses_the_page_that_contains_the_article() -> None:
    articles = parse_akn(
        FIXTURE.read_bytes(),
        language="fr",
        law_sr="832.10",
        law_abbr="LAMal",
        law_title="t",
        eli="cc/1995/1328_1328_1328",
        source_url="https://example.test",
        binding=True,
    )
    pages = [
        "Table des matières Art. 64 .... 3",
        "Art. 12 Autre disposition sans lien avec la franchise.",
        (
            "Art. 64 Participation aux coûts. Les assurés participent aux coûts "
            "des prestations. franchise quote-part."
        ),
    ]
    mapped = assign_pages(pages, articles)
    article = next(item for item in articles if item.article == "64")
    assert mapped == 1
    assert article.page == 3
