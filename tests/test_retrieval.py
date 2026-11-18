from datetime import date

from swissdocs.evaluate import hit_at_k, mean_reciprocal_rank
from swissdocs.generation.answer import context_is_insufficient, parse_model_output
from swissdocs.ingest.fedlex import choose_files
from swissdocs.models import Chunk, Hit
from swissdocs.retrieval.hybrid import reciprocal_rank_fusion
from swissdocs.text import detect_language


def _chunk(chunk_id: str, text: str, language: str = "fr", article: str = "1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        law_sr="832.10",
        law_abbr="LAMal",
        law_title="LAMal",
        language=language,
        article=article,
        article_title="Titre",
        page=4,
        version_date="2026-07-01",
        binding=True,
        source_url="https://www.fedlex.admin.ch/eli/cc/1995/1328_1328_1328/fr",
        eli="cc/1995/1328_1328_1328",
    )


def test_language_detection() -> None:
    assert detect_language("Quelles prestations l'assurance prend-elle en charge ?") == "fr"
    assert detect_language("Wie beteiligen sich die Versicherten an den Kosten?") == "de"
    assert detect_language("What does compulsory health insurance cover?") == "en"


def test_future_consolidation_is_ignored() -> None:
    rows = [
        {"date": "2032-01-01", "format": "xml", "url": "https://example.test/future.xml"},
        {"date": "2026-07-01", "format": "xml", "url": "https://example.test/current.xml"},
        {"date": "2026-07-01", "format": "pdf-a", "url": "https://example.test/current.pdf"},
        {"date": "2024-01-01", "format": "xml", "url": "https://example.test/old.xml"},
    ]
    chosen = choose_files(rows, date(2026, 11, 18))
    assert chosen.version_date == "2026-07-01"
    assert chosen.xml_url.endswith("current.xml")
    assert chosen.pdf_url is not None and chosen.pdf_url.endswith("current.pdf")


def test_rrf_prefers_a_document_found_by_both_retrievers() -> None:
    dense = [("a", 0.9), ("b", 0.8)]
    lexical = [("b", 4.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ranking = sorted(fused, key=fused.get, reverse=True)
    assert ranking[0] == "b"


def test_metrics() -> None:
    ranks = [1, None, 2]
    assert hit_at_k(ranks, 1) == 1 / 3
    assert hit_at_k(ranks, 5) == 2 / 3
    assert mean_reciprocal_rank(ranks) == (1 + 0 + 0.5) / 3


def test_model_output_parser_flags_insufficient_context() -> None:
    hits = [
        Hit(chunk=_chunk("c1", "Texte de l'article sur la franchise."), score=0.2, dense_score=0.9)
    ]
    answer, insufficient, citations = parse_model_output(
        "INSUFFICIENT_CONTEXT: true\nLe corpus ne donne pas le montant de la prime.",
        hits,
    )
    assert insufficient is True
    assert "prime" in answer
    assert citations[0].article == "1"
    assert "p. 4" in citations[0].citation


def test_abstention_gate_requires_both_signals_to_be_weak() -> None:
    strong_dense = Hit(chunk=_chunk("c1", "franchise"), score=0.1, dense_score=0.86, bm25_score=0)
    weak = Hit(chunk=_chunk("c2", "autre"), score=0.01, dense_score=0.4, bm25_score=0)
    lexical = Hit(chunk=_chunk("c3", "prime"), score=0.1, dense_score=0.4, bm25_score=2.5)
    assert context_is_insufficient([strong_dense], 0.78) is False
    assert context_is_insufficient([weak], 0.78) is True
    assert context_is_insufficient([lexical], 0.78) is False
    assert context_is_insufficient([], 0.78) is True
