import json
from pathlib import Path

from swissdocs.corpus import load_chunks
from swissdocs.evaluate import load_gold

ROOT = Path(__file__).resolve().parents[1]


def test_sample_corpus_and_gold_articles_line_up() -> None:
    chunks = load_chunks(ROOT / "data" / "sample" / "chunks.jsonl")
    keys = {chunk.article_key() for chunk in chunks}
    assert len(chunks) > 100
    languages = {chunk.language for chunk in chunks}
    assert languages == {"fr", "de"}
    laws = {chunk.law_sr for chunk in chunks}
    assert laws == {"832.10", "221.229.1"}
    items = load_gold(ROOT / "data" / "eval" / "qa.jsonl")
    answerable = [item for item in items if not item.expect_insufficient]
    assert len(answerable) >= 24
    for item in answerable:
        assert item.language in {"fr", "de", "en"}
        assert item.gold_articles
        missing = [key for key in item.gold_articles if key not in keys]
        assert not missing, f"{item.id} missing {missing}"
    manifest = json.loads((ROOT / "data" / "sample" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunks"] == len(chunks)
