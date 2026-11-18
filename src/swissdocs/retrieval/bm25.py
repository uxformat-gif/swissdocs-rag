"""In-memory BM25 over the corpus, one index per language."""

from collections import defaultdict

from rank_bm25 import BM25Okapi

from swissdocs.models import Chunk
from swissdocs.text import tokenize


class BM25Index:
    def __init__(self) -> None:
        self._indexes: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}

    def build(self, chunks: list[Chunk]) -> None:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.language].append(chunk)
        self._indexes = {}
        for language, items in grouped.items():
            corpus = []
            for item in items:
                tokens = tokenize(f"{item.article_title} {item.text} {item.law_abbr}")
                corpus.append(tokens or ["empty"])
            self._indexes[language] = (BM25Okapi(corpus), items)

    @property
    def languages(self) -> list[str]:
        return sorted(self._indexes)

    def search(
        self,
        query: str,
        *,
        language: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        if language is None:
            languages = self.languages
        else:
            languages = [language]
        scored: list[tuple[str, float]] = []
        tokens = tokenize(query) or ["empty"]
        for lang in languages:
            entry = self._indexes.get(lang)
            if entry is None:
                continue
            model, items = entry
            for item, score in zip(items, model.get_scores(tokens), strict=True):
                scored.append((item.id, float(score)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]
