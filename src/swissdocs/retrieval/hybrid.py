"""Reciprocal rank fusion of BM25 and dense retrieval."""

from swissdocs.models import Chunk, Hit
from swissdocs.retrieval.bm25 import BM25Index
from swissdocs.retrieval.embeddings import Embedder
from swissdocs.retrieval.store import VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        retrieve_k: int = 20,
    ) -> None:
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.retrieve_k = retrieve_k

    def dense(
        self,
        query: str,
        *,
        language: str | None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, language=language, limit=limit or self.retrieve_k)

    def lexical(
        self,
        query: str,
        *,
        language: str | None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        return self.bm25.search(query, language=language, limit=limit or self.retrieve_k)

    def search(
        self,
        query: str,
        *,
        language: str | None,
        top_k: int,
        cross_lingual: bool = False,
    ) -> list[Hit]:
        if cross_lingual:
            languages: list[str | None] = list(self.bm25.languages)
            dense_hits = self.dense(query, language=None)
            rankings = [dense_hits]
            bm25_scores: dict[str, float] = {}
            for lang in languages:
                lexical_hits = self.lexical(query, language=lang)
                rankings.append(lexical_hits)
                for doc_id, score in lexical_hits:
                    bm25_scores[doc_id] = max(bm25_scores.get(doc_id, 0.0), score)
        else:
            dense_hits = self.dense(query, language=language)
            lexical_hits = self.lexical(query, language=language)
            rankings = [dense_hits, lexical_hits]
            bm25_scores = dict(lexical_hits)

        dense_scores = dict(dense_hits)
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        # RRF scores tie often. Break ties with the raw signals, dense first,
        # so a cross-lingual paraphrase outranks an alphabetical accident.
        ordered = sorted(
            fused.items(),
            key=lambda item: (
                -item[1],
                -(dense_scores.get(item[0]) or 0.0),
                -(bm25_scores.get(item[0]) or 0.0),
                item[0],
            ),
        )[:top_k]
        hits: list[Hit] = []
        for doc_id, score in ordered:
            chunk = self.by_id.get(doc_id)
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=score,
                    dense_score=dense_scores.get(doc_id),
                    bm25_score=bm25_scores.get(doc_id),
                )
            )
        return hits
