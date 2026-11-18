"""Index the corpus and answer a question."""

import logging
import time

import httpx

from swissdocs.config import Settings
from swissdocs.generation.answer import (
    build_context,
    context_is_insufficient,
    extractive_answer,
    parse_model_output,
    system_prompt,
    user_prompt,
)
from swissdocs.generation.llm import LLM, LLMError, build_llm
from swissdocs.ingest.pipeline import load_active_corpus
from swissdocs.models import Answer, Chunk
from swissdocs.retrieval.bm25 import BM25Index
from swissdocs.retrieval.embeddings import Embedder, build_embedder
from swissdocs.retrieval.hybrid import HybridRetriever
from swissdocs.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(
            settings.embedding_backend, settings.embedding_model
        )
        self.llm = llm if llm is not None or settings.llm_backend == "extractive" else None
        if llm is None and settings.llm_backend != "extractive":
            self.llm = build_llm(
                settings.llm_backend,
                ollama_base_url=settings.ollama_base_url,
                ollama_model=settings.ollama_model,
                openai_base_url=settings.openai_base_url,
                openai_api_key=settings.openai_api_key,
                openai_model=settings.openai_model,
            )
        self.store = store or VectorStore.open(
            url=settings.qdrant_url,
            path=settings.qdrant_path,
            collection=settings.qdrant_collection,
        )
        self.chunks: list[Chunk] = []
        self.retriever: HybridRetriever | None = None

    def _wait_for_qdrant(self) -> None:
        url = self.settings.qdrant_url
        if self.settings.qdrant_path or url in {":memory:", "memory"}:
            return
        deadline = time.time() + 90
        health = url.rstrip("/") + "/healthz"
        while time.time() < deadline:
            try:
                httpx.get(health, timeout=2).raise_for_status()
                return
            except httpx.HTTPError:
                time.sleep(1)
        raise RuntimeError(f"Qdrant is not reachable at {url}")

    def prepare(self) -> None:
        self._wait_for_qdrant()
        self.chunks = load_active_corpus(self.settings)
        bm25 = BM25Index()
        bm25.build(self.chunks)
        self.store.ensure(self.chunks, self.embedder)
        self.retriever = HybridRetriever(
            self.chunks,
            self.store,
            bm25,
            self.embedder,
            rrf_k=self.settings.rrf_k,
            retrieve_k=self.settings.retrieve_k,
        )
        logger.info("Ready: %s chunks, backend %s", len(self.chunks), self.backend_name)

    @property
    def backend_name(self) -> str:
        if self.llm is None:
            return "extractive"
        return self.llm.name

    def health(self) -> dict[str, object]:
        languages = sorted({chunk.language for chunk in self.chunks})
        laws = sorted({chunk.law_sr for chunk in self.chunks})
        return {
            "status": "ok" if self.retriever else "starting",
            "chunks": len(self.chunks),
            "languages": languages,
            "laws": laws,
            "embedding_model": self.embedder.model_name,
            "llm_backend": self.backend_name,
            "collection": self.settings.qdrant_collection,
        }

    def ask(
        self,
        question: str,
        *,
        language: str | None = None,
        cross_lingual: bool = False,
    ) -> Answer:
        if self.retriever is None:
            raise RuntimeError("Index is not ready")
        from swissdocs.text import detect_language

        resolved = language or detect_language(question)
        if resolved not in {"fr", "de", "en"}:
            resolved = detect_language(question)
        indexed = {chunk.language for chunk in self.chunks}
        # English questions search the official FR/DE texts. Fedlex does not
        # currently publish a downloadable English consolidation for this corpus.
        cross = cross_lingual or resolved not in indexed
        hits = self.retriever.search(
            question,
            language=None if cross else resolved,
            top_k=self.settings.top_k,
            cross_lingual=cross,
        )
        insufficient = context_is_insufficient(hits, self.settings.min_dense_score)
        logger.info(
            "ask lang=%s hits=%s insufficient=%s backend=%s",
            resolved,
            len(hits),
            insufficient,
            self.backend_name,
        )
        if insufficient or self.llm is None:
            backend = self.backend_name if not insufficient else f"{self.backend_name}+abstain"
            if self.llm is None:
                backend = "extractive" if not insufficient else "extractive+abstain"
            return extractive_answer(
                question=question,
                language=resolved,
                hits=hits,
                insufficient=insufficient,
                backend=backend,
                disclaimer=self.settings.disclaimer,
            )
        try:
            raw = self.llm.complete(
                system_prompt(resolved),
                user_prompt(question, build_context(hits)),
            )
        except LLMError:
            if not self.settings.llm_fallback_extractive:
                raise
            logger.warning("LLM failed; falling back to extractive answers")
            return extractive_answer(
                question=question,
                language=resolved,
                hits=hits,
                insufficient=insufficient,
                backend="extractive-fallback",
                disclaimer=self.settings.disclaimer,
            )
        answer, model_flag, citations = parse_model_output(raw, hits)
        return Answer(
            question=question,
            language=resolved,
            answer=answer,
            insufficient_context=insufficient or model_flag,
            backend=self.backend_name,
            citations=citations,
            disclaimer=self.settings.disclaimer,
        )
