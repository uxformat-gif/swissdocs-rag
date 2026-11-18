"""Qdrant vector store. Local path, memory, or a server URL."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from swissdocs.corpus import fingerprint, passage_text
from swissdocs.models import Chunk
from swissdocs.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("8f3c1d2e-6a4b-5c7d-9e0f-1234567890ab")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._meta_id = _point_id("__meta__")

    @classmethod
    def open(cls, *, url: str, path: str, collection: str) -> "VectorStore":
        if path:
            client = QdrantClient(path=path)
        elif url in {":memory:", "memory"}:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=url, timeout=30)
        return cls(client, collection)

    def _chunk_filter(self, language: str | None) -> Filter:
        must = [FieldCondition(key="kind", match=MatchValue(value="chunk"))]
        if language:
            must.append(FieldCondition(key="language", match=MatchValue(value=language)))
        return Filter(must=must)

    def fingerprint(self) -> str | None:
        if not self.client.collection_exists(self.collection):
            return None
        found = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._meta_id],
            with_payload=True,
        )
        if not found or not found[0].payload:
            return None
        value = found[0].payload.get("fingerprint")
        return str(value) if value else None

    def ensure(self, chunks: list[Chunk], embedder: Embedder) -> None:
        digest = fingerprint(chunks)
        if self.fingerprint() == digest:
            logger.info("Vector index already matches corpus %s", digest)
            return
        logger.info(
            "Building vector index (%s chunks, %s, dim %s)",
            len(chunks),
            embedder.model_name,
            embedder.dim,
        )
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
        )
        batch = 32
        for start in range(0, len(chunks), batch):
            group = chunks[start : start + batch]
            vectors = embedder.embed_documents([passage_text(chunk) for chunk in group])
            points = [
                PointStruct(
                    id=_point_id(chunk.id),
                    vector=vector,
                    payload={"kind": "chunk", "chunk_id": chunk.id, "language": chunk.language},
                )
                for chunk, vector in zip(group, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info("Embedded %s/%s", min(start + batch, len(chunks)), len(chunks))
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._meta_id,
                    vector=[1.0] + [0.0] * (embedder.dim - 1),
                    payload={"kind": "meta", "fingerprint": digest, "model": embedder.model_name},
                )
            ],
        )

    def search(
        self,
        vector: list[float],
        *,
        language: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=self._chunk_filter(language),
            with_payload=True,
        )
        hits: list[tuple[str, float]] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                continue
            hits.append((str(chunk_id), float(point.score or 0.0)))
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        result = self.client.count(
            collection_name=self.collection,
            count_filter=self._chunk_filter(None),
            exact=True,
        )
        return int(result.count)
