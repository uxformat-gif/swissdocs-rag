"""Embedding backends.

``fastembed`` runs multilingual-e5 locally via ONNX. Queries and passages never
leave the machine. Tests use :class:`HashingEmbedder`, which needs no download.
"""

from collections.abc import Sequence
from typing import Protocol

from swissdocs.text import l2_normalize, stable_bucket, tokenize


class Embedder(Protocol):
    dim: int
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-words vectors for tests and offline wiring checks."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[stable_bucket(token, self.dim)] += 1.0
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FastEmbedEmbedder:
    """Local multilingual embeddings. E5 models apply query/passage prefixes."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self.dim = int(self._model.embedding_size)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.query_embed([text])).tolist()


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name)
    raise ValueError(f"Unknown embedding backend: {backend}")
