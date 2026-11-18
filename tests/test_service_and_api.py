from fastapi.testclient import TestClient

from swissdocs.api import create_app
from swissdocs.config import Settings
from swissdocs.corpus import save_chunks
from swissdocs.generation.llm import LLM
from swissdocs.models import Chunk
from swissdocs.retrieval.store import VectorStore
from swissdocs.service import RagService


class ScriptedLLM(LLM):
    name = "scripted"

    def complete(self, system: str, user: str) -> str:
        assert "French" in system
        return "INSUFFICIENT_CONTEXT: false\nLes assurés participent aux coûts [1]."


def _chunks() -> list[Chunk]:
    common = dict(
        law_sr="832.10",
        law_abbr="LAMal",
        law_title="Loi fédérale sur l'assurance-maladie",
        language="fr",
        page=12,
        version_date="2026-07-01",
        binding=True,
        source_url="https://www.fedlex.admin.ch/eli/cc/1995/1328_1328_1328/fr",
        eli="cc/1995/1328_1328_1328",
    )
    return [
        Chunk(
            id="832.10:fr:art_64:0",
            text=(
                "Les assurés participent aux coûts des prestations dont ils bénéficient. "
                "La participation comprend la franchise et la quote-part."
            ),
            article="64",
            article_title="Participation aux coûts",
            **common,
        ),
        Chunk(
            id="832.10:fr:art_31:0",
            text=(
                "L'assurance obligatoire des soins prend en charge les coûts des soins "
                "dentaires lors d'une maladie grave du système de la mastication."
            ),
            article="31",
            article_title="Soins dentaires",
            **common,
        ),
    ]


def _service(tmp_path, llm: LLM | None = None, min_dense_score: float = 0.0) -> RagService:
    path = tmp_path / "chunks.jsonl"
    save_chunks(path, _chunks())
    settings = Settings(
        corpus_path=str(path),
        qdrant_url=":memory:",
        qdrant_path="",
        qdrant_collection="test-swissdocs",
        embedding_backend="hashing",
        llm_backend="extractive" if llm is None else "ollama",
        min_dense_score=min_dense_score,
        top_k=2,
        retrieve_k=5,
    )
    store = VectorStore.open(url=":memory:", path="", collection="test-swissdocs")
    service = RagService(settings, llm=llm, store=store)
    service.prepare()
    return service


def test_extractive_answer_cites_the_cost_sharing_article(tmp_path) -> None:
    service = _service(tmp_path)
    answer = service.ask(
        "Quelle est la participation des assurés, franchise et quote-part ?",
        language="fr",
    )
    assert answer.insufficient_context is False
    assert answer.citations
    assert any(citation.article == "64" for citation in answer.citations)
    assert "p. 12" in answer.citations[0].citation
    assert "Fedlex" in answer.disclaimer or "legal" in answer.disclaimer.lower()


def test_off_topic_question_is_flagged(tmp_path) -> None:
    service = _service(tmp_path, min_dense_score=0.99)
    answer = service.ask("zzzz quantum pasta recipe on mars", language="en")
    assert answer.insufficient_context is True


def test_scripted_llm_is_used_when_context_is_sufficient(tmp_path) -> None:
    service = _service(tmp_path, llm=ScriptedLLM(), min_dense_score=0.0)
    answer = service.ask("franchise quote-part assurés", language="fr")
    assert answer.backend == "scripted"
    assert answer.insufficient_context is False
    assert "[1]" in answer.answer
    assert answer.citations[0].n == 1


def test_api_ask_and_validation(tmp_path) -> None:
    service = _service(tmp_path)
    with TestClient(create_app(service)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["chunks"] == 2
        rejected = client.post("/ask", json={"question": "ok"})
        assert rejected.status_code == 422
        accepted = client.post(
            "/ask",
            json={"question": "franchise et quote-part des assurés", "language": "fr"},
        )
        assert accepted.status_code == 200
        body = accepted.json()
        assert body["citations"]
        assert body["language"] == "fr"
