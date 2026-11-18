"""Runtime configuration. All secrets come from the environment."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DISCLAIMER = (
    "SwissDocs is a research demo, not legal advice and not a product of the "
    "Swiss Confederation. Consolidated texts on Fedlex prevail. English Fedlex "
    "versions are non-binding translations and may lag the official French, "
    "German and Italian texts."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""
    qdrant_collection: str = "swissdocs"
    corpus_path: str = "data/sample/chunks.jsonl"
    custom_corpus_path: str = ""

    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_backend: str = "fastembed"

    llm_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_fallback_extractive: bool = True

    top_k: int = 5
    retrieve_k: int = 20
    rrf_k: int = 60
    # Cosine gate calibrated on data/eval (see README). 0 disables it.
    min_dense_score: float = 0.78
    chunk_chars: int = 1100
    chunk_overlap: int = 180

    host: str = "0.0.0.0"
    port: int = 8000

    fedlex_sparql_url: str = "https://fedlex.data.admin.ch/sparqlendpoint"
    raw_dir: str = "data/raw"

    disclaimer: str = DISCLAIMER

    def corpus_file(self) -> Path:
        return Path(self.corpus_path)

    def custom_corpus_file(self) -> Path | None:
        if not self.custom_corpus_path:
            return None
        path = Path(self.custom_corpus_path)
        return path if path.is_file() else None


def get_settings() -> Settings:
    return Settings()


class LawSpec(BaseModel):
    """Static description of a Classified Compilation act."""

    sr: str
    eli: str
    abbr: dict[str, str]
    title: dict[str, str] = Field(default_factory=dict)


LAWS: tuple[LawSpec, ...] = (
    LawSpec(
        sr="832.10",
        eli="cc/1995/1328_1328_1328",
        abbr={"fr": "LAMal", "de": "KVG", "en": "HIA"},
        title={
            "fr": "Loi fédérale sur l'assurance-maladie",
            "de": "Bundesgesetz über die Krankenversicherung",
            "en": "Federal Act on Health Insurance",
        },
    ),
    LawSpec(
        sr="221.229.1",
        eli="cc/24/719_735_717",
        abbr={"fr": "LCA", "de": "VVG", "en": "ICA"},
        title={
            "fr": "Loi fédérale sur le contrat d'assurance",
            "de": "Bundesgesetz über den Versicherungsvertrag",
            "en": "Federal Act on Insurance Contracts",
        },
    ),
)

LANGUAGES: tuple[str, ...] = ("fr", "de", "en")

LANG_URI = {
    "fr": "http://publications.europa.eu/resource/authority/language/FRA",
    "de": "http://publications.europa.eu/resource/authority/language/DEU",
    "en": "http://publications.europa.eu/resource/authority/language/ENG",
}
