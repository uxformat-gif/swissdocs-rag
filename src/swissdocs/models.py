"""Shared data models."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable passage with the metadata needed to cite it."""

    id: str
    text: str
    law_sr: str
    law_abbr: str
    law_title: str
    language: str
    article: str
    article_title: str
    page: int | None = None
    version_date: str
    binding: bool = True
    source_url: str
    eli: str
    heading_path: list[str] = Field(default_factory=list)

    def label(self) -> str:
        """Short citation: statute, article, page, consolidation date."""
        title = f" {self.article_title}" if self.article_title else ""
        page = f", p. {self.page}" if self.page else ""
        note = "" if self.binding else ", non-binding translation"
        return (
            f"{self.law_abbr} (SR {self.law_sr}), art. {self.article}{title}"
            f"{page}{note} — version {self.version_date}"
        )

    def article_key(self) -> str:
        return f"{self.law_sr}:{self.language}:{self.article}"


class Hit(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None


class Citation(BaseModel):
    n: int
    chunk_id: str
    citation: str
    law_sr: str
    law_abbr: str
    article: str
    article_title: str
    language: str
    page: int | None
    version_date: str
    binding: bool
    source_url: str
    snippet: str
    heading_path: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    language: str
    answer: str
    insufficient_context: bool
    backend: str
    citations: list[Citation]
    disclaimer: str


class ParsedArticle(BaseModel):
    """Article extracted from an Akoma Ntoso consolidation, before chunking."""

    law_sr: str
    law_abbr: str
    law_title: str
    language: str
    article: str
    eId: str
    article_title: str
    text: str
    page: int | None = None
    version_date: str
    binding: bool
    source_url: str
    eli: str
    heading_path: list[str] = Field(default_factory=list)
