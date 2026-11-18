"""FastAPI application."""

import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from swissdocs import __version__
from swissdocs.config import get_settings
from swissdocs.generation.llm import LLMError
from swissdocs.models import Answer
from swissdocs.service import RagService

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    language: str | None = Field(default=None, pattern="^(fr|de|en)$")
    cross_lingual: bool = False


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        if service is not None:
            app.state.service = service
            yield
            return
        if os.getenv("SWISSDOCS_EAGER_INDEX", "1") != "1":
            app.state.service = None
            yield
            return
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        built = RagService(get_settings())
        logger.info("Loading corpus and embeddings")
        built.prepare()
        app.state.service = built
        yield

    app = FastAPI(
        title="SwissDocs RAG",
        version=__version__,
        summary="Multilingual question answering over Swiss insurance legislation.",
        lifespan=lifespan,
    )

    def get_service() -> RagService:
        current = getattr(app.state, "service", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Index is not ready")
        return current

    @app.get("/health")
    def health(rag: RagService = Depends(get_service)) -> dict[str, object]:
        return rag.health()

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, rag: RagService = Depends(get_service)) -> Answer:
        try:
            return rag.ask(
                body.question,
                language=body.language,
                cross_lingual=body.cross_lingual,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "swissdocs.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
