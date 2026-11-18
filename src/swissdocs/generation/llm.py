"""Ollama and OpenAI-compatible chat backends."""

import httpx


class LLMError(RuntimeError):
    pass


class LLM:
    name: str

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=180,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama request failed ({self.base_url}, model {self.model}). "
                "Is the daemon running and has the model been pulled?"
            ) from exc
        content = response.json().get("message", {}).get("content", "")
        if not content:
            raise LLMError("Ollama returned an empty message")
        return str(content)


class OpenAICompatibleLLM(LLM):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is empty")
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected chat completion payload") from exc
        return str(content)


def build_llm(
    backend: str,
    *,
    ollama_base_url: str,
    ollama_model: str,
    openai_base_url: str,
    openai_api_key: str,
    openai_model: str,
) -> LLM | None:
    if backend == "extractive":
        return None
    if backend == "ollama":
        return OllamaLLM(ollama_base_url, ollama_model)
    if backend == "openai":
        return OpenAICompatibleLLM(openai_base_url, openai_api_key, openai_model)
    raise ValueError(f"Unknown LLM backend: {backend}")
