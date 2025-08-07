"""Local LLM clients (Ollama, HF Text Generation Inference).

These wrappers let the existing orchestration layer treat on-device / on-prem
models exactly like cloud providers.  For now HFLocalClient is a stub; Ollama
client supports sync + streaming (non-chunked fallback) calls.
"""
from __future__ import annotations

import time
from typing import Dict, AsyncGenerator

import httpx

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.models import ModelProvider
from .ai_orchestrator import ModelClient, ModelRequest, ModelResponse  # relative import within same pkg


class OllamaClient(ModelClient, LoggerMixin):
    """Interact with an Ollama daemon (http://localhost:11434 by default)."""

    def __init__(self):
        super().__init__(ModelProvider.LOCAL_OLLAMA)
        self.base_url: str = settings.OLLAMA_HOST or "http://localhost:11434"
        self.default_model: str = "llama2"

    # ---------------------------------------------------------------------
    # Life-cycle
    # ---------------------------------------------------------------------
    async def initialize(self) -> bool:  # noqa: D401 – simple status return
        """Verify the Ollama daemon is reachable and list available models."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            self.is_available = True
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Ollama health-check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    async def generate(self, request: ModelRequest) -> ModelResponse:  # type: ignore[override]
        t0 = time.perf_counter()
        payload = {
            "model": request.model_preference or self.default_model,
            "prompt": request.prompt,
            "stream": False,
            "temperature": request.temperature,
            "top_p": 1,
            "max_tokens": request.max_tokens,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data: Dict = resp.json()

        t1 = time.perf_counter()
        # Ollama returns eval_count & prompt_eval_count in recent versions
        eval_count = int(data.get("eval_count", 0))
        prompt_count = int(data.get("prompt_eval_count", 0))
        tokens = eval_count + prompt_count

        return ModelResponse(
            content=data.get("response", ""),
            model_used=payload["model"],
            provider=self.provider,
            tokens_used=tokens,
            cost=0.0,  # local inference cost considered zero for now
            response_time=t1 - t0,
            finish_reason=data.get("done_reason", "stop"),
            metadata=data,
        )

    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Very simple streaming: performs non-stream generate & yields once.
        A full implementation could parse Server-Sent Events.
        """
        res = await self.generate(request)
        yield res.content

    async def health_check(self) -> Dict[str, str]:  # type: ignore[override]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
            status = "healthy" if r.status_code == 200 else "unhealthy"
            return {"status": status}
        except Exception as exc:  # pylint: disable=broad-except
            return {"status": "unhealthy", "error": str(exc)}


class HFLocalClient(ModelClient, LoggerMixin):
    """Placeholder for a Hugging Face Text-Generation-Inference local server."""

    def __init__(self):
        super().__init__(ModelProvider.LOCAL_HF)
        self.base_url = settings.HF_LOCAL_HOST

    async def initialize(self) -> bool:  # noqa: D401
        """Ping the /health endpoint of the local TGI instance."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/health")
            self.is_available = r.status_code == 200
            return self.is_available
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("HFLocal health-check failed: %s", exc)
            return False

    async def generate(self, request: ModelRequest) -> ModelResponse:  # type: ignore[override]
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
            },
        }
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{self.base_url}/generate", json=payload)
        resp.raise_for_status()
        data: Dict = resp.json()
        t1 = time.perf_counter()

        text = data.get("generated_text", "")
        tokens = data.get("details", {}).get("tokens", len(text.split()))

        return ModelResponse(
            content=text,
            model_used="local-hf",
            provider=self.provider,
            tokens_used=tokens,
            cost=0.0,
            response_time=t1 - t0,
            finish_reason="stop",
            metadata=data,
        )

    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:  # type: ignore[override]
        # Fallback to non-stream generate for now
        res = await self.generate(request)
        yield res.content

    async def health_check(self) -> Dict[str, str]:  # type: ignore[override]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/health")
            status = "healthy" if r.status_code == 200 else "unhealthy"
            return {"status": status}
        except Exception as exc:  # pylint: disable=broad-except
            return {"status": "unhealthy", "error": str(exc)}
