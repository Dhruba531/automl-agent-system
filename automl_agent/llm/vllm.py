"""Connector for vLLM's OpenAI-compatible serving API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import httpx


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.2

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["VLLMConfig"]:
        """Build a config from VLLM_* variables, or None when no server is configured."""
        env = os.environ if env is None else env
        base_url = env.get("VLLM_BASE_URL")
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            model=env.get("VLLM_MODEL"),
            api_key=env.get("VLLM_API_KEY"),
            timeout_seconds=float(env.get("VLLM_TIMEOUT_SECONDS", "60")),
            max_tokens=int(env.get("VLLM_MAX_TOKENS", "512")),
            temperature=float(env.get("VLLM_TEMPERATURE", "0.2")),
        )


class VLLMConnector:
    """Minimal client for a vLLM server exposing the OpenAI-compatible API."""

    def __init__(self, config: Optional[VLLMConfig] = None, transport: Optional[httpx.BaseTransport] = None) -> None:
        self.config = config or VLLMConfig()
        headers = {"content-type": "application/json"}
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            headers=headers,
            timeout=self.config.timeout_seconds,
            transport=transport,
        )
        self._resolved_model: Optional[str] = self.config.model

    def list_models(self) -> List[str]:
        response = self._client.get("/models")
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]

    def resolve_model(self) -> str:
        """Return the configured model, or the first model served by vLLM."""
        if self._resolved_model:
            return self._resolved_model
        models = self.list_models()
        if not models:
            raise RuntimeError("vLLM server reports no available models.")
        self._resolved_model = models[0]
        return self._resolved_model

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        payload: Dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        payload.update(overrides)
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            raise RuntimeError("vLLM returned no choices for the chat completion.")
        return choices[0]["message"]["content"]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VLLMConnector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
