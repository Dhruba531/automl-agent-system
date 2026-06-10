"""Connector for RunPod serverless endpoints running vLLM workers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

import httpx

from automl_agent.llm.vllm import VLLMConfig, VLLMConnector

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass
class RunPodConfig:
    endpoint_id: str
    api_key: str
    model: Optional[str] = None
    api_base: str = RUNPOD_API_BASE
    # Generous default because serverless workers may cold-start.
    timeout_seconds: float = 120.0
    max_tokens: int = 512
    temperature: float = 0.2

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["RunPodConfig"]:
        """Build a config from RUNPOD_* variables, or None when not configured."""
        env = os.environ if env is None else env
        endpoint_id = env.get("RUNPOD_ENDPOINT_ID")
        api_key = env.get("RUNPOD_API_KEY")
        if not endpoint_id or not api_key:
            return None
        return cls(
            endpoint_id=endpoint_id,
            api_key=api_key,
            model=env.get("RUNPOD_MODEL"),
            api_base=env.get("RUNPOD_API_BASE", RUNPOD_API_BASE),
            timeout_seconds=float(env.get("RUNPOD_TIMEOUT_SECONDS", "120")),
            max_tokens=int(env.get("RUNPOD_MAX_TOKENS", "512")),
            temperature=float(env.get("RUNPOD_TEMPERATURE", "0.2")),
        )

    def openai_base_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/{self.endpoint_id}/openai/v1"


class RunPodConnector(VLLMConnector):
    """vLLM connector pointed at a RunPod serverless vLLM worker.

    RunPod exposes the worker's OpenAI-compatible API under
    ``https://api.runpod.ai/v2/<endpoint_id>/openai/v1`` with the account
    API key as bearer token, so the base connector works unchanged.
    """

    def __init__(self, config: RunPodConfig, transport: Optional[httpx.BaseTransport] = None) -> None:
        self.runpod_config = config
        super().__init__(
            VLLMConfig(
                base_url=config.openai_base_url(),
                model=config.model,
                api_key=config.api_key,
                timeout_seconds=config.timeout_seconds,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
            transport=transport,
        )
