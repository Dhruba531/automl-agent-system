import json
from pathlib import Path

import httpx
import pytest

from automl_agent.cli import _build_llm_connector, _resolve_user_prompt, build_parser
from automl_agent.llm import RunPodConfig, RunPodConnector, VLLMConnector


def _mock_transport(captured: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["last_request"] = request
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "served-model"}]})
        if request.url.path.endswith("/chat/completions"):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_runpod_connector_targets_endpoint_openai_route() -> None:
    captured: dict = {}
    config = RunPodConfig(endpoint_id="abc123", api_key="rp-secret")
    with RunPodConnector(config, transport=_mock_transport(captured)) as connector:
        reply = connector.chat([{"role": "user", "content": "hi"}])

    assert reply == "ok"
    request = captured["last_request"]
    assert request.url.host == "api.runpod.ai"
    assert request.url.path == "/v2/abc123/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer rp-secret"
    # Model auto-discovered from the worker when RUNPOD_MODEL is not set.
    assert captured["payload"]["model"] == "served-model"


def test_runpod_config_from_env() -> None:
    assert RunPodConfig.from_env({}) is None
    assert RunPodConfig.from_env({"RUNPOD_ENDPOINT_ID": "abc123"}) is None
    config = RunPodConfig.from_env(
        {
            "RUNPOD_ENDPOINT_ID": "abc123",
            "RUNPOD_API_KEY": "rp-secret",
            "RUNPOD_MODEL": "llama",
            "RUNPOD_TIMEOUT_SECONDS": "30",
        }
    )
    assert config is not None
    assert config.model == "llama"
    assert config.timeout_seconds == 30.0
    assert config.openai_base_url() == "https://api.runpod.ai/v2/abc123/openai/v1"


def test_cli_builds_runpod_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    args = build_parser().parse_args(["run", "--llm-model", "llama"])
    connector = _build_llm_connector(args)
    assert isinstance(connector, RunPodConnector)
    assert connector.runpod_config.endpoint_id == "abc123"
    assert connector.config.model == "llama"
    connector.close()


def test_cli_prefers_vllm_over_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    args = build_parser().parse_args(["run"])
    connector = _build_llm_connector(args)
    assert isinstance(connector, VLLMConnector)
    assert not isinstance(connector, RunPodConnector)
    connector.close()


def test_cli_runpod_flag_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    args = build_parser().parse_args(["run", "--runpod-endpoint-id", "abc123"])
    with pytest.raises(SystemExit):
        _build_llm_connector(args)


def test_resolve_user_prompt_inline_and_file(tmp_path: Path) -> None:
    assert _resolve_user_prompt(None) is None
    assert _resolve_user_prompt("focus on risks") == "focus on risks"

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("summarize for executives", encoding="utf-8")
    assert _resolve_user_prompt(f"@{prompt_file}") == "summarize for executives"


def test_resolve_user_prompt_missing_file_errors() -> None:
    with pytest.raises(SystemExit):
        _resolve_user_prompt("@/nonexistent/prompt.txt")
