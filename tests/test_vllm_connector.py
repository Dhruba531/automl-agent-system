import json
from pathlib import Path

import httpx

from automl_agent.llm import VLLMConfig, VLLMConnector
from automl_agent.orchestrator import AutoMLOrchestrator


def _mock_transport(captured: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["last_request"] = request
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen-2.5"}, {"id": "other"}]})
        if request.url.path.endswith("/chat/completions"):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Looks good."}}]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_connector_chat_resolves_first_served_model() -> None:
    captured: dict = {}
    config = VLLMConfig(base_url="http://vllm.test/v1", api_key="secret")
    with VLLMConnector(config, transport=_mock_transport(captured)) as connector:
        assert connector.list_models() == ["qwen-2.5", "other"]
        reply = connector.chat([{"role": "user", "content": "hi"}])

    assert reply == "Looks good."
    assert captured["payload"]["model"] == "qwen-2.5"
    assert captured["payload"]["messages"][0]["content"] == "hi"
    assert captured["last_request"].headers["authorization"] == "Bearer secret"
    assert captured["last_request"].url.path == "/v1/chat/completions"


def test_connector_uses_configured_model_and_overrides() -> None:
    captured: dict = {}
    config = VLLMConfig(base_url="http://vllm.test/v1", model="my-model", max_tokens=64)
    with VLLMConnector(config, transport=_mock_transport(captured)) as connector:
        connector.chat([{"role": "user", "content": "hi"}], temperature=0.9)

    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["temperature"] == 0.9


def test_config_from_env() -> None:
    assert VLLMConfig.from_env({}) is None
    config = VLLMConfig.from_env(
        {"VLLM_BASE_URL": "http://host:8000/v1", "VLLM_MODEL": "llama", "VLLM_MAX_TOKENS": "128"}
    )
    assert config is not None
    assert config.base_url == "http://host:8000/v1"
    assert config.model == "llama"
    assert config.max_tokens == 128


class _FakeConnector:
    def __init__(self, reply: str = "## Run Summary\nSolid baseline.") -> None:
        self.reply = reply
        self.calls: list = []

    def chat(self, messages, **overrides):
        self.calls.append(messages)
        return self.reply


class _FailingConnector:
    def chat(self, messages, **overrides):
        raise httpx.ConnectError("vLLM server unreachable")


def test_pipeline_writes_llm_summary(tmp_path: Path) -> None:
    connector = _FakeConnector()
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=0, llm_connector=connector)
    report = orchestrator.run(output_dir=tmp_path, dataset="iris")

    assert report.llm_summary == connector.reply
    assert (tmp_path / "llm_summary.md").read_text(encoding="utf-8") == connector.reply
    prompt = connector.calls[0][1]["content"]
    assert "Leaderboard" in prompt
    assert "hist_gradient_boosting" in prompt


def test_pipeline_survives_llm_failure(tmp_path: Path) -> None:
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=0, llm_connector=_FailingConnector())
    report = orchestrator.run(output_dir=tmp_path, dataset="iris")

    assert report.llm_summary is None
    assert not (tmp_path / "llm_summary.md").exists()
    assert any("LLM summary failed" in note for note in report.notes)
