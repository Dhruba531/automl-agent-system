from pathlib import Path

import pytest

from automl_agent.paper2code import PaperToCodeAgent, extract_files, load_paper
from automl_agent.paper2code.prompts import SYSTEM_PROMPT, build_prompt


class StubClaude:
    """Records the prompt and returns a canned, delimited response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = None
        self.last_system = None

    def complete(self, prompt, system_prompt=None):
        self.last_prompt = prompt
        self.last_system = system_prompt
        return self.response


CANNED_RESPONSE = """\
=== FILE: README.md ===
# Demo Implementation

Reproduces the toy method from the paper.
=== END FILE ===
=== FILE: src/model.py ===
def predict(x):
    return x * 2
=== END FILE ===
=== FILE: requirements.txt ===
numpy
=== END FILE ===
"""


def test_extract_files_parses_blocks() -> None:
    files = extract_files(CANNED_RESPONSE)
    paths = {f.path for f in files}
    assert paths == {"README.md", "src/model.py", "requirements.txt"}
    model = next(f for f in files if f.path == "src/model.py")
    assert "return x * 2" in model.content
    assert model.content.endswith("\n")


def test_extract_files_rejects_unsafe_paths() -> None:
    with pytest.raises(ValueError):
        extract_files("=== FILE: ../escape.py ===\nx = 1\n=== END FILE ===\n")


def test_load_paper_inline_text() -> None:
    paper = load_paper("Attention Is All You Need\n\nWe propose the Transformer.")
    assert paper.source == "inline-text"
    assert paper.title.startswith("Attention Is All You Need")


def test_load_paper_truncates(tmp_path: Path) -> None:
    big = tmp_path / "paper.txt"
    big.write_text("A useful title line\n" + "x" * 500, encoding="utf-8")
    paper = load_paper(str(big), max_chars=100)
    assert "truncated" in paper.text


def test_build_prompt_includes_paper_and_rules() -> None:
    paper = load_paper("My Paper Title\n\nBody text here.")
    prompt = build_prompt(paper, language="Python", project_name="demo")
    assert "Body text here." in prompt
    assert "=== FILE:" in prompt
    assert "README.md" in prompt


def test_agent_converts_and_writes_files(tmp_path: Path) -> None:
    client = StubClaude(CANNED_RESPONSE)
    agent = PaperToCodeAgent(client=client)
    result = agent.convert("A Toy Paper\n\nMethod description.", tmp_path)

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "src" / "model.py").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / "paper2code_manifest.json").exists()
    assert (tmp_path / "paper2code_raw_response.txt").exists()
    assert set(result.files) >= {"README.md", "src/model.py", "requirements.txt"}
    assert client.last_system == SYSTEM_PROMPT
    assert "Method description." in client.last_prompt


def test_agent_skips_existing_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("keep me", encoding="utf-8")
    agent = PaperToCodeAgent(client=StubClaude(CANNED_RESPONSE))
    agent.convert("Title line here\n\nx", tmp_path)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "keep me"

    agent2 = PaperToCodeAgent(client=StubClaude(CANNED_RESPONSE))
    agent2.convert("Title line here\n\nx", tmp_path, overwrite=True)
    assert "Demo Implementation" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_agent_raises_when_no_files_returned(tmp_path: Path) -> None:
    agent = PaperToCodeAgent(client=StubClaude("no file blocks here"))
    with pytest.raises(ValueError):
        agent.convert("Some title\n\nbody", tmp_path)
    assert (tmp_path / "paper2code_raw_response.txt").read_text(encoding="utf-8") == "no file blocks here"


def test_agent_handles_relative_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent = PaperToCodeAgent(client=StubClaude(CANNED_RESPONSE))
    result = agent.convert("A Toy Paper\n\nMethod.", Path("artifacts/demo"))
    assert set(result.files) >= {"README.md", "src/model.py"}
    assert (tmp_path / "artifacts" / "demo" / "README.md").exists()


def test_inline_text_citing_arxiv_url_stays_inline() -> None:
    paper = load_paper("Our Method Title\n\nWe build on prior work (arxiv.org/abs/1706.03762).")
    assert paper.source == "inline-text"
    assert "prior work" in paper.text
