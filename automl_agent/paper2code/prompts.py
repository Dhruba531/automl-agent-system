from __future__ import annotations

from automl_agent.paper2code.paper_loader import LoadedPaper

FILE_BEGIN = "=== FILE:"
FILE_END = "=== END FILE ==="

SYSTEM_PROMPT = (
    "You are a senior research engineer who reproduces academic papers as clean, "
    "runnable code. You read the paper carefully, identify the core method, "
    "algorithms, model architecture, training procedure, and experiments, then "
    "implement them faithfully. You favour readable, well-structured, idiomatic "
    "code with docstrings and type hints. You never invent results; when the paper "
    "omits a detail you choose a sensible default and note the assumption."
)


def build_prompt(paper: LoadedPaper, *, language: str, project_name: str) -> str:
    """Build the single user prompt sent to Claude.

    The model is asked to emit each file inside an unambiguous delimiter block so
    the response can be parsed without depending on Markdown fences or JSON
    escaping.
    """
    return f"""\
Convert the following research paper into a working {language} implementation.

Project name: {project_name}
Paper title: {paper.title or "(untitled)"}
Paper source: {paper.source}

Requirements:
- Implement the paper's core contribution (model/algorithm/method), not just a stub.
- Produce a small, runnable project: source modules, a minimal runnable entry point
  or example, a requirements/dependency file, and a README explaining how to run it.
- Include unit-testable functions and at least one lightweight test where reasonable.
- Add concise docstrings citing the relevant section/equation of the paper.
- Where the paper is ambiguous, pick a reasonable default and record the assumption
  in the README under an "Assumptions" section.

Output format — STRICT. Emit ONLY a sequence of file blocks and nothing else.
For every file use exactly this structure:

{FILE_BEGIN} relative/path/to/file.ext ===
<full file content here>
{FILE_END}

Rules for the output:
- Use forward-slash relative paths. Do not use absolute paths or '..'.
- Do not wrap file blocks in Markdown code fences.
- Do not add commentary before the first block or after the last block.
- The first file must be README.md.

=== BEGIN PAPER ===
{paper.text}
=== END PAPER ===
"""
