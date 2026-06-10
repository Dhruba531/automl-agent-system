from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol

from automl_agent.agents.base import BaseAgent
from automl_agent.paper2code.claude_client import ClaudeClient
from automl_agent.paper2code.extractor import GeneratedFile, extract_files
from automl_agent.paper2code.paper_loader import LoadedPaper, load_paper
from automl_agent.paper2code.prompts import SYSTEM_PROMPT, build_prompt


class SupportsComplete(Protocol):
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str: ...


@dataclass
class PaperToCodeResult:
    project_name: str
    paper_title: str
    paper_source: str
    output_dir: Path
    files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class PaperToCodeAgent(BaseAgent):
    """Turns an academic paper into a runnable code project via Claude.

    The agent loads the paper, prompts Claude (by default through the local
    Claude Code CLI, so it uses your subscription), parses the delimited
    response into files, and writes them to an output directory.
    """

    name = "Paper-to-Code Agent"

    def __init__(
        self,
        client: Optional[SupportsComplete] = None,
        *,
        language: str = "Python",
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.client: SupportsComplete = client or ClaudeClient(model=model)
        self.language = language

    def convert(
        self,
        source: str,
        output_dir: Path,
        *,
        project_name: Optional[str] = None,
        max_chars: int = 120_000,
        overwrite: bool = False,
    ) -> PaperToCodeResult:
        paper = load_paper(source, max_chars=max_chars)
        self.log(f"Loaded paper '{paper.title or paper.source}' from {paper.source}.")

        resolved_name = project_name or _slugify(paper.title) or "paper_implementation"
        prompt = build_prompt(paper, language=self.language, project_name=resolved_name)

        self.log("Requesting implementation from Claude.")
        response = self.client.complete(prompt, system_prompt=SYSTEM_PROMPT)

        files = extract_files(response)
        if not files:
            raise ValueError(
                "Claude did not return any file blocks. Re-run, or inspect the raw "
                "response saved alongside the output."
            )
        self.log(f"Parsed {len(files)} generated file(s) from the response.")

        output_dir.mkdir(parents=True, exist_ok=True)
        written = self._write_files(files, output_dir, overwrite=overwrite)
        (output_dir / "paper2code_raw_response.txt").write_text(response, encoding="utf-8")

        result = PaperToCodeResult(
            project_name=resolved_name,
            paper_title=paper.title,
            paper_source=paper.source,
            output_dir=output_dir,
            files=[str(path.relative_to(output_dir)) for path in written],
            notes=[f"{event.agent}: {event.message}" for event in self.events],
        )
        self._write_manifest(result, paper, output_dir)
        self.log(f"Wrote {len(written)} file(s) to {output_dir}.")
        return result

    def _write_files(
        self, files: List[GeneratedFile], output_dir: Path, *, overwrite: bool
    ) -> List[Path]:
        written: List[Path] = []
        for generated in files:
            destination = (output_dir / generated.path).resolve()
            if output_dir.resolve() not in destination.parents and destination != output_dir.resolve():
                raise ValueError(f"Refusing to write outside output dir: {generated.path}")
            if destination.exists() and not overwrite:
                self.log(f"Skipping existing file (use overwrite): {generated.path}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(generated.content, encoding="utf-8")
            written.append(destination)
        return written

    def _write_manifest(
        self, result: PaperToCodeResult, paper: LoadedPaper, output_dir: Path
    ) -> None:
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_name": result.project_name,
            "language": self.language,
            "paper_title": paper.title,
            "paper_source": paper.source,
            "files": result.files,
            "notes": result.notes,
        }
        (output_dir / "paper2code_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def _slugify(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:60]
