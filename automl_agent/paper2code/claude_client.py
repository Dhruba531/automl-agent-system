from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


class ClaudeUnavailableError(RuntimeError):
    """Raised when no usable Claude backend can be located."""


@dataclass
class ClaudeClient:
    """Drives Claude using the local Claude Code CLI in headless mode.

    Generation runs through ``claude --print`` which authenticates with the
    signed-in Claude subscription, so calls are billed against the
    subscription rather than a metered ``ANTHROPIC_API_KEY``. The prompt is
    streamed over stdin to avoid argument-length limits on large papers.
    """

    binary: str = "claude"
    model: Optional[str] = None
    timeout: int = 1200
    extra_args: List[str] = field(default_factory=list)

    def resolve_binary(self) -> str:
        path = shutil.which(self.binary)
        if path is None:
            raise ClaudeUnavailableError(
                f"Could not find the '{self.binary}' CLI on PATH. Install Claude Code "
                "and sign in with your subscription, or pass a custom client."
            )
        return path

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        binary = self.resolve_binary()
        command = [binary, "--print", "--output-format", "text"]
        if self.model:
            command += ["--model", self.model]
        if system_prompt:
            command += ["--append-system-prompt", system_prompt]
        command += self.extra_args

        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ},
            )
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise ClaudeUnavailableError(str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeUnavailableError(
                f"Claude CLI timed out after {self.timeout}s. Increase the timeout "
                "or shorten the paper."
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise ClaudeUnavailableError(
                f"Claude CLI exited with code {completed.returncode}: {stderr or 'no stderr'}"
            )

        output = (completed.stdout or "").strip()
        if not output:
            raise ClaudeUnavailableError("Claude CLI returned an empty response.")
        return output
