from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List, Optional

from automl_agent.paper2code.prompts import FILE_END

_FILE_HEADER = re.compile(r"^===\s*FILE:\s*(?P<path>.+?)\s*===\s*$")


@dataclass
class GeneratedFile:
    path: str
    content: str


def extract_files(response: str) -> List[GeneratedFile]:
    """Parse Claude's delimited response into a list of files.

    Tolerates the model occasionally wrapping the whole response in a Markdown
    fence, and strips trailing whitespace from each file. Unsafe paths
    (absolute, parent traversal) are rejected.
    """
    files: List[GeneratedFile] = []
    current_path: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in response.splitlines():
        header = _FILE_HEADER.match(raw_line.strip())
        if header:
            if current_path is not None:
                files.append(_finalize(current_path, current_lines))
            current_path = _safe_path(header.group("path"))
            current_lines = []
            continue
        if raw_line.strip() == FILE_END:
            if current_path is not None:
                files.append(_finalize(current_path, current_lines))
                current_path = None
                current_lines = []
            continue
        if current_path is not None:
            current_lines.append(raw_line)

    if current_path is not None:
        files.append(_finalize(current_path, current_lines))

    return files


def _finalize(path: str, lines: List[str]) -> GeneratedFile:
    content = "\n".join(lines).strip("\n")
    if content and not content.endswith("\n"):
        content += "\n"
    return GeneratedFile(path=path, content=content)


def _safe_path(path: str) -> str:
    cleaned = path.strip().strip("`").strip().replace("\\", "/")
    pure = PurePosixPath(cleaned)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"Unsafe file path from model output: {path!r}")
    return str(pure)
