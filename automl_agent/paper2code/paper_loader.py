from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ARXIV_ID = re.compile(r"^(arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)


@dataclass
class LoadedPaper:
    title: str
    text: str
    source: str


def load_paper(source: str, *, max_chars: int = 120_000) -> LoadedPaper:
    """Load paper text from a local file, an arXiv id/URL, or raw text.

    Resolution order:
      1. An existing local path (``.txt``, ``.md``, or ``.pdf``).
      2. An arXiv id (e.g. ``2410.02958``) or ``arxiv.org`` URL, fetched as PDF.
      3. Otherwise the string is treated as the paper text itself.
    """
    path = Path(source)
    if path.exists() and path.is_file():
        return _load_file(path, max_chars=max_chars)

    arxiv_id = _arxiv_id(source)
    if arxiv_id:
        return _load_arxiv(arxiv_id, max_chars=max_chars)

    text = _truncate(source, max_chars)
    return LoadedPaper(title=_guess_title(text), text=text, source="inline-text")


def _load_file(path: Path, *, max_chars: int) -> LoadedPaper:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    text = _truncate(text, max_chars)
    return LoadedPaper(title=_guess_title(text) or path.stem, text=text, source=str(path))


def _arxiv_id(source: str) -> Optional[str]:
    candidate = source.strip()
    match = ARXIV_ID.match(candidate)
    if match:
        return match.group(2)
    # Only treat single-token sources as URLs; inline paper text often *cites*
    # arXiv links and must not trigger a download of the cited paper.
    if not re.search(r"\s", candidate):
        url_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", candidate, re.IGNORECASE)
        if url_match:
            return url_match.group(1)
    return None


def _load_arxiv(arxiv_id: str, *, max_chars: int) -> LoadedPaper:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Fetching arXiv papers requires 'httpx'. Install it or pass a local file."
        ) from exc

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = httpx.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not download arXiv paper {arxiv_id}: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="paper2code_") as tmp_dir:
        tmp = Path(tmp_dir) / f"{arxiv_id}.pdf"
        tmp.write_bytes(response.content)
        text = _truncate(_extract_pdf(tmp), max_chars)
    return LoadedPaper(title=_guess_title(text) or f"arXiv:{arxiv_id}", text=text, source=f"arxiv:{arxiv_id}")


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading PDF papers requires 'pypdf'. Install it with "
            "'pip install \"automl-agent-system[paper]\"' or convert the paper to text/markdown."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n\n[... paper truncated for length ...]"
    return text


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if len(stripped) >= 8:
            return stripped[:140]
    return ""
