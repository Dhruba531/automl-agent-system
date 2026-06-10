"""Paper-to-Code agent.

Converts an academic paper into a runnable code repository using your Claude
subscription. The Claude Code CLI is driven in headless mode so generation is
billed against your existing subscription rather than a metered API key.
"""

from automl_agent.paper2code.agent import PaperToCodeAgent, PaperToCodeResult
from automl_agent.paper2code.claude_client import ClaudeClient, ClaudeUnavailableError
from automl_agent.paper2code.extractor import GeneratedFile, extract_files
from automl_agent.paper2code.paper_loader import LoadedPaper, load_paper

__all__ = [
    "PaperToCodeAgent",
    "PaperToCodeResult",
    "ClaudeClient",
    "ClaudeUnavailableError",
    "GeneratedFile",
    "extract_files",
    "LoadedPaper",
    "load_paper",
]
