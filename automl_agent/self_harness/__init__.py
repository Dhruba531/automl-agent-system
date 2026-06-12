"""Self-Harness: an AutoML harness that improves itself.

Instantiates the Self-Harness loop (arXiv:2606.09498) over the AutoML agent's
search configuration: mine weaknesses from held-in dataset failures, propose
bounded harness edits, and promote only those passing a held-in/held-out
regression gate.
"""

from automl_agent.self_harness.config import EDIT_OPS, HarnessConfig, HarnessEdit
from automl_agent.self_harness.evidence import (
    CaseOutcome,
    FailurePattern,
    HarnessCase,
    SplitResult,
    build_evidence_bundle,
    evaluate,
)
from automl_agent.self_harness.loop import SelfHarness, SelfHarnessResult
from automl_agent.self_harness.memory import HarnessMemory
from automl_agent.self_harness.proposer import HarnessProposer

__all__ = [
    "EDIT_OPS",
    "HarnessConfig",
    "HarnessEdit",
    "HarnessCase",
    "CaseOutcome",
    "FailurePattern",
    "SplitResult",
    "build_evidence_bundle",
    "evaluate",
    "HarnessProposer",
    "HarnessMemory",
    "SelfHarness",
    "SelfHarnessResult",
]
