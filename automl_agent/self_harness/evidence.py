"""Weakness Mining: evaluate a harness and cluster failures into evidence.

Implements the evaluation and evidence-bundle stages of Algorithm 1. A "task" is
a dataset case judged by a deterministic verifier (a pass threshold on the
cross-validated score, which is higher-is-better for both task types). Failures
are clustered by an evaluator-grounded signature so the proposer reasons about
recurring mechanisms rather than isolated cases.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from automl_agent.orchestrator import AutoMLOrchestrator
from automl_agent.self_harness.config import HarnessConfig


@dataclass
class HarnessCase:
    """A verifier-grounded task: a dataset plus the score it must reach."""

    name: str
    pass_threshold: float
    dataset: Optional[str] = None
    csv_path: Optional[Path] = None
    target: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "HarnessCase":
        if not payload.get("name"):
            raise ValueError("Each harness case needs a non-empty 'name'.")
        if "pass_threshold" not in payload:
            raise ValueError(f"Case '{payload['name']}' must set a 'pass_threshold'.")
        csv = payload.get("csv")
        return cls(
            name=str(payload["name"]),
            pass_threshold=float(payload["pass_threshold"]),
            dataset=payload.get("dataset"),
            csv_path=Path(csv) if csv else None,
            target=payload.get("target"),
        )


@dataclass
class CaseOutcome:
    case_name: str
    passed: bool
    score: Optional[float]
    threshold: float
    best_model: Optional[str]
    failed_candidates: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SplitResult:
    """Pass counts and per-case outcomes for one split under one harness."""

    passed: int
    total: int
    outcomes: List[CaseOutcome]


@dataclass
class FailurePattern:
    """A cluster of failures sharing an evaluator-grounded signature."""

    signature: Tuple[str, str, str]
    cause: str
    mechanism: str
    support: int
    cases: List[str]
    detail: str

    def actionability(self) -> int:
        # Candidate errors and a weak model pool map cleanly to bounded edits.
        ranked = {"candidate_error": 3, "weak_model_pool": 2, "below_threshold": 1}
        return ranked.get(self.cause, 0)


def evaluate(
    config: HarnessConfig,
    cases: List[HarnessCase],
    workdir: Path,
    max_workers: int = 2,
) -> SplitResult:
    """Run the AutoML harness on every case and verify against its threshold."""
    outcomes: List[CaseOutcome] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        outcomes.append(_evaluate_case(config, case, workdir / case.name, max_workers))
    passed = sum(1 for outcome in outcomes if outcome.passed)
    return SplitResult(passed=passed, total=len(cases), outcomes=outcomes)


def _evaluate_case(config: HarnessConfig, case: HarnessCase, output_dir: Path, max_workers: int) -> CaseOutcome:
    try:
        orchestrator = AutoMLOrchestrator(max_workers=max_workers, harness_config=config)
        report = orchestrator.run(
            output_dir=output_dir,
            dataset=case.dataset,
            csv_path=case.csv_path,
            target=case.target,
        )
    except Exception as exc:  # pipeline failed before a verifiable result
        return CaseOutcome(
            case_name=case.name,
            passed=False,
            score=None,
            threshold=case.pass_threshold,
            best_model=None,
            error=str(exc),
        )
    score = report.best_cv_score
    passed = score is not None and score >= case.pass_threshold
    return CaseOutcome(
        case_name=case.name,
        passed=passed,
        score=score,
        threshold=case.pass_threshold,
        best_model=report.best_model_name,
        failed_candidates=report.failed_candidates,
    )


def build_evidence_bundle(result: SplitResult) -> List[FailurePattern]:
    """Cluster held-in failures into ordered, evaluator-grounded patterns."""
    clusters: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = defaultdict(list)
    for outcome in result.outcomes:
        for signature, detail in _signatures(outcome):
            clusters[signature].append((outcome.case_name, detail))

    patterns: List[FailurePattern] = []
    for signature, members in clusters.items():
        cause, mechanism, _ = signature
        cases = sorted({name for name, _ in members})
        detail = members[0][1]
        patterns.append(
            FailurePattern(
                signature=signature,
                cause=cause,
                mechanism=mechanism,
                support=len(cases),
                cases=cases,
                detail=detail,
            )
        )
    # Order by support, then actionability (paper: surface recurring, actionable first).
    patterns.sort(key=lambda pattern: (pattern.support, pattern.actionability()), reverse=True)
    return patterns


def _signatures(outcome: CaseOutcome) -> List[Tuple[Tuple[str, str, str], str]]:
    """Map a case outcome to zero or more (signature, detail) pairs.

    Signature = (verifier-level cause, agent mechanism, scope) per the paper's
    deterministic clustering key.
    """
    if outcome.passed:
        return []
    signatures: List[Tuple[Tuple[str, str, str], str]] = []

    # A candidate that errored is a concrete, addressable mechanism.
    for failed in outcome.failed_candidates:
        model = failed.get("name", "unknown")
        signatures.append(
            (
                ("candidate_error", model, "candidate"),
                f"candidate '{model}' raised: {failed.get('error', '')[:160]}",
            )
        )

    if outcome.error is not None:
        signatures.append(
            (("pipeline_error", "orchestrator", "pipeline"), f"pipeline raised: {outcome.error[:160]}")
        )
    elif outcome.score is None:
        signatures.append(
            (("no_cv_score", "cv_splitter", "evaluation"), "no cross-validation score was produced")
        )
    else:
        gap = outcome.threshold - outcome.score
        # Below threshold but the pipeline ran: the model pool may be too weak.
        signatures.append(
            (
                ("weak_model_pool", outcome.best_model or "unknown", "selection"),
                f"best model '{outcome.best_model}' scored {outcome.score:.4f} < {outcome.threshold:.4f} "
                f"(gap {gap:.4f})",
            )
        )
    return signatures
