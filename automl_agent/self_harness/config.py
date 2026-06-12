"""Editable harness state and bounded edits for the Self-Harness loop.

A ``HarnessConfig`` is the non-parametric scaffolding around the fixed AutoML
agents: which candidate models are searched, how many CV folds are used, and the
tuning budget. The Self-Harness loop proposes bounded ``HarnessEdit`` operations
over these surfaces and promotes only those that pass a regression gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Tuple

# The bounded vocabulary of edit operations the proposer may emit.
EDIT_OPS: Tuple[str, ...] = (
    "enable_candidate",
    "disable_candidate",
    "set_cv_splits",
    "set_tuning_trials",
)


@dataclass(frozen=True)
class HarnessConfig:
    """A declarative, immutable snapshot of the agent harness."""

    disabled_candidates: FrozenSet[str] = field(default_factory=frozenset)
    enabled_extra_candidates: FrozenSet[str] = field(default_factory=frozenset)
    cv_splits: int = 3
    tuning_trials: int = 0

    def apply(self, edit: "HarnessEdit") -> "HarnessConfig":
        """Return a new config with the edit applied (no-op surfaces stay equal)."""
        if edit.op == "enable_candidate":
            return replace(
                self,
                enabled_extra_candidates=self.enabled_extra_candidates | {edit.value},
                disabled_candidates=self.disabled_candidates - {edit.value},
            )
        if edit.op == "disable_candidate":
            return replace(
                self,
                disabled_candidates=self.disabled_candidates | {edit.value},
                enabled_extra_candidates=self.enabled_extra_candidates - {edit.value},
            )
        if edit.op == "set_cv_splits":
            return replace(self, cv_splits=int(edit.value))
        if edit.op == "set_tuning_trials":
            return replace(self, tuning_trials=int(edit.value))
        raise ValueError(f"Unknown harness edit op: {edit.op}")

    def fingerprint(self) -> str:
        parts = [
            f"disabled={sorted(self.disabled_candidates)}",
            f"extra={sorted(self.enabled_extra_candidates)}",
            f"cv={self.cv_splits}",
            f"trials={self.tuning_trials}",
        ]
        return "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disabled_candidates": sorted(self.disabled_candidates),
            "enabled_extra_candidates": sorted(self.enabled_extra_candidates),
            "cv_splits": self.cv_splits,
            "tuning_trials": self.tuning_trials,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HarnessConfig":
        return cls(
            disabled_candidates=frozenset(payload.get("disabled_candidates", [])),
            enabled_extra_candidates=frozenset(payload.get("enabled_extra_candidates", [])),
            cv_splits=int(payload.get("cv_splits", 3)),
            tuning_trials=int(payload.get("tuning_trials", 0)),
        )


@dataclass(frozen=True)
class HarnessEdit:
    """A single bounded modification to one editable harness surface.

    ``target_pattern`` ties the edit to the failure-pattern signature it
    addresses, and ``rationale`` is the human-readable audit record (the paper's
    ``a_j``).
    """

    op: str
    value: Any
    target_pattern: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in EDIT_OPS:
            raise ValueError(f"Unsupported edit op '{self.op}'. Allowed: {EDIT_OPS}.")

    def surface(self) -> str:
        """The editable surface this edit touches, used for diversity checks."""
        if self.op in ("enable_candidate", "disable_candidate"):
            return "candidate_pool"
        if self.op == "set_cv_splits":
            return "cv_splits"
        return "tuning_trials"

    def key(self) -> Tuple[str, Any]:
        return (self.op, self.value)

    def describe(self) -> str:
        return f"{self.op}({self.value})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "value": self.value,
            "target_pattern": self.target_pattern,
            "rationale": self.rationale,
        }
