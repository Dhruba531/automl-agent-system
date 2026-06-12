"""Persistent memory for the Self-Harness loop.

The loop is otherwise stateless: every run starts from a fresh harness and
re-explores edits it has already tested. ``HarnessMemory`` gives it cross-run
memory — it persists the current best harness, every attempted edit and its
outcome, and a per-run history. On the next run the loop resumes from the stored
harness and skips edits it has already tried, so improvement accumulates instead
of restarting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

from automl_agent.self_harness.config import HarnessConfig

if TYPE_CHECKING:
    from automl_agent.self_harness.loop import SelfHarnessResult


@dataclass
class HarnessMemory:
    """Durable record of harness evolution across Self-Harness runs."""

    path: Path
    config: HarnessConfig = field(default_factory=HarnessConfig)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    runs: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "HarnessMemory":
        """Load memory from ``path``, or return empty memory if it does not exist."""
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            config=HarnessConfig.from_dict(payload.get("config", {})),
            attempts=list(payload.get("attempts", [])),
            runs=list(payload.get("runs", [])),
        )

    def attempted_keys(self) -> Set[Tuple[str, Any]]:
        """Every (op, value) edit already tested, to avoid re-proposing them."""
        return {(attempt["op"], attempt["value"]) for attempt in self.attempts}

    def is_empty(self) -> bool:
        return not self.runs

    def update(self, result: "SelfHarnessResult") -> None:
        """Fold a completed run into memory: best harness, attempts, run record."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.config = HarnessConfig.from_dict(result.final_config)
        for record in result.rounds:
            for candidate in record.candidates:
                edit = candidate.edit
                self.attempts.append(
                    {
                        "op": edit["op"],
                        "value": edit["value"],
                        "accepted": candidate.accepted,
                        "delta_in": candidate.delta_in,
                        "delta_ho": candidate.delta_ho,
                        "round": record.round_index,
                        "run": run_id,
                    }
                )
        self.runs.append(
            {
                "run": run_id,
                "passed_in": [result.initial_passed_in, result.final_passed_in],
                "passed_ho": [result.initial_passed_ho, result.final_passed_ho],
                "total_in": result.total_in,
                "total_ho": result.total_ho,
                "final_config": result.final_config,
            }
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "attempts": self.attempts,
            "runs": self.runs,
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
