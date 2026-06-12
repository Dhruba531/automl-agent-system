"""Self-Harness: an iterative loop that improves the AutoML harness.

Faithful instantiation of Algorithm 1 from "Self-Harness: Harnesses That Improve
Themselves". Each round evaluates the current harness on held-in and held-out
splits, mines weaknesses from held-in failures, proposes K bounded edits, and
promotes only edits passing the conservative acceptance rule:

    accept iff  d_in >= 0 and d_ho >= 0 and max(d_in, d_ho) > 0

Accepted edits are merged into the next harness; everything is logged to an
auditable lineage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from automl_agent.self_harness.config import HarnessConfig, HarnessEdit
from automl_agent.self_harness.evidence import (
    HarnessCase,
    SplitResult,
    build_evidence_bundle,
    evaluate,
)
from automl_agent.self_harness.proposer import HarnessProposer


@dataclass
class CandidateRecord:
    edit: dict
    delta_in: int
    delta_ho: int
    accepted: bool
    reason: str


@dataclass
class RoundRecord:
    round_index: int
    config_before: dict
    passed_in: int
    total_in: int
    passed_ho: int
    total_ho: int
    failure_patterns: List[dict]
    candidates: List[CandidateRecord] = field(default_factory=list)
    config_after: dict = field(default_factory=dict)


@dataclass
class SelfHarnessResult:
    initial_config: dict
    final_config: dict
    initial_passed_in: int
    final_passed_in: int
    initial_passed_ho: int
    final_passed_ho: int
    total_in: int
    total_ho: int
    rounds: List[RoundRecord]


class SelfHarness:
    def __init__(
        self,
        held_in: List[HarnessCase],
        held_out: List[HarnessCase],
        output_dir: Path,
        connector=None,
        proposal_width: int = 3,
        rounds: int = 3,
        max_workers: int = 2,
    ) -> None:
        if not held_in:
            raise ValueError("Self-Harness needs at least one held-in case.")
        if not held_out:
            raise ValueError("Self-Harness needs at least one held-out case for regression testing.")
        self.held_in = held_in
        self.held_out = held_out
        self.output_dir = output_dir
        self.proposer = HarnessProposer(connector)
        self.proposal_width = proposal_width
        self.rounds = rounds
        self.max_workers = max_workers

    def run(self, initial: Optional[HarnessConfig] = None) -> SelfHarnessResult:
        config = initial or HarnessConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        attempted: set = set()
        round_records: List[RoundRecord] = []

        initial_in = self._eval(config, self.held_in, 0, "in")
        initial_ho = self._eval(config, self.held_out, 0, "ho")
        passed_in, passed_ho = initial_in.passed, initial_ho.passed

        for t in range(self.rounds):
            in_result = initial_in if t == 0 else self._eval(config, self.held_in, t, "in")
            ho_result = initial_ho if t == 0 else self._eval(config, self.held_out, t, "ho")
            patterns = build_evidence_bundle(in_result)

            record = RoundRecord(
                round_index=t,
                config_before=config.to_dict(),
                passed_in=in_result.passed,
                total_in=in_result.total,
                passed_ho=ho_result.passed,
                total_ho=ho_result.total,
                failure_patterns=[_pattern_dict(p) for p in patterns],
            )

            proposals = self.proposer.propose(config, patterns, self.proposal_width, attempted)
            accepted: List[HarnessEdit] = []
            for j, edit in enumerate(proposals):
                attempted.add(edit.key())
                candidate = config.apply(edit)
                cand_in = self._eval(candidate, self.held_in, t, f"cand{j}_in")
                cand_ho = self._eval(candidate, self.held_out, t, f"cand{j}_ho")
                delta_in = cand_in.passed - in_result.passed
                delta_ho = cand_ho.passed - ho_result.passed
                ok = delta_in >= 0 and delta_ho >= 0 and max(delta_in, delta_ho) > 0
                record.candidates.append(
                    CandidateRecord(
                        edit=edit.to_dict(),
                        delta_in=delta_in,
                        delta_ho=delta_ho,
                        accepted=ok,
                        reason=_decision_reason(delta_in, delta_ho, ok),
                    )
                )
                if ok:
                    accepted.append(edit)

            # Merge accepted edits; re-verify the merged harness so a merge that
            # interacts badly cannot regress the splits (conservative promotion).
            if accepted:
                merged = config
                for edit in accepted:
                    merged = merged.apply(edit)
                merged_in = self._eval(merged, self.held_in, t, "merged_in")
                merged_ho = self._eval(merged, self.held_out, t, "merged_ho")
                if merged_in.passed >= passed_in and merged_ho.passed >= passed_ho and (
                    merged_in.passed > passed_in or merged_ho.passed > passed_ho
                ):
                    config = merged
                    passed_in, passed_ho = merged_in.passed, merged_ho.passed

            record.config_after = config.to_dict()
            round_records.append(record)

        result = SelfHarnessResult(
            initial_config=(initial or HarnessConfig()).to_dict(),
            final_config=config.to_dict(),
            initial_passed_in=initial_in.passed,
            final_passed_in=passed_in,
            initial_passed_ho=initial_ho.passed,
            final_passed_ho=passed_ho,
            total_in=initial_in.total,
            total_ho=initial_ho.total,
            rounds=round_records,
        )
        self._write(result)
        return result

    def _eval(self, config: HarnessConfig, cases: List[HarnessCase], round_index: int, tag: str) -> SplitResult:
        workdir = self.output_dir / f"round_{round_index}" / tag
        return evaluate(config, cases, workdir, max_workers=self.max_workers)

    def _write(self, result: SelfHarnessResult) -> None:
        (self.output_dir / "lineage.json").write_text(
            json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
        )
        self._write_summary(result)

    def _write_summary(self, result: SelfHarnessResult) -> None:
        lines = [
            "# Self-Harness Summary",
            "",
            f"- Held-in pass: {result.initial_passed_in}/{result.total_in} -> "
            f"{result.final_passed_in}/{result.total_in}",
            f"- Held-out pass: {result.initial_passed_ho}/{result.total_ho} -> "
            f"{result.final_passed_ho}/{result.total_ho}",
            "",
            "## Final harness",
            "```json",
            json.dumps(result.final_config, indent=2),
            "```",
            "",
            "## Accepted edits by round",
        ]
        any_accepted = False
        for record in result.rounds:
            for candidate in record.candidates:
                if candidate.accepted:
                    any_accepted = True
                    edit = candidate.edit
                    lines.append(
                        f"- round {record.round_index}: {edit['op']}({edit['value']}) "
                        f"[d_in={candidate.delta_in}, d_ho={candidate.delta_ho}] — {edit['rationale']}"
                    )
        if not any_accepted:
            lines.append("- none (no edit passed the acceptance rule)")
        (self.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pattern_dict(pattern) -> dict:
    return {
        "cause": pattern.cause,
        "mechanism": pattern.mechanism,
        "support": pattern.support,
        "cases": pattern.cases,
        "detail": pattern.detail,
    }


def _decision_reason(delta_in: int, delta_ho: int, accepted: bool) -> str:
    if accepted:
        return f"accepted: improved a split without regression (d_in={delta_in}, d_ho={delta_ho})"
    if delta_in < 0 or delta_ho < 0:
        return f"rejected: regressed a split (d_in={delta_in}, d_ho={delta_ho})"
    return f"rejected: no net improvement (d_in={delta_in}, d_ho={delta_ho})"
