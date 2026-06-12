"""Harness Proposal: turn failure evidence into bounded candidate edits.

Implements §3.3. The proposer is the *same* agent operating its current harness,
invoked in a proposer role. When an LLM connector is available it generates the
``K`` diverse-yet-minimal edits; otherwise a deterministic fallback maps failure
signatures to edits so the loop runs without a model. Both paths emit edits drawn
from the same bounded vocabulary and validated identically downstream.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from automl_agent.agents.model_search import all_base_candidate_names, all_extra_candidate_names
from automl_agent.self_harness.config import EDIT_OPS, HarnessConfig, HarnessEdit
from automl_agent.self_harness.evidence import FailurePattern

# Extras ordered by how often a stronger learner lifts a weak pool.
_EXTRA_PRIORITY = ["gradient_boosting", "knn", "ada_boost", "gaussian_nb", "lasso"]


class HarnessProposer:
    def __init__(self, connector=None) -> None:
        self.connector = connector

    def propose(
        self,
        config: HarnessConfig,
        patterns: Sequence[FailurePattern],
        width: int,
        attempted: Optional[set] = None,
    ) -> List[HarnessEdit]:
        attempted = attempted or set()
        edits: List[HarnessEdit] = []
        if self.connector is not None:
            try:
                edits = self._propose_with_llm(config, patterns, width)
            except Exception:
                edits = []
        if not edits:
            edits = self._propose_deterministic(config, patterns, width)
        return self._distinct(edits, config, attempted, width)

    # ------------------------------------------------------------------ LLM
    def _propose_with_llm(
        self, config: HarnessConfig, patterns: Sequence[FailurePattern], width: int
    ) -> List[HarnessEdit]:
        content = self.connector.chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._prompt(config, patterns, width)},
            ]
        )
        return self._parse(content)

    def _prompt(self, config: HarnessConfig, patterns: Sequence[FailurePattern], width: int) -> str:
        lines = [
            "Current harness configuration:",
            json.dumps(config.to_dict(), indent=2),
            "",
            "Editable surfaces and allowed operations:",
            f"- enable_candidate(name): add an optional model. Options: {sorted(all_extra_candidate_names())}",
            f"- disable_candidate(name): drop a model. Active base models: {sorted(all_base_candidate_names())}",
            "- set_cv_splits(int): change cross-validation folds (2-10).",
            "- set_tuning_trials(int): change Optuna tuning budget (0-50).",
            "",
            "Verifier-grounded failure patterns (highest support first):",
        ]
        for pattern in patterns:
            lines.append(
                f"- cause={pattern.cause}, mechanism={pattern.mechanism}, support={pattern.support}, "
                f"cases={pattern.cases}: {pattern.detail}"
            )
        lines += [
            "",
            f"Propose {width} materially distinct, minimal edits (one operation each), each tied to a "
            "failure pattern above. Return ONLY a JSON array of objects with keys: op, value, "
            "target_pattern, rationale.",
        ]
        return "\n".join(lines)

    def _parse(self, content: str) -> List[HarnessEdit]:
        text = content.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        edits: List[HarnessEdit] = []
        for item in raw:
            op = item.get("op")
            if op not in EDIT_OPS:
                continue
            try:
                edits.append(
                    HarnessEdit(
                        op=op,
                        value=item.get("value"),
                        target_pattern=str(item.get("target_pattern", "")),
                        rationale=str(item.get("rationale", "")),
                    )
                )
            except ValueError:
                continue
        return edits

    # -------------------------------------------------------- deterministic
    def _propose_deterministic(
        self, config: HarnessConfig, patterns: Sequence[FailurePattern], width: int
    ) -> List[HarnessEdit]:
        edits: List[HarnessEdit] = []
        for pattern in patterns:
            edit = self._edit_for_pattern(config, pattern, edits)
            if edit is not None:
                edits.append(edit)
            if len(edits) >= width:
                break
        if not edits:
            edits = self._fallback_edits(config)
        return edits

    def _edit_for_pattern(
        self, config: HarnessConfig, pattern: FailurePattern, chosen: List[HarnessEdit]
    ) -> Optional[HarnessEdit]:
        sig = f"{pattern.cause}:{pattern.mechanism}"
        if pattern.cause == "candidate_error":
            model = pattern.mechanism
            if model not in config.disabled_candidates:
                return HarnessEdit(
                    "disable_candidate",
                    model,
                    target_pattern=sig,
                    rationale=f"Model '{model}' repeatedly errors; drop it from the search.",
                )
        if pattern.cause in ("weak_model_pool", "below_threshold"):
            candidate = self._next_extra(config, chosen)
            if candidate is not None:
                return HarnessEdit(
                    "enable_candidate",
                    candidate,
                    target_pattern=sig,
                    rationale=f"Pool underperforms; add stronger learner '{candidate}'.",
                )
            if config.tuning_trials < 20:
                return HarnessEdit(
                    "set_tuning_trials",
                    20,
                    target_pattern=sig,
                    rationale="Pool exhausted; raise tuning budget to lift the selected model.",
                )
        if pattern.cause == "no_cv_score" and config.cv_splits > 2:
            return HarnessEdit(
                "set_cv_splits",
                2,
                target_pattern=sig,
                rationale="Folds exceed the smallest class; reduce to 2 to obtain a CV score.",
            )
        return None

    def _next_extra(self, config: HarnessConfig, chosen: List[HarnessEdit]) -> Optional[str]:
        taken = {edit.value for edit in chosen if edit.op == "enable_candidate"}
        for name in _EXTRA_PRIORITY:
            if name not in all_extra_candidate_names():
                continue
            if name in config.enabled_extra_candidates or name in taken:
                continue
            return name
        return None

    def _fallback_edits(self, config: HarnessConfig) -> List[HarnessEdit]:
        candidate = self._next_extra(config, [])
        if candidate is not None:
            return [
                HarnessEdit(
                    "enable_candidate",
                    candidate,
                    target_pattern="exploration",
                    rationale="No actionable failure pattern; explore a stronger learner.",
                )
            ]
        return []

    # --------------------------------------------------------------- shared
    def _distinct(
        self, edits: List[HarnessEdit], config: HarnessConfig, attempted: set, width: int
    ) -> List[HarnessEdit]:
        """Keep materially distinct edits that change the harness and are new.

        Distinctness is by (op, value): two branches must not restate the same
        edit. Different operations on the same surface (e.g. enabling one model
        and disabling another) are kept, since they target different mechanisms.
        """
        seen_keys: set = set()
        result: List[HarnessEdit] = []
        for edit in edits:
            if edit.key() in attempted or edit.key() in seen_keys:
                continue
            if config.apply(edit) == config:  # no-op edit changes no surface
                continue
            seen_keys.add(edit.key())
            result.append(edit)
            if len(result) >= width:
                break
        return result


_SYSTEM_PROMPT = (
    "You improve an AutoML agent's harness. Diagnose recurring, verifier-grounded failure patterns "
    "and propose minimal, bounded edits to the harness configuration. Each edit must target one "
    "failure mechanism and touch only the surface needed. Do not rewrite the architecture."
)
