from pathlib import Path

import pytest

from automl_agent.self_harness import (
    HarnessCase,
    HarnessConfig,
    HarnessEdit,
    HarnessProposer,
    SelfHarness,
    build_evidence_bundle,
)
from automl_agent.self_harness.evidence import CaseOutcome, SplitResult
from automl_agent.self_harness import loop as loop_module


# --------------------------------------------------------------------- config
def test_config_apply_enable_disable_and_noop() -> None:
    config = HarnessConfig()
    enabled = config.apply(HarnessEdit("enable_candidate", "gradient_boosting"))
    assert "gradient_boosting" in enabled.enabled_extra_candidates
    # Re-applying the same enable is a no-op (used to reject empty edits).
    assert enabled.apply(HarnessEdit("enable_candidate", "gradient_boosting")) == enabled

    disabled = config.apply(HarnessEdit("disable_candidate", "svc_rbf"))
    assert "svc_rbf" in disabled.disabled_candidates
    assert config.apply(HarnessEdit("set_cv_splits", 5)).cv_splits == 5


def test_unknown_edit_op_rejected() -> None:
    with pytest.raises(ValueError):
        HarnessEdit("rewrite_everything", 1)


# ------------------------------------------------------------------- evidence
def test_evidence_clusters_by_signature_and_orders_by_support() -> None:
    outcomes = [
        CaseOutcome("a", False, 0.80, 0.95, "svc_rbf", [{"name": "knn", "error": "boom"}]),
        CaseOutcome("b", False, 0.81, 0.95, "logistic_regression", [{"name": "knn", "error": "boom"}]),
        CaseOutcome("c", True, 0.99, 0.95, "random_forest", []),
    ]
    patterns = build_evidence_bundle(SplitResult(passed=1, total=3, outcomes=outcomes))

    # The shared candidate_error('knn') cluster has support 2 and ranks first.
    assert patterns[0].cause == "candidate_error"
    assert patterns[0].mechanism == "knn"
    assert patterns[0].support == 2
    assert patterns[0].cases == ["a", "b"]
    # Passing case 'c' contributes no failure signature.
    assert all("c" not in pattern.cases for pattern in patterns)


# ------------------------------------------------------------------- proposer
def test_proposer_emits_distinct_minimal_edits() -> None:
    outcomes = [
        CaseOutcome("a", False, 0.80, 0.95, "svc_rbf", [{"name": "knn", "error": "boom"}]),
        CaseOutcome("b", False, 0.81, 0.95, "logistic_regression", []),
    ]
    patterns = build_evidence_bundle(SplitResult(0, 2, outcomes))
    edits = HarnessProposer().propose(HarnessConfig(), patterns, width=3)

    keys = [edit.key() for edit in edits]
    assert len(keys) == len(set(keys))  # materially distinct
    assert any(edit.op == "disable_candidate" and edit.value == "knn" for edit in edits)
    assert any(edit.op == "enable_candidate" for edit in edits)
    # Every edit is tied to a mined failure pattern.
    assert all(edit.target_pattern for edit in edits)


def test_proposer_skips_already_attempted_edits() -> None:
    patterns = build_evidence_bundle(
        SplitResult(0, 1, [CaseOutcome("a", False, 0.80, 0.95, "svc_rbf", [])])
    )
    attempted = {("enable_candidate", "gradient_boosting")}
    edits = HarnessProposer().propose(HarnessConfig(), patterns, width=3, attempted=attempted)
    assert all(edit.key() != ("enable_candidate", "gradient_boosting") for edit in edits)


# ----------------------------------------------------- acceptance rule (loop)
def _make_split(cases, predicate) -> SplitResult:
    outcomes = []
    for case in cases:
        passed = predicate(case)
        outcomes.append(
            CaseOutcome(
                case_name=case.name,
                passed=passed,
                score=0.99 if passed else 0.50,
                threshold=case.pass_threshold,
                best_model="logistic_regression",
                failed_candidates=[],
            )
        )
    return SplitResult(passed=sum(o.passed for o in outcomes), total=len(cases), outcomes=outcomes)


def test_loop_accepts_edit_that_improves_without_regression(tmp_path: Path, monkeypatch) -> None:
    held_in = [HarnessCase("hard", pass_threshold=0.95, dataset="iris")]
    held_out = [HarnessCase("safe", pass_threshold=0.10, dataset="wine")]

    def fake_evaluate(config, cases, workdir, max_workers=2):
        if cases[0].name == "hard":
            # Held-in passes only once the booster is enabled.
            return _make_split(cases, lambda c: "gradient_boosting" in config.enabled_extra_candidates)
        return _make_split(cases, lambda c: True)  # held-out always passes

    monkeypatch.setattr(loop_module, "evaluate", fake_evaluate)
    loop = SelfHarness(held_in, held_out, tmp_path, proposal_width=2, rounds=1)
    result = loop.run()

    assert result.initial_passed_in == 0
    assert result.final_passed_in == 1
    assert "gradient_boosting" in result.final_config["enabled_extra_candidates"]
    accepted = [c for r in result.rounds for c in r.candidates if c.accepted]
    assert any(c.edit["op"] == "enable_candidate" for c in accepted)
    assert (tmp_path / "lineage.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_loop_rejects_edit_that_regresses_other_split(tmp_path: Path, monkeypatch) -> None:
    held_in = [HarnessCase("hard", pass_threshold=0.95, dataset="iris")]
    held_out = [HarnessCase("safe", pass_threshold=0.95, dataset="wine")]

    def fake_evaluate(config, cases, workdir, max_workers=2):
        enabled = "gradient_boosting" in config.enabled_extra_candidates
        if cases[0].name == "hard":
            # Enabling helps held-in...
            return _make_split(cases, lambda c: enabled)
        # ...but regresses held-out: it passes only when NOT enabled.
        return _make_split(cases, lambda c: not enabled)

    monkeypatch.setattr(loop_module, "evaluate", fake_evaluate)
    loop = SelfHarness(held_in, held_out, tmp_path, proposal_width=1, rounds=1)
    result = loop.run()

    # The conservative rule forbids trading one split for another.
    assert result.final_config["enabled_extra_candidates"] == []
    assert result.final_passed_in == 0
    rejected = [c for r in result.rounds for c in r.candidates if not c.accepted]
    assert any("regressed" in c.reason for c in rejected)


def test_loop_requires_held_out() -> None:
    with pytest.raises(ValueError):
        SelfHarness([HarnessCase("a", 0.9, dataset="iris")], [], Path("/tmp/x"))


# --------------------------------------------------------------- real pipeline
def test_self_harness_real_run_writes_lineage(tmp_path: Path) -> None:
    held_in = [HarnessCase("iris", pass_threshold=0.999, dataset="iris")]  # unreachable -> mines weakness
    held_out = [HarnessCase("wine", pass_threshold=0.10, dataset="wine")]
    loop = SelfHarness(held_in, held_out, tmp_path, proposal_width=1, rounds=1, max_workers=2)
    result = loop.run()

    assert result.total_in == 1 and result.total_ho == 1
    assert result.rounds[0].failure_patterns  # a real weakness was mined
    assert (tmp_path / "lineage.json").exists()
