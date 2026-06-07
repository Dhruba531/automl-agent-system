from pathlib import Path

from automl_agent.harness import ExperimentHarness, HarnessCase


def test_harness_runs_cases_and_writes_outputs(tmp_path: Path) -> None:
    harness = ExperimentHarness(tmp_path)
    results = harness.run([HarnessCase(name="iris-case", dataset="iris", workers=2, trials=0)])

    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].primary_metric == "f1_macro"
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "iris-case" / "model_bundle.joblib").exists()


def test_harness_config_loads_cases(tmp_path: Path) -> None:
    config = tmp_path / "harness.json"
    config.write_text(
        """
        {
          "output_dir": "ignored",
          "cases": [
            {"name": "iris-fast", "dataset": "iris", "workers": 1, "trials": 0}
          ]
        }
        """,
        encoding="utf-8",
    )

    harness, cases = ExperimentHarness.from_config_file(config, output_dir=tmp_path / "out")

    assert harness.output_dir == tmp_path / "out"
    assert cases == [HarnessCase(name="iris-fast", dataset="iris", workers=1, trials=0)]


def test_harness_records_failed_case(tmp_path: Path) -> None:
    harness = ExperimentHarness(tmp_path)
    results = harness.run([HarnessCase(name="bad-case", dataset="not-a-dataset", workers=1, trials=0)])

    assert results[0].status == "failed"
    assert results[0].error
    assert (tmp_path / "results.json").exists()
