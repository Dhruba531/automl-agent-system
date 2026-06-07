from pathlib import Path

from automl_agent.orchestrator import AutoMLOrchestrator


def test_end_to_end_iris_pipeline(tmp_path: Path) -> None:
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=0)
    report = orchestrator.run(output_dir=tmp_path, dataset="iris")

    assert report.model_bundle_path.exists()
    assert report.leaderboard
    assert report.dataset.task_type == "classification"
    assert "f1_macro" in report.best_metrics
    assert report.explainability is not None
    assert report.explainability.importances
    assert report.monitoring_baseline is not None
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "explainability.json").exists()
    assert (tmp_path / "monitoring_baseline.json").exists()
