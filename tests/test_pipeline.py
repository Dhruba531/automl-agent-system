from pathlib import Path

from automl_agent.orchestrator import AutoMLOrchestrator


def test_end_to_end_iris_pipeline(tmp_path: Path) -> None:
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=0)
    report = orchestrator.run(output_dir=tmp_path, dataset="iris")

    assert report.model_bundle_path.exists()
    assert report.leaderboard
    assert report.dataset.task_type == "classification"
    assert "f1_macro" in report.best_metrics
    assert "roc_auc" in report.best_metrics
    assert report.explainability is not None
    assert report.explainability.importances
    assert report.monitoring_baseline is not None
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "explainability.json").exists()
    assert (tmp_path / "monitoring_baseline.json").exists()

    names = {result.name for result in report.leaderboard}
    assert "hist_gradient_boosting" in names
    assert all(result.cv_score is not None for result in report.leaderboard)
    scores = [result.cv_score for result in report.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_end_to_end_regression_pipeline(tmp_path: Path) -> None:
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=0)
    report = orchestrator.run(output_dir=tmp_path, dataset="diabetes")

    assert report.model_bundle_path.exists()
    assert report.dataset.task_type == "regression"
    assert "rmse" in report.best_metrics
    assert report.best_metrics["rmse"] > 0
    assert all(result.cv_score is not None for result in report.leaderboard)


def test_tuning_records_cv_score(tmp_path: Path) -> None:
    orchestrator = AutoMLOrchestrator(max_workers=2, tuning_trials=2)
    report = orchestrator.run(output_dir=tmp_path, dataset="iris")

    assert report.tuned_metrics
    assert report.best_metrics
