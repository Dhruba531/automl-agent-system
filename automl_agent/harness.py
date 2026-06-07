from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from automl_agent.orchestrator import AutoMLOrchestrator
from automl_agent.types import TaskType


@dataclass(frozen=True)
class HarnessCase:
    name: str
    dataset: Optional[str] = None
    csv_path: Optional[Path] = None
    target: Optional[str] = None
    task_type: Optional[TaskType] = None
    workers: int = 2
    trials: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HarnessCase":
        if not payload.get("name"):
            raise ValueError("Each harness case must include a non-empty 'name'.")
        dataset = payload.get("dataset")
        csv_path = Path(payload["csv"]) if payload.get("csv") else None
        if dataset and csv_path:
            raise ValueError(f"Harness case '{payload['name']}' cannot set both dataset and csv.")
        if not dataset and not csv_path:
            dataset = "breast_cancer"
        task_type = payload.get("task")
        if task_type not in {None, "classification", "regression"}:
            raise ValueError(f"Harness case '{payload['name']}' has invalid task: {task_type}")
        return cls(
            name=str(payload["name"]),
            dataset=dataset,
            csv_path=csv_path,
            target=payload.get("target"),
            task_type=task_type,
            workers=int(payload.get("workers", 2)),
            trials=int(payload.get("trials", 0)),
        )


@dataclass
class HarnessResult:
    case_name: str
    status: str
    dataset: Optional[str]
    task_type: Optional[str]
    best_model: Optional[str]
    primary_metric: Optional[str]
    primary_score: Optional[float]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifact_dir: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


class ExperimentHarness:
    """Runs repeatable AutoML experiments and writes aggregate results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @classmethod
    def from_config_file(cls, config_path: Path, output_dir: Optional[Path] = None) -> tuple["ExperimentHarness", List[HarnessCase]]:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cases = [HarnessCase.from_dict(item) for item in payload.get("cases", [])]
        if not cases:
            raise ValueError("Harness config must include at least one case.")
        resolved_output = output_dir or Path(payload.get("output_dir", "artifacts/harness"))
        return cls(resolved_output), cases

    def run(self, cases: Iterable[HarnessCase], fail_fast: bool = False) -> List[HarnessResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: List[HarnessResult] = []
        for case in cases:
            result = self._run_case(case)
            results.append(result)
            self._write_outputs(results)
            if fail_fast and result.status == "failed":
                break
        return results

    def _run_case(self, case: HarnessCase) -> HarnessResult:
        start = time.perf_counter()
        artifact_dir = self.output_dir / case.name
        try:
            orchestrator = AutoMLOrchestrator(max_workers=case.workers, tuning_trials=case.trials)
            report = orchestrator.run(
                output_dir=artifact_dir,
                dataset=case.dataset,
                csv_path=case.csv_path,
                target=case.target,
                task_type=case.task_type,
            )
            primary_metric = self._primary_metric(report.dataset.task_type)
            return HarnessResult(
                case_name=case.name,
                status="passed",
                dataset=case.dataset or str(case.csv_path),
                task_type=report.dataset.task_type,
                best_model=report.best_model_name,
                primary_metric=primary_metric,
                primary_score=report.best_metrics.get(primary_metric),
                metrics=report.best_metrics,
                artifact_dir=str(artifact_dir),
                duration_seconds=round(time.perf_counter() - start, 4),
            )
        except Exception as exc:
            return HarnessResult(
                case_name=case.name,
                status="failed",
                dataset=case.dataset or str(case.csv_path),
                task_type=case.task_type,
                best_model=None,
                primary_metric=None,
                primary_score=None,
                artifact_dir=str(artifact_dir),
                duration_seconds=round(time.perf_counter() - start, 4),
                error=str(exc),
            )

    def _write_outputs(self, results: List[HarnessResult]) -> None:
        records = [asdict(result) for result in results]
        (self.output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        self._write_csv(results, self.output_dir / "results.csv")
        self._write_markdown(results, self.output_dir / "summary.md")

    def _write_csv(self, results: List[HarnessResult], path: Path) -> None:
        fieldnames = [
            "case_name",
            "status",
            "dataset",
            "task_type",
            "best_model",
            "primary_metric",
            "primary_score",
            "artifact_dir",
            "duration_seconds",
            "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = asdict(result)
                writer.writerow({field: row.get(field) for field in fieldnames})

    def _write_markdown(self, results: List[HarnessResult], path: Path) -> None:
        lines = [
            "# Harness Summary",
            "",
            "| Case | Status | Dataset | Task | Best Model | Metric | Score | Seconds |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
        for result in results:
            score = "" if result.primary_score is None else f"{result.primary_score:.6f}"
            lines.append(
                "| {case} | {status} | {dataset} | {task} | {model} | {metric} | {score} | {seconds:.2f} |".format(
                    case=result.case_name,
                    status=result.status,
                    dataset=result.dataset or "",
                    task=result.task_type or "",
                    model=result.best_model or "",
                    metric=result.primary_metric or "",
                    score=score,
                    seconds=result.duration_seconds,
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _primary_metric(self, task_type: TaskType) -> str:
        return "f1_macro" if task_type == "classification" else "rmse"

