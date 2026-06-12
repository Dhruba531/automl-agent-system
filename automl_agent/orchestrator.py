from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from automl_agent.agents import (
    DataAgent,
    DeploymentAgent,
    EvaluationAgent,
    ExplainabilityAgent,
    FeatureAgent,
    HyperparameterAgent,
    InsightAgent,
    MonitoringAgent,
    ModelSearchAgent,
)
from automl_agent.types import PipelineReport, TaskType

if TYPE_CHECKING:
    from automl_agent.self_harness.config import HarnessConfig


class AutoMLOrchestrator:
    def __init__(
        self,
        max_workers: int = 4,
        tuning_trials: int = 20,
        llm_connector=None,
        harness_config: "Optional[HarnessConfig]" = None,
    ) -> None:
        if harness_config is not None:
            tuning_trials = harness_config.tuning_trials
        self.harness_config = harness_config
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.model_agent = ModelSearchAgent(max_workers=max_workers, config=harness_config)
        self.evaluation_agent = EvaluationAgent()
        self.hyperparameter_agent = HyperparameterAgent(trials=tuning_trials)
        self.explainability_agent = ExplainabilityAgent()
        self.monitoring_agent = MonitoringAgent()
        self.deployment_agent = DeploymentAgent()
        self.insight_agent = InsightAgent(llm_connector)

    def run(
        self,
        output_dir: Path,
        dataset: Optional[str] = None,
        csv_path: Optional[Path] = None,
        target: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        user_prompt: Optional[str] = None,
    ) -> PipelineReport:
        output_dir.mkdir(parents=True, exist_ok=True)
        data = self.data_agent.load(dataset=dataset, csv_path=csv_path, target=target, task_type=task_type)
        feature_plan = self.feature_agent.plan(data)
        candidates = self.model_agent.search(data, feature_plan)
        failed_candidates = [
            {"name": result.name, "error": result.error or ""}
            for result in candidates
            if result.error is not None
        ]
        leaderboard = self.evaluation_agent.rank(candidates, data.task_type)
        tuned = self.hyperparameter_agent.tune(leaderboard[0], data, feature_plan)
        final_best = self._choose_final(leaderboard[0], tuned, data.task_type)
        explainability = self.explainability_agent.explain(final_best, data)
        monitoring_baseline = self.monitoring_agent.build_baseline(data)
        artifacts = self.deployment_agent.package(
            final_best,
            data,
            output_dir,
            explainability=explainability,
            monitoring_baseline=monitoring_baseline,
        )
        llm_summary = self.insight_agent.summarize(
            data, leaderboard, final_best, explainability, user_prompt=user_prompt
        )
        if llm_summary:
            (output_dir / "llm_summary.md").write_text(llm_summary, encoding="utf-8")

        report = PipelineReport(
            dataset=data.profile,
            leaderboard=leaderboard,
            best_model_name=final_best.name,
            best_metrics=final_best.metrics,
            tuned_metrics=tuned.metrics,
            best_cv_score=final_best.cv_score,
            failed_candidates=failed_candidates,
            explainability=explainability,
            monitoring_baseline=monitoring_baseline,
            artifact_dir=output_dir,
            model_bundle_path=artifacts["bundle"],
            llm_summary=llm_summary,
            notes=self._events(),
        )
        self._write_report(report, output_dir / "pipeline_report.json")
        return report

    def _choose_final(self, original, tuned, task_type: TaskType):
        if tuned is original:
            return original
        if original.cv_score is not None and tuned.cv_score is not None:
            return tuned if tuned.cv_score >= original.cv_score else original
        metric = self.evaluation_agent.primary_metric(task_type)
        original_score = original.metrics[metric]
        tuned_score = tuned.metrics.get(metric, original_score)
        if task_type == "classification":
            return tuned if tuned_score >= original_score else original
        return tuned if tuned_score <= original_score else original

    def _events(self) -> list[str]:
        agents = [
            self.data_agent,
            self.feature_agent,
            self.model_agent,
            self.evaluation_agent,
            self.hyperparameter_agent,
            self.explainability_agent,
            self.monitoring_agent,
            self.deployment_agent,
            self.insight_agent,
        ]
        return [f"{event.agent}: {event.message}" for agent in agents for event in agent.events]

    def _write_report(self, report: PipelineReport, path: Path) -> None:
        payload = asdict(report)
        payload["artifact_dir"] = str(report.artifact_dir)
        payload["model_bundle_path"] = str(report.model_bundle_path)
        payload["leaderboard"] = [
            {
                "name": result.name,
                "metrics": result.metrics,
                "cv_score": result.cv_score,
                "train_seconds": result.train_seconds,
                "error": result.error,
            }
            for result in report.leaderboard
        ]
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
