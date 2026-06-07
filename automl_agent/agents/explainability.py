from __future__ import annotations

from sklearn.inspection import permutation_importance

from automl_agent.agents.base import BaseAgent
from automl_agent.agents.evaluation import EvaluationAgent
from automl_agent.types import CandidateResult, DataBundle, ExplainabilityReport, FeatureImportance


class ExplainabilityAgent(BaseAgent):
    name = "Explainability Agent"

    def explain(self, model: CandidateResult, data: DataBundle, max_features: int = 12) -> ExplainabilityReport:
        scoring = "f1_macro" if data.task_type == "classification" else "neg_root_mean_squared_error"
        result = permutation_importance(
            model.estimator,
            data.X_test,
            data.y_test,
            scoring=scoring,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        ranked = sorted(
            zip(data.X_test.columns, result.importances_mean, result.importances_std),
            key=lambda item: item[1],
            reverse=True,
        )[:max_features]
        importances = [
            FeatureImportance(
                feature=str(feature),
                importance_mean=float(mean),
                importance_std=float(std),
            )
            for feature, mean, std in ranked
        ]
        primary_metric = EvaluationAgent().primary_metric(data.task_type)
        self.log(f"Computed permutation importance for top {len(importances)} features.")
        return ExplainabilityReport(
            method="permutation_importance",
            primary_metric=primary_metric,
            importances=importances,
        )

