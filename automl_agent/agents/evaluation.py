from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from automl_agent.agents.base import BaseAgent
from automl_agent.types import CandidateResult, DataBundle, TaskType


class EvaluationAgent(BaseAgent):
    name = "Evaluation Agent"

    def evaluate(self, name: str, estimator, data: DataBundle) -> CandidateResult:
        predictions = estimator.predict(data.X_test)
        metrics = self.metrics(data.task_type, data.y_test, predictions, estimator, data.X_test)
        return CandidateResult(name=name, estimator=estimator, metrics=metrics, train_seconds=0.0)

    def metrics(self, task_type: TaskType, y_true, predictions, estimator=None, X_test=None) -> Dict[str, float]:
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_true, predictions)),
                "f1_macro": float(f1_score(y_true, predictions, average="macro")),
            }
            if estimator is not None and X_test is not None:
                try:
                    probabilities = estimator.predict_proba(X_test)
                    if probabilities.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y_true,
                                probabilities,
                                multi_class="ovr",
                                average="macro",
                                labels=estimator.classes_,
                            )
                        )
                except Exception:
                    pass
            return metrics

        rmse = root_mean_squared_error(y_true, predictions)
        return {
            "rmse": float(rmse),
            "mae": float(mean_absolute_error(y_true, predictions)),
            "r2": float(r2_score(y_true, predictions)),
        }

    def rank(self, results: Iterable[CandidateResult], task_type: TaskType) -> List[CandidateResult]:
        successful = [result for result in results if result.error is None]
        if not successful:
            raise RuntimeError("No candidate models trained successfully.")
        if any(result.cv_score is not None for result in successful):
            ranked = sorted(
                successful,
                key=lambda result: -np.inf if result.cv_score is None else result.cv_score,
                reverse=True,
            )
            self.log(f"Ranked {len(ranked)} successful candidates by cross-validated {self.scoring(task_type)}.")
            return ranked
        key = self.primary_metric(task_type)
        reverse = task_type == "classification"
        missing_value = -np.inf if reverse else np.inf
        ranked = sorted(successful, key=lambda result: result.metrics.get(key, missing_value), reverse=reverse)
        self.log(f"Ranked {len(ranked)} successful candidates by test-set {key}.")
        return ranked

    def primary_metric(self, task_type: TaskType) -> str:
        return "f1_macro" if task_type == "classification" else "rmse"

    def scoring(self, task_type: TaskType) -> str:
        """sklearn scoring string used for model selection; higher is always better."""
        return "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"
