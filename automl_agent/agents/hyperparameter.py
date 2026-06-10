from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from automl_agent.agents.base import BaseAgent
from automl_agent.agents.evaluation import EvaluationAgent
from automl_agent.agents.feature import FeatureAgent
from automl_agent.types import CandidateResult, DataBundle, FeaturePlan


class HyperparameterAgent(BaseAgent):
    name = "Hyperparameter Agent"

    def __init__(self, trials: int = 20, random_state: int = 42) -> None:
        super().__init__()
        self.trials = trials
        self.random_state = random_state
        self.evaluator = EvaluationAgent()

    def tune(self, best: CandidateResult, data: DataBundle, features: FeaturePlan) -> CandidateResult:
        if self.trials <= 0:
            self.log("Skipping tuning because trials was set to 0.")
            return best
        try:
            import optuna

            return self._tune_with_optuna(optuna, best, data, features)
        except Exception as exc:
            self.log(f"Optuna unavailable or failed ({exc}); falling back to RandomizedSearchCV.")
            return self._tune_with_random_search(best, data, features)

    def _tune_with_optuna(self, optuna, best: CandidateResult, data: DataBundle, features: FeaturePlan) -> CandidateResult:
        scoring = self.evaluator.scoring(data.task_type)
        cv = self._cv(data)

        def objective(trial):
            estimator = self._suggest_estimator(trial, best.name, data.task_type)
            pipeline = Pipeline(
                [
                    ("preprocess", FeatureAgent().build_preprocessor(features)),
                    ("model", estimator),
                ]
            )
            scores = cross_val_score(pipeline, data.X_train, data.y_train, cv=cv, scoring=scoring, n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.trials, show_progress_bar=False)
        tuned_estimator = self._estimator_from_params(best.name, data.task_type, study.best_params)
        pipeline = Pipeline(
            [
                ("preprocess", FeatureAgent().build_preprocessor(features)),
                ("model", tuned_estimator),
            ]
        )
        pipeline.fit(data.X_train, data.y_train)
        tuned = self.evaluator.evaluate(f"{best.name}_tuned", pipeline, data)
        tuned.cv_score = float(study.best_value)
        tuned.train_seconds = best.train_seconds
        self.log(f"Tuned {best.name} with Optuna over {self.trials} trials.")
        return tuned

    def _tune_with_random_search(self, best: CandidateResult, data: DataBundle, features: FeaturePlan) -> CandidateResult:
        params = self._random_search_space(best.name, data.task_type)
        if not params:
            return best
        pipeline = Pipeline(
            [
                ("preprocess", FeatureAgent().build_preprocessor(features)),
                ("model", clone(best.estimator.named_steps["model"])),
            ]
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            n_iter=min(self.trials, 12),
            scoring=self.evaluator.scoring(data.task_type),
            cv=self._cv(data),
            random_state=self.random_state,
            n_jobs=1,
        )
        search.fit(data.X_train, data.y_train)
        tuned = self.evaluator.evaluate(f"{best.name}_tuned", search.best_estimator_, data)
        tuned.cv_score = float(search.best_score_)
        self.log(f"Tuned {best.name} with RandomizedSearchCV.")
        return tuned

    def _cv(self, data: DataBundle):
        if data.task_type == "classification":
            return StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=3, shuffle=True, random_state=self.random_state)

    def _suggest_estimator(self, trial, name: str, task_type: str):
        params = self._suggest_params(trial, name, task_type)
        return self._estimator_from_params(name, task_type, params)

    def _suggest_params(self, trial, name: str, task_type: str) -> dict:
        if name == "logistic_regression":
            return {"C": trial.suggest_float("C", 0.01, 20.0, log=True)}
        if name == "svc_rbf":
            return {
                "C": trial.suggest_float("C", 0.1, 50.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0001, 1.0, log=True),
            }
        if name == "svr_rbf":
            return {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0001, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            }
        if name == "ridge":
            return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
        if name in {"random_forest", "extra_trees"}:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 320),
                "max_depth": trial.suggest_int("max_depth", 2, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        if name == "hist_gradient_boosting":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                "max_iter": trial.suggest_int("max_iter", 80, 400),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.001, 10.0, log=True),
            }
        return {}

    def _estimator_from_params(self, name: str, task_type: str, params: dict):
        if name == "logistic_regression":
            return LogisticRegression(max_iter=2000, class_weight="balanced", **params)
        if name == "svc_rbf":
            return SVC(kernel="rbf", probability=True, class_weight="balanced", **params)
        if name == "svr_rbf":
            return SVR(kernel="rbf", **params)
        if name == "ridge":
            return Ridge(**params)
        if name == "random_forest" and task_type == "classification":
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        if name == "random_forest":
            return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        if name == "extra_trees" and task_type == "classification":
            return ExtraTreesClassifier(random_state=42, n_jobs=-1, **params)
        if name == "extra_trees":
            return ExtraTreesRegressor(random_state=42, n_jobs=-1, **params)
        if name == "hist_gradient_boosting" and task_type == "classification":
            return HistGradientBoostingClassifier(random_state=42, **params)
        if name == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(random_state=42, **params)
        raise ValueError(f"Unsupported estimator for tuning: {name}")

    def _random_search_space(self, name: str, task_type: str) -> dict:
        if name in {"random_forest", "extra_trees"}:
            return {
                "model__n_estimators": [80, 120, 200, 320],
                "model__max_depth": [None, 4, 8, 16, 24],
                "model__min_samples_leaf": [1, 2, 4, 8],
            }
        if name == "hist_gradient_boosting":
            return {
                "model__learning_rate": [0.03, 0.1, 0.3],
                "model__max_iter": [100, 200, 300],
                "model__max_leaf_nodes": [15, 31, 63],
            }
        if name in {"svc_rbf", "svr_rbf"}:
            return {"model__C": [0.1, 1.0, 10.0, 50.0], "model__gamma": ["scale", 0.01, 0.001]}
        if name == "logistic_regression":
            return {"model__C": [0.01, 0.1, 1.0, 10.0, 20.0]}
        if name == "ridge":
            return {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        return {}
