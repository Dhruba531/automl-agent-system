from __future__ import annotations

from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl_agent.agents.base import BaseAgent
from automl_agent.types import DataBundle, FeaturePlan


class FeatureAgent(BaseAgent):
    name = "Feature Agent"

    def plan(self, data: DataBundle) -> FeaturePlan:
        profile = data.profile
        self.log(
            f"Planned preprocessing for {len(profile.numeric_features)} numeric and "
            f"{len(profile.categorical_features)} categorical features."
        )
        return FeaturePlan(
            numeric_features=profile.numeric_features,
            categorical_features=profile.categorical_features,
            profile=profile,
        )

    def plan_and_build(self, data: DataBundle) -> Tuple[FeaturePlan, ColumnTransformer]:
        plan = self.plan(data)
        return plan, self.build_preprocessor(plan)

    def build_preprocessor(self, plan: FeaturePlan) -> ColumnTransformer:
        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", self._one_hot_encoder()),
            ]
        )
        transformers = []
        if plan.numeric_features:
            transformers.append(("numeric", numeric, plan.numeric_features))
        if plan.categorical_features:
            transformers.append(("categorical", categorical, plan.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _one_hot_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

