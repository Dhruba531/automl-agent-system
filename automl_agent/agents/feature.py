from __future__ import annotations

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

    def _one_hot_encoder(self, max_categories: int = 32) -> OneHotEncoder:
        # max_categories groups rare levels of high-cardinality columns into an
        # infrequent bucket so one-hot encoding cannot explode the feature space.
        return OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, max_categories=max_categories)

