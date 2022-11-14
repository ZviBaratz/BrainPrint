import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from brainprint.recon_all.execution_configuration import ExecutionConfiguration
from brainprint.recon_all.models.model_configurations import (
    MODEL_CONFIGURATIONS,
    EstimatorType,
)
from brainprint.recon_all.results import ReconAllResults
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger()
HERE: Path = Path(__file__).parent


class EstimatorSearch:
    def __init__(
        self,
        results: ReconAllResults,
        target: str = None,
        model_configurations: Optional[
            Dict[BaseEstimator, Dict[str, Any]]
        ] = None,
        configurations: Optional[List[ExecutionConfiguration]] = None,
        scaling: Optional[List[BaseEstimator]] = None,
        random_state: Optional[int] = None,
        scoring: Optional[str] = "roc_auc",
        cv: Optional[int] = 5,
    ) -> None:
        self.results = results
        self.target = target
        self.estimator_type = self.infer_estimator_type()

        self.model_configurations = (
            MODEL_CONFIGURATIONS.copy()
            if model_configurations is None
            else model_configurations
        )
        self.configurations = (
            results.configuration if configurations is None else configurations
        )
        self.scaling = (
            [StandardScaler, RobustScaler] if scaling is None else scaling
        )
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state

    def infer_estimator_type(self) -> EstimatorType:
        target_values = self.results.context[self.target]
        if target_values.dtype.name == "category":
            return EstimatorType.CLASSIFICATION
        return EstimatorType.REGRESSION

    def get_pipeline_path(
        self,
        estimator: BaseEstimator,
        configuration: ExecutionConfiguration,
        scaler: BaseEstimator,
    ) -> Path:
        configuration_id = configuration.name.lower()
        scaler_name = scaler.__name__.lower()
        name = f"{configuration_id}_{scaler_name}_{estimator.__name__}.pkl"
        return HERE / "pipelines" / self.target / name

    def run(self, force: bool = False) -> None:
        scores = {"Train": {}, "Test": {}}
        model_configurations = self.model_configurations[self.estimator_type]
        for configuration in self.configurations:
            logger.info(f"Starting {configuration.value} model search.")
            X_train, X_test, y_train, y_test = self.results.split(
                execution_configuration=configuration,
                target=self.target,
                single_mode="last",
                random_state=self.random_state,
            )
            for scaler in self.scaling:
                logger.info(
                    f"Starting model search with {scaler.__name__} transformation."
                )
                for estimator, param_grid in model_configurations.items():
                    destination = self.get_pipeline_path(
                        estimator=estimator,
                        configuration=configuration,
                        scaler=scaler,
                    )
                    if destination.exists() and not force:
                        name = destination.name.split(".")[0].replace("_", " ")
                        logger.info(
                            f"Found existing {name} pipeline, skipping..."
                        )
                        continue
                    logger.info(
                        f"Creating pipeline for {estimator.__name__} with {scaler.__name__} scaling."
                    )
                    steps = [("scaling", scaler()), ("clf", estimator())]
                    pipeline = Pipeline(steps=steps)
                    param_grid = {
                        f"clf__{key}": value
                        for key, value in param_grid.items()
                    }
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=-1,
                        return_train_score=True,
                    )
                    logger.info(f"Fitting {estimator.__name__} model...")
                    if (
                        self.estimator_type == EstimatorType.REGRESSION
                        and isinstance(y_train, pd.Series)
                    ):
                        target_scaler = scaler()
                        y_train = target_scaler.fit_transform(
                            y_train.values.reshape(-1, 1)
                        )
                        y_test = target_scaler.transform(
                            y_test.values.reshape(-1, 1)
                        )
                    grid_search.fit(X_train, y_train)
                    logger.info(f"Best parameters: {grid_search.best_params_}")
                    logger.info(f"Best score: {grid_search.best_score_}")
                    test_score = grid_search.score(X_test, y_test)
                    logger.info(f"Test score: {test_score}")
                    index = (
                        configuration.value,
                        estimator.__name__,
                        scaler.__name__,
                    )
                    scores["Train"][index] = grid_search.best_score_
                    scores["Test"][index] = test_score
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        f"Saving GridSearchCV object to {destination}..."
                    )
                    joblib.dump(grid_search, destination)
                    logger.info(
                        f"Successfully saved pipeline to {destination}."
                    )
                    cv_destination = destination.with_suffix(".csv")
                    logger.info(
                        f"Saving cross-validation results to {destination}..."
                    )
                    pd.DataFrame(grid_search.cv_results_).to_csv(
                        cv_destination
                    )
                    logger.info(
                        f"Successfully saved cross-validation results to {cv_destination}."  # noqa: E501
                    )
        pd.DataFrame(scores).to_csv(destination.parent / "scores.csv")
        return scores

    def load_pipeline(
        self,
        estimator: BaseEstimator,
        configuration: ExecutionConfiguration,
        scaler: BaseEstimator,
    ) -> Dict[str, Any]:
        path = self.get_pipeline_path(estimator, configuration, scaler)
        if path.exists():
            return joblib.load(path)

    def get_scores_csv_path(self) -> Path:
        return HERE / "pipelines" / self.target / "scores.csv"

    def read_pipeline_scores_df(self) -> pd.DataFrame:
        csv_path = self.get_scores_csv_path()
        try:
            return pd.read_csv(
                csv_path,
                names=[
                    "Configuration",
                    "Estimator",
                    "Scaler",
                    "Train",
                    "Test",
                ],
                header=0,
            )
        except FileNotFoundError:
            scores = self.calculate_pipeline_scores_df()
            scores.to_csv(csv_path)
            return scores

    def calculate_pipeline_scores(
        self,
        estimator: BaseEstimator,
        configuration: ExecutionConfiguration,
        scaler: BaseEstimator,
    ) -> Dict[str, Any]:
        pipeline = self.load_pipeline(estimator, configuration, scaler)
        if pipeline is None:
            print(
                f"No pipeline found for {estimator.__name__} with configuration {configuration.value} and {scaler.__name__}!"
            )
            return None, None
        _, X_test, _, y_test = self.results.split(
            execution_configuration=configuration,
            target=self.target,
            single_mode="last",
            random_state=self.random_state,
        )
        test_score = pipeline.score(X_test, y_test)
        return pipeline.best_score_, test_score

    def calculate_pipeline_scores_df(self):
        scores = {"train": {}, "test": {}}
        for estimator in self.model_configurations[self.estimator_type]:
            for configuration in self.configurations:
                for scaler in self.scaling:
                    train_score, test_score = self.calculate_pipeline_scores(
                        estimator=estimator,
                        configuration=configuration,
                        scaler=scaler,
                    )
                    index = (
                        configuration.value,
                        estimator.__name__,
                        scaler.__name__,
                    )
                    scores["train"][index] = train_score
                    scores["test"][index] = test_score
        df = pd.DataFrame(scores)
        df.index.names = ["configuration", "estimator", "scaler"]
        return df.sort_index()

    def read_estimator_results(self, estimator, configuration, scaler):
        path = self.get_pipeline_path(estimator, configuration, scaler)
        if path.exists():
            csv_path = path.with_suffix(".csv")
            df = pd.read_csv(csv_path, index_col=0)
            df["estimator"] = estimator.__name__
            df["configuration"] = configuration.value
            df["scaler"] = scaler.__name__
            return df

    def read_cv_results(self):
        dfs = []
        model_configurations = self.model_configurations[self.estimator_type]
        for estimator in model_configurations.keys():
            for configuration in self.configurations:
                for scaler in self.scaling:
                    df = self.read_estimator_results(
                        estimator, configuration, scaler
                    )
                    if df is not None:
                        dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)
