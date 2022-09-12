from enum import Enum
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC, SVR


class EstimatorType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"


# Categorical estimators
LOGISTIC_REGRESSION_CONFIGURATION: Dict[str, Any] = {
    "penalty": ["elasticnet"],
    "solver": ["saga"],
    "random_state": [0],
    "n_jobs": [-1],
    "l1_ratio": np.linspace(0, 1, 11),
    "C": np.logspace(-5, 5, 11),
    "max_iter": [10_000],
}
RANDOM_FOREST_CONFIGURATION: Dict[str, Any] = {
    "n_estimators": np.arange(80, 180, 20),
    "max_depth": np.arange(1, 15, 2),
    "min_samples_split": np.arange(2, 8, 2),
    "min_samples_leaf": np.arange(1, 4),
    "random_state": [0],
    "n_jobs": [-1],
}
SVC_CONFIGURATION: Dict[str, Any] = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": np.logspace(-5, 5, 11),
    "random_state": [0],
}

# Numeric estimators
ELASTIC_NET_CONFIGURATION: Dict[str, Any] = {
    "alpha": [0.1, 0.5, 1, 5, 10, 50, 100],
    "l1_ratio": np.linspace(0, 1, 11),
    "random_state": [0],
    "selection": ["random"],
    "max_iter": [10_000_000],
}
SVR_CONFIGURATION: Dict[str, Any] = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": np.logspace(-5, 5, 11),
    "epsilon": np.logspace(-5, 5, 11),
}

MODEL_CONFIGURATIONS: Dict[
    EstimatorType, Dict[BaseEstimator, Dict[str, Any]]
] = {
    EstimatorType.CLASSIFICATION: {
        LogisticRegression: LOGISTIC_REGRESSION_CONFIGURATION,
        RandomForestClassifier: RANDOM_FOREST_CONFIGURATION,
        SVC: SVC_CONFIGURATION,
    },
    EstimatorType.REGRESSION: {
        ElasticNet: ELASTIC_NET_CONFIGURATION,
        SVR: SVR_CONFIGURATION,
    },
}


def create_configurations_df(configuration: Dict = None) -> pd.DataFrame:
    if configuration is None:
        configuration = MODEL_CONFIGURATIONS
    table = pd.DataFrame(
        [
            {
                "Estimator": estimator.__name__,
                "Parameter": parameter,
                "Values": values,
            }
            for configurations in configuration.values()
            for estimator, parameters in configurations.items()
            for parameter, values in parameters.items()
        ]
    )
    return table.set_index(["Estimator", "Parameter"]).sort_index()


def format_parameter_list(value: List[Any]) -> str:
    if isinstance(value[0], str):
        return ", ".join(value)
    return ", ".join(map(str, np.round(value, 2)))


def create_configurations_table(configuration: Dict = None) -> str:
    if configuration is None:
        configuration = MODEL_CONFIGURATIONS
    table = create_configurations_df(configuration)
    table["Values"] = table["Values"].apply(format_parameter_list)
    return table
