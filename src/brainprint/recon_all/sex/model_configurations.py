from enum import Enum
from typing import Any, Dict

import numpy as np
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
