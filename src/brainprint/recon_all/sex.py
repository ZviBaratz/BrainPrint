from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from brainprint.recon_all.read import DEFAULT_ATLAS, read_results
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

THE_BASE_PROTOCOL = {
    "Corrected": True,
    "Inversion Time": 1000,
    "Echo Time": 2.78,
    "Repetition Time": 2400,
    "Spatial Resolution": "[0.9, 0.9, 0.9]",
}

HCP_PROTOCOL = {
    "Corrected": False,
    "Inversion Time": 1000,
    "Echo Time": 2.14,
    "Repetition Time": 2400,
    "Spatial Resolution": "[0.7, 0.7, 0.7]",
}

ATLASES: Tuple[str] = "Desikan-Killiany", "Destrieux"


class Dataset(Enum):
    HCP = "Human Connectome Project"
    BASE = "The Base Protocol"


DATASET_SEQUENCE: Dict[Dataset, Dict[str, Any]] = {
    Dataset.BASE: THE_BASE_PROTOCOL,
    Dataset.HCP: HCP_PROTOCOL,
}

LOGIT_PARAMETERS = {
    "penalty": "elasticnet",
    "solver": "saga",
    "max_iter": 1e4,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
LOGIT_GRID = {
    "C": loguniform(1e-2, 1e2),
    "l1_ratio": uniform(),
}
KNN_PARAMETERS = {"n_jobs": -1}
KNN_GRID = {
    "n_neighbors": range(1, 11),
    "weights": ("uniform", "distance"),
}
MODEL_CONFIGURATION = {
    LogisticRegression: (LOGIT_PARAMETERS, LOGIT_GRID),
    KNeighborsClassifier: (KNN_PARAMETERS, KNN_GRID),
}
SEARCHER = {
    LogisticRegression: RandomizedSearchCV,
    KNeighborsClassifier: GridSearchCV,
}
MODELS = LogisticRegression, KNeighborsClassifier


def filter_by_dataset(
    context: pd.DataFrame, results: pd.DataFrame, dataset: Dataset
):
    protocol = DATASET_SEQUENCE[dataset]
    dataset_context = context.loc[
        (context[list(protocol)] == pd.Series(protocol)).all(axis=1)
    ]
    dataset_results = results.loc[dataset_context.index]
    return dataset_context, dataset_results


def preprocess_results(
    dataset: Dataset = Dataset.BASE,
    atlas: str = DEFAULT_ATLAS,
    test_size: float = 0.1,
    random_state=RANDOM_STATE,
):
    context, results = read_results(atlas=atlas)
    dataset_context, dataset_results = filter_by_dataset(
        context, results, dataset
    )
    X = dataset_results.to_numpy()
    y = dataset_context["Sex"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test, y_train, y_test


def train_models(
    dataset: Dataset = Dataset.BASE,
    atlas: str = DEFAULT_ATLAS,
    random_state=RANDOM_STATE,
):
    X_train, X_test, y_train, y_test = preprocess_results(
        dataset, atlas, test_size=0.1, random_state=random_state
    )
    results = []
    for model in MODELS:
        parameters, grid = MODEL_CONFIGURATION[model]
        searcher = SEARCHER[model](model(**parameters), grid)
        searcher.fit(X_train, y_train)
        results.append((searcher, searcher.score(X_test, y_test)))
    return results
