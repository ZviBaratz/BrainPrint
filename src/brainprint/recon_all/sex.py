import numpy as np
import pandas as pd
from brainprint.recon_all.read import DEFAULT_ATLAS, read_results
from brainprint.recon_all.utils import DATASET_SEQUENCE, Dataset
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

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
LOGIT_SEARCH = {
    "n_iter": 10,
    "n_jobs": -1,
}
KNN_PARAMETERS = {"n_jobs": -1}
KNN_GRID = {
    "n_neighbors": range(1, 11),
    "weights": ("uniform", "distance"),
}
KNN_SEARCH = {
    "n_jobs": -1,
}
RF_PARAMETERS = {"n_jobs": -1}
RF_GRID = {
    "n_estimators": randint(80, 150),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 6),
}
RF_SEARCH = {
    "n_iter": 10,
    "n_jobs": -1,
}
GNB_GRID = {"var_smoothing": loguniform(1e-11, 1e-5)}
GNB_PARAMETERS = {}
GNB_SEARCH = {
    "n_iter": 10,
    "n_jobs": -1,
}
SVC_PARAMETERS = {"kernel": "linear"}
SVC_GRID = {"C": loguniform(1e-2, 1e2)}
SVC_SEARCH = {
    "n_iter": 10,
    "n_jobs": -1,
}
MLP_PARAMETERS = {"solver": "lbfgs", "max_iter": 100}
MLP_GRID = {
    "hidden_layer_sizes": randint(100, 1000),
    "activation": ("logistic", "tanh", "relu"),
    "alpha": loguniform(1e-5, 1e1),
}
MLP_SEARCH = {
    "n_iter": 10,
    "n_jobs": -1,
}
MODEL_CONFIGURATION = {
    LogisticRegression: (LOGIT_PARAMETERS, LOGIT_GRID, LOGIT_SEARCH),
    KNeighborsClassifier: (KNN_PARAMETERS, KNN_GRID, KNN_SEARCH),
    RandomForestClassifier: (RF_PARAMETERS, RF_GRID, RF_SEARCH),
    GaussianNB: (GNB_PARAMETERS, GNB_GRID, GNB_SEARCH),
    SVC: (SVC_PARAMETERS, SVC_GRID, SVC_SEARCH),
    MLPClassifier: (MLP_PARAMETERS, MLP_GRID, MLP_SEARCH),
}
FULL_GRID = (KNeighborsClassifier,)
MODELS = (
    LogisticRegression,
    KNeighborsClassifier,
    RandomForestClassifier,
    GaussianNB,
    SVC,
    MLPClassifier,
)


def filter_by_dataset(
    context: pd.DataFrame, results: pd.DataFrame, dataset: Dataset
):
    if dataset is None:
        return context, results
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
    target_name: str = "Sex",
):
    context, results = read_results(atlas=atlas)
    dataset_context, dataset_results = filter_by_dataset(
        context, results, dataset
    )
    X = dataset_results.to_numpy()
    y = dataset_context[target_name].copy()
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
    test_size: float = 0.1,
):
    X_train, X_test, y_train, y_test = preprocess_results(
        dataset, atlas, test_size=test_size, random_state=random_state
    )
    results = []
    for model in MODELS:
        parameters, grid, searcher_params = MODEL_CONFIGURATION[model]
        SearcherModel = (
            GridSearchCV if model in FULL_GRID else RandomizedSearchCV
        )
        searcher = SearcherModel(model(**parameters), grid, **searcher_params)
        searcher.fit(X_train, y_train)
        results.append(searcher)
    return results, X_test, y_test
