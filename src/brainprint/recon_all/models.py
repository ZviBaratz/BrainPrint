"""
Definition of the :class:`ReconAllModels` class.
"""
from typing import Any, Dict, List

from brainprint.recon_all.model_specs import SEX_GRIDS, SEX_MODELS
from brainprint.recon_all.results import ReconAllResults
from sklearn.base import BaseEstimator


class ReconAllModels:
    """
    Handles the creation and evaluation of models for predicting various
    subject attributes from recon-all results.
    """

    MODELS: Dict[str, List[BaseEstimator]] = {"Sex": SEX_MODELS}
    GRID: Dict[str, Dict[BaseEstimator, Dict[str, Any]]] = {"Sex": SEX_GRIDS}

    def __init__(self, results: ReconAllResults):
        self.results = results
