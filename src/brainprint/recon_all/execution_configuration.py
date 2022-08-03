from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd


class ExecutionConfiguration(Enum):
    DEFAULT = "Default"
    T2 = "T2"
    FLAIR = "FLAIR"
    MPRAGE = "MPRAGE"
    MPRAGE_AND_T2 = "T2 + MPRAGE"
    MPRAGE_AND_FLAIR = "FLAIR + MPRAGE"
    MPRAGE_AND_3T_AND_T2 = "T2 + MPRAGE + 3T"
    MPRAGE_AND_3T_AND_FLAIR = "FLAIR + MPRAGE + 3T"


EXECUTIONS_CONFIGURATIONS: Dict[ExecutionConfiguration, Dict[str, Any]] = {
    ExecutionConfiguration.DEFAULT: {
        "use_T2": False,
        "use_FLAIR": False,
        "mprage": False,
        "flags": None,
    },
    ExecutionConfiguration.FLAIR: {
        "use_T2": False,
        "use_FLAIR": True,
        "mprage": False,
        "flags": None,
    },
    ExecutionConfiguration.T2: {
        "use_T2": True,
        "use_FLAIR": False,
        "mprage": False,
        "flags": None,
    },
    ExecutionConfiguration.MPRAGE: {
        "use_T2": False,
        "use_FLAIR": False,
        "mprage": True,
        "flags": None,
    },
    ExecutionConfiguration.MPRAGE_AND_FLAIR: {
        "use_T2": False,
        "use_FLAIR": True,
        "mprage": True,
        "flags": None,
    },
    ExecutionConfiguration.MPRAGE_AND_T2: {
        "use_T2": True,
        "use_FLAIR": False,
        "mprage": True,
        "flags": None,
    },
    ExecutionConfiguration.MPRAGE_AND_3T_AND_T2: {
        "use_T2": True,
        "use_FLAIR": False,
        "mprage": True,
        "flags": "['-3T']",
    },
    ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR: {
        "use_T2": False,
        "use_FLAIR": True,
        "mprage": True,
        "flags": "['-3T']",
    },
}

CONFIGURATIONS_LABELS: Dict[ExecutionConfiguration, str] = {
    ExecutionConfiguration.DEFAULT: "Default",
    ExecutionConfiguration.FLAIR: "FLAIR",
    ExecutionConfiguration.T2: "T2",
    ExecutionConfiguration.MPRAGE: "MPRAGE",
    ExecutionConfiguration.MPRAGE_AND_FLAIR: "FLAIR + MPRAGE",
    ExecutionConfiguration.MPRAGE_AND_T2: "T2 + MPRAGE",
    ExecutionConfiguration.MPRAGE_AND_3T_AND_T2: "T2 + MPRAGE + 3T",
    ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR: "FLAIR + MPRAGE + 3T",
}


def infer_row_execution_configuration(
    row: pd.Series,
) -> ExecutionConfiguration:
    """
    Infer the execution configuration from a row of the raw configurations CSV.

    Parameters
    ----------
    row : pd.Series
        A row of the raw configurations CSV

    Returns
    -------
    ExecutionConfiguration
        The inferred execution configuration
    """
    for configuration, parameters_dict in EXECUTIONS_CONFIGURATIONS.items():
        if all(row[key] == parameters_dict[key] for key in parameters_dict):
            return configuration.value


def infer_execution_configurations(
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    return configurations.replace({np.nan: None}).apply(
        infer_row_execution_configuration, axis=1
    )
