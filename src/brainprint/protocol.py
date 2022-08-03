from enum import Enum
from typing import Any, Dict

import pandas as pd

THE_BASE_PROTOCOL = {
    "Corrected": True,
    "Inversion Time": 1000,
    "Echo Time": 2.78,
    "Repetition Time": 2400,
    "Spatial Resolution": "[0.9, 0.9, 0.9]",
}

THE_BASE_UNCORRECTED_PROTOCOL = {
    "Corrected": False,
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


class Protocol(Enum):
    HCP = "Human Connectome Project"
    BASE = "The Base (corrected)"
    BASE_UNCORRECTED = "The Base (uncorrected)"


PROTOCOL_PARAMETERS: Dict[Protocol, Dict[str, Any]] = {
    Protocol.BASE: THE_BASE_PROTOCOL,
    Protocol.BASE_UNCORRECTED: THE_BASE_UNCORRECTED_PROTOCOL,
    Protocol.HCP: HCP_PROTOCOL,
}


def infer_protocol_from_row(row: pd.Series) -> Protocol:
    """
    Infer the protocol from a row of the raw results CSV.

    Parameters
    ----------
    row : pd.Series
        A row of the raw results CSV.

    Returns
    -------
    Protocol
        The inferred protocol.
    """
    for protocol, parameter_dict in PROTOCOL_PARAMETERS.items():
        match = all(
            row[parameter] == parameter_dict[parameter]
            for parameter in parameter_dict
        )
        if match:
            return protocol.value


def infer_protocol(context: pd.DataFrame) -> pd.Series:
    return context.apply(lambda row: infer_protocol_from_row(row), axis=1)
