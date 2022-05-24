from enum import Enum
from typing import Any, Dict, Tuple

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
