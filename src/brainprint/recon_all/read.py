"""
Definition of the :func:`read_results` utility function.
"""
import logging
from datetime import tzinfo
from pathlib import Path
from typing import List

import pandas as pd
from brainprint.recon_all import logs

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR: Path = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_RESULTS_NAME: str = "results.csv"
DEFAULT_RESULTS_PATH: Path = DEFAULT_RESULTS_DIR / DEFAULT_RESULTS_NAME
DEFAULT_CONTEXT_NAME: str = "context.csv"
DEFAULT_CONTEXT_PATH: Path = DEFAULT_RESULTS_DIR / DEFAULT_CONTEXT_NAME
DEFAULT_CONFIGURATIONS_NAME: str = "configurations.csv"
DEFAULT_CONFIGURATIONS_PATH: Path = (
    DEFAULT_RESULTS_DIR / DEFAULT_CONFIGURATIONS_NAME
)
DEFAULT_QUESTIONNAIRE_NAME: str = "questionnaire.csv"
DEFAULT_QUESTIONNAIRE_PATH: Path = (
    DEFAULT_RESULTS_DIR / DEFAULT_QUESTIONNAIRE_NAME
)

#
# Default pivoted DataFrame configuration.
#
INDEX: str = "Run ID"
COLUMNS: List[str] = ["Atlas", "Hemisphere", "Region Name"]
VALUES: List[str] = [
    "Surface Area",
    "Gray Matter Volume",
    "Average Thickness",
    "Thickness StdDev",
    "Integrated Rectified Mean Curvature",
    "Integrated Rectified Gaussian Curvature",
    "Folding Index",
    "Intrinsic Curvature Index",
]
COLUMN_LEVELS: List[str] = ["Atlas", "Hemisphere", "Region Name", "Metric"]
DEFAULT_ATLAS: str = "Destrieux"


def read_results(
    path: Path = DEFAULT_RESULTS_PATH, atlas: str = DEFAULT_ATLAS
) -> pd.DataFrame:
    """
    Read a DataFrame of FreeSurfer's recon-all workflow results as exported
    by pylabber and returns a tranformed DataFrame, more suitable for EDA and
    statistical evaluation.

    Parameters
    ----------
    path : Path, optional
        Path to results CSV as exported by pylabber, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Research context, transformed DataFrame
    """
    # Log start.
    start_message = logs.READ_RESULTS_START.format(path=path)
    logger.debug(start_message)

    # Read results in raw export format.
    results = pd.read_csv(path).pivot(
        index=INDEX, columns=COLUMNS, values=VALUES
    )
    # Rename pivotted columns.
    results.columns.names = [
        value if value is not None else "Metric"
        for value in results.columns.names
    ]
    # Reorder and sort levels.
    results = results.reorder_levels(COLUMN_LEVELS, axis=1)
    results.sort_index(axis=1, inplace=True)
    # Drop null values.
    results = results.dropna(axis=1, how="all")

    # Log end.
    end_message = logs.READ_RESULTS_END.format(n=len(results), path=path)
    logger.info(end_message)
    return results


def read_context(path: Path = DEFAULT_CONTEXT_PATH) -> pd.DataFrame:
    # Log start.
    start_message = logs.READ_CONTEXT_START.format(path=path)
    logger.debug(start_message)

    context = pd.read_csv(path, index_col=0)
    context["Corrected"] = context["Scan File Name"].str.contains(
        "ce-corrected"
    )
    context["Spatial Resolution"] = (
        context["Spatial Resolution"]
        .apply(
            lambda values: [
                round(float(value), 2) for value in values[1:-1].split(",")
            ]
        )
        .astype(str)
    )
    context["Session Time"] = pd.to_datetime(
        context["Session Time"], utc=False
    )
    context["Date of Birth"] = pd.to_datetime(
        context["Date of Birth"], utc=False
    )
    context["Age (days)"] = (
        context["Session Time"].dt.date - context["Date of Birth"].dt.date
    ).dt.days
    context["Sex"] = (
        context["Sex"].replace({"M": "Male", "F": "Female"}).astype("category")
    )

    # Log end.
    end_message = logs.READ_CONTEXT_END.format(n=len(context), path=path)
    logger.info(end_message)
    return context


BOOLEAN_CONFIGURATIONS = "use_T2", "use_FLAIR", "mprage"


def read_configurations(
    path: Path = DEFAULT_CONFIGURATIONS_PATH,
) -> pd.DataFrame:
    # Log start.
    start_message = logs.READ_CONFIGURATIONS_START.format(path=path)
    logger.debug(start_message)

    df = pd.read_csv(path, index_col=0).drop(
        columns=[
            "subjects_dir",
            "directive",
            "T2_file",
            "FLAIR_file",
            "T1_files",
        ]
    )
    for label in BOOLEAN_CONFIGURATIONS:
        if label in df:
            df[label] = df[label].fillna(False)

    # Log end.
    end_message = logs.READ_CONFIGURATIONS_END.format(n=len(df), path=path)
    logger.info(end_message)
    return df
