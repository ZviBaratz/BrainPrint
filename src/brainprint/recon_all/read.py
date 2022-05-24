"""
Definition of the :func:`read_results` utility function.
"""
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_RESULTS_DIR: Path = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_RESULTS_NAME: str = "recon_all.csv"
DEFAULT_RESULTS_PATH: Path = DEFAULT_RESULTS_DIR / DEFAULT_RESULTS_NAME

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
    path: Path = None, atlas: str = DEFAULT_ATLAS
) -> pd.DataFrame:
    """
    Read a DataFrame of FreeSurfer's recon-all workflow results as exported
    by pylabber and returns a tranformed DataFrame, more suitable for EDA and
    statistical evaluation.

    Parameters
    ----------
    path : Path, optional
        Path to results CSV as exported by pylabber, by default None
    atlas : str, optional
        Whether to index by a particular atlas, by default
        :attr:`DEFAULT_ATLAS`

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Research context, transformed DataFrame
    """
    # Read results in raw export format.
    path = path or DEFAULT_RESULTS_PATH
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
    if atlas:
        results = results.xs(atlas, level="Atlas", axis=1)
        results = results.dropna(axis=0, how="any")
    context_path = path.with_name("context.csv")
    context = pd.read_csv(context_path, index_col=0)
    context = context.loc[results.index]
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
    context["Session Time"] = pd.to_datetime(context["Session Time"])
    return context, results
