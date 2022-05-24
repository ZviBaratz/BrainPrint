"""
TODO:
* Add complete run configuration to server CSV export.
* Create class to easily extract samples with various queries:
    context:
      - between and within subject scans
      - analysis configuration
      - scan parameters
    results:
      - atlas
      - anatomical statistics subset
      - brainprint
* Reproduce existing results with FS stats subsets.
* Reproduce existing results with brainprint features.
"""
from pathlib import Path

import pandas as pd
from brainprint.recon_all.read import read_results
from brainprint.recon_all.utils import DATASET_SEQUENCE, Dataset


def filter_by_atlas(df: pd.DataFrame, atlas_name: str) -> pd.DataFrame:
    atlas_results = df.xs(atlas_name, level="Atlas", axis=1)
    return atlas_results.dropna(axis=0, how="any")


def filter_by_dataset(
    context: pd.DataFrame, results: pd.DataFrame, dataset_id: str
):
    dataset = Dataset[dataset_id]
    sequence = DATASET_SEQUENCE[dataset]
    dataset_context = context.loc[
        (context[list(sequence)] == pd.Series(sequence)).all(axis=1)
    ]
    dataset_results = results.loc[dataset_context.index]
    return dataset_context, dataset_results


class ReconAllResults:
    def __init__(self, path: Path = None) -> None:
        self.path = path
        self.context, self.raw_results = read_results(self.path)
