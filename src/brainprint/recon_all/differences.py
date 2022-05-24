from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import swifter  # noqa: F401
from brainprint.recon_all.read import read_results
from brainprint.recon_all.utils import DATASET_SEQUENCE, Dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
RECON_ALL_DIR = RESULTS_DIR / "recon_all"
DIFFERENCES_DIR = RECON_ALL_DIR / "differences"
ATLASES = "Desikan-Killiany", "Destrieux"


def calculate_difference(
    results: pd.DataFrame, run_1: int, run_2: int
) -> np.array:
    return results.loc[run_1].to_numpy() - results.loc[run_2].to_numpy()


def check_same_subject(context: pd.DataFrame, run_1: int, run_2: int) -> bool:
    try:
        return (
            context.loc[run_1, "Subject ID"]
            == context.loc[run_2, "Subject ID"]
        )
    except AssertionError:
        print(f"Runs {run_1} and {run_2} raised AssertionError!")


def calculate_differences(
    context: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    differences = {}
    run_combinations = pd.Series(combinations(results.index, 2))
    values = run_combinations.swifter.apply(
        lambda runs: calculate_difference(results, runs[0], runs[1])
    ).reset_index(drop=True)
    same_subject = run_combinations.swifter.apply(
        lambda runs: check_same_subject(context, runs[0], runs[1])
    )
    index = pd.MultiIndex.from_tuples(run_combinations)
    same_subject.index = index
    differences = pd.DataFrame.from_dict(dict(zip(index, values))).T
    differences.columns = results.columns
    differences.index.names = "Run 1", "Run 2"
    return differences, same_subject
