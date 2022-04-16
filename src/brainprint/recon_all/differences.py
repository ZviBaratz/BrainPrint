from itertools import combinations
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from brainprint.recon_all.read import read_results
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent.parent / "results"
RECON_ALL_DIR = RESULTS_DIR / "recon_all"
DIFFERENCES_DIR = RECON_ALL_DIR / "differences"


def calculate_euclidean_distance(row: pd.Series, results: pd.DataFrame = None):
    if results is None:
        _, results = read_results()
    values_1 = results.loc[row["Run 1"]]
    values_2 = results.loc[row["Run 2"]]
    return np.linalg.norm(values_1 - values_2)


def check_same_subject(row: pd.Series, context: pd.DataFrame = None) -> bool:
    if context is None:
        context, _ = read_results()
    subject_1 = context.loc[row["Run 1"], "Subject ID"]
    subject_2 = context.loc[row["Run 2"], "Subject ID"]
    return subject_1 == subject_2


def calculate_differences():
    context, results = read_results()
    run_combinations = combinations(results.index, 2)
    df = pd.DataFrame(
        list(run_combinations),
        columns=["Run 1", "Run 2"],
    )
    print("Calcuclating distances...")
    df["Euclidean Distance"] = df.swifter.apply(
        calculate_euclidean_distance, results=results, axis=1
    )
    print("Comparing subject identities...")
    df["Same Subject"] = df.swifter.apply(
        check_same_subject, context=context, axis=1
    )
    return df


def calculate_run_differences(run: int, results: pd.DataFrame) -> pd.DataFrame:
    run_stats = results.loc[run]
    index = results.index[results.index != run]
    return results.loc[index] - run_stats


def write_differences(results: pd.DataFrame, destination: Path = None):
    destination = destination or DIFFERENCES_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for run in tqdm(results.index, unit="run"):
        differences = calculate_run_differences(run, results)
        run_destination = destination / f"{run}.csv"
        differences.to_csv(run_destination)


def read_differences(source: Path) -> dd.DataFrame:
    pattern = str(source / "*.csv")
    _, results = read_results()
    names = ["Run 2"] + [
        f"{p1[0]}_{p2}_{p3.replace(' ', '_')}"
        for p1, p2, p3 in results.columns
    ]
    ddf = dd.read_csv(
        pattern,
        skiprows=[0, 1, 2, 3],
        include_path_column=True,
        dtype={"Run ID": int},
        names=names,
    )
    ddf = ddf.repartition(partition_size="100MB")
    ddf["Run 1"] = ddf["path"].apply(
        lambda p: int(Path(p).name.split(".")[0]), meta=("Run 1", int)
    )
    ddf = ddf.drop("path", axis=1)
    ordered = ["Run 1", "Run 2"] + list(ddf.columns)[1:-1]
    ddf = ddf.loc[:, ordered]
    ddf.persist()
    return ddf
