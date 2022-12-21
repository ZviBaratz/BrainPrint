"""
Definition of the :class:`ReconAllDifferences` class.
"""
import logging
from itertools import combinations
from pathlib import Path
from typing import Generator, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import swifter  # noqa: F401
from brainprint.protocol import Protocol
from brainprint.recon_all import logs
from brainprint.recon_all.execution_configuration import ExecutionConfiguration
from brainprint.recon_all.metric import AXIS_KWARGS, Metric
from brainprint.recon_all.results import ReconAllResults, load_results
from brainprint.recon_all.utils import get_default_cache_dir
from scipy.spatial.distance import cityblock
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ReconAllDifferences:
    DIFFERENCES_FILE_PATTERN: str = (
        "{protocol}_{configuration}_{within_or_between}.parquet"
    )
    DISTANCES_FILE_PATTERN: str = (
        "{protocol}_{configuration}_{within_or_between}_distances.parquet"
    )

    def __init__(
        self,
        results: ReconAllResults = None,
    ) -> None:
        self.results = (
            load_results(
                completed_only=True, multi_only=True, questionnaire_only=False
            )
            if results is None
            else results
        )

    def get_multi_run_indices(
        self,
        protocol: Union[Protocol, Iterable[Protocol]] = None,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ] = None,
    ) -> pd.Series:
        filtered_results = self.results.filter_results(
            protocol=protocol,
            configuration=configuration,
        )
        filtered_context = self.results.context.loc[filtered_results.index]
        multi_run_mask = filtered_context.duplicated(
            keep=False, subset="Subject ID"
        )

        n_subjects = filtered_context["Subject ID"].unique().size
        n_runs = multi_run_mask.sum()
        message = logs.MULTI_RUN_INDICES.format(
            n_subjects=n_subjects, n_runs=n_runs
        )
        logger.info(message)

        return filtered_context[multi_run_mask].index

    def generate_index_pairs(
        self, indices: Iterable[int], mode: str = None
    ) -> Generator[Tuple[int, int], None, None]:
        for index_1, index_2 in combinations(indices, 2):
            if mode is None:
                yield index_1, index_2
            same_subject = self.results.check_same_subject(index_1, index_2)
            match = (same_subject and mode == "within") or (
                not same_subject and mode == "between"
            )
            if match:
                yield index_1, index_2

    def get_index_pairs(
        self,
        protocol: Union[Protocol, Iterable[Protocol]] = None,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ] = None,
    ) -> Tuple[
        Union[List[Tuple[int, int]], None], Union[List[Tuple[int, int]], None]
    ]:
        # Log start.
        start_message = logs.INDEX_PAIRS_START.format(
            n=len(self.results.context)
        )
        logger.debug(start_message)

        indices = self.get_multi_run_indices(
            protocol=protocol,
            configuration=configuration,
        )
        within = list(self.generate_index_pairs(indices, mode="within"))
        between = list(self.generate_index_pairs(indices, mode="between"))

        # Log end.
        end_message = logs.INDEX_PAIRS_END.format(
            n_within=len(within), n_between=len(between), n_runs=len(indices)
        )
        logger.info(end_message)
        return within, between

    def calculate_difference(
        self, values: pd.DataFrame, run_1: int, run_2: int
    ) -> np.array:
        results_1 = values.loc[run_1].to_numpy()
        results_2 = values.loc[run_2].to_numpy()
        return results_1 - results_2

    def get_differences_name(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        within_or_between: str,
    ) -> Path:
        return self.DIFFERENCES_FILE_PATTERN.format(
            protocol=protocol.name,
            configuration=configuration.name,
            within_or_between=within_or_between,
        )

    def read_existing_difference(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        within_or_between: str,
        destination: Path = None,
    ) -> pd.DataFrame:
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        name = self.get_differences_name(
            protocol=protocol,
            configuration=configuration,
            within_or_between=within_or_between,
        )
        path = destination / name
        if path.exists():
            return pd.read_parquet(path)

    def read_existing_differences(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        destination: Path = None,
    ) -> pd.DataFrame:
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        existing = []
        for within_or_between in ["within", "between"]:
            subsample = self.read_existing_difference(
                protocol=protocol,
                configuration=configuration,
                within_or_between=within_or_between,
                destination=destination,
            )
            if subsample is not None:
                existing.append(subsample)
        if len(existing) == 2:
            existing_found_message = logs.EXISTING_DIFFERENCES_FOUND.format(
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.info(existing_found_message)
            return tuple(existing)

    def calculate_filtered_differences(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        destination: Path = None,
        force: bool = False,
        standardize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates differences between runs with the provided *protocol* and
        *configuration*.

        Parameters
        ----------
        protocol : Protocol
            Protocol to filter runs by
        configuration : ExecutionConfiguration
            Configuration to filter runs by
        destination : Path, optional
            Destination to save the result to, by default None
        force : bool, optional
            Whether to recalculate differences even if an existing result is
            found at the default destination, by default False
        standardize : bool, optional
            Whether to standardize all selected runs before calculating
            differences, by default True

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Within-subject differences, between-subject differences
        """
        # Mediate destination and validate existence.
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        destination.mkdir(parents=True, exist_ok=True)

        # Log start.
        iteration_start_log = logs.DIFFERENCES_ITERATION.format(
            protocol=protocol, execution_configuration=configuration
        )
        logger.debug(iteration_start_log)

        within = between = None

        if not force:
            # Check if the differences have already been calculated.
            existing = self.read_existing_differences(
                protocol=protocol,
                configuration=configuration,
                destination=destination,
            )
            if existing is not None:
                return existing

        # Query filtered results.
        values = self.results.filter_results(
            protocol=protocol, configuration=configuration
        ).copy()

        # Standardize if requested.
        if standardize:
            logger.info(
                f"Standardizing {protocol} + {configuration} values..."
            )
            values.loc[:, :] = StandardScaler().fit_transform(values)
            logger.info(
                f"{len(values.index)} {protocol} + {configuration} values standardized."
            )

        #
        # Calculate differences
        #

        # Generate index pairs.
        (
            within_index_pairs,
            between_index_pairs,
        ) = self.get_index_pairs(protocol, configuration)

        # Convert to dataframe to parallelize more easily using
        # swifter.
        within_df = pd.DataFrame(
            within_index_pairs, columns=["Run 1", "Run 2"]
        )
        between_df = pd.DataFrame(
            between_index_pairs, columns=["Run 1", "Run 2"]
        )

        # Calculate within-subject differences.
        if within_df.empty:
            no_within_message = logs.NO_WITHIN_SUBJECT_RUNS.format(
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.warn(no_within_message)
        else:
            # Log within subject differences calculation start.
            within_start_message = logs.DIFFERENCES_WITHIN_START.format(
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.debug(within_start_message)

            # Create pandas index from index pairs.
            index = pd.Index(
                within_index_pairs,
                name=("Run 1", "Run 2"),
            )

            # Calculate differences.
            within = (
                within_df.swifter.apply(
                    lambda pair: self.calculate_difference(
                        values, pair["Run 1"], pair["Run 2"]
                    ),
                    axis=1,
                )
                .apply(pd.Series)
                .set_index(index)
                .sort_index()
                .set_axis(self.results.clean.columns, axis=1)
            )

            # Log within subject differences calculation end.
            within_end_message = logs.DIFFERENCES_WITHIN_END.format(
                n_pairs=len(within_index_pairs),
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.info(within_end_message)

            # Save within subject differences to file.
            name = self.DIFFERENCES_FILE_PATTERN.format(
                protocol=protocol.name,
                configuration=configuration.name,
                within_or_between="within",
            )
            path = destination / name
            within.to_parquet(path)

        # Calculate between subject differences.
        if between_df.empty:
            no_between_message = logs.NO_BETWEEN_SUBJECT_RUNS.format(
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.warn(no_between_message)
        else:
            # Log between subject differences calculation start.
            between_start_message = logs.DIFFERENCES_BETWEEN_START.format(
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.debug(between_start_message)

            # Create pandas index from index pairs.
            index = pd.Index(
                between_index_pairs,
                name=("Run 1", "Run 2"),
            )

            # Calculate differences.
            between = (
                between_df.swifter.apply(
                    lambda pair: self.calculate_difference(
                        values, pair["Run 1"], pair["Run 2"]
                    ),
                    axis=1,
                )
                .apply(pd.Series)
                .set_index(index)
                .sort_index()
                .set_axis(self.results.clean.columns, axis=1)
            )

            # Log between subject differences calculation end.
            between_end_message = logs.DIFFERENCES_BETWEEN_END.format(
                n_pairs=len(between_index_pairs),
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.info(between_end_message)

            # Save between subject differences to file.
            name = self.DIFFERENCES_FILE_PATTERN.format(
                protocol=protocol.name,
                configuration=configuration.name,
                within_or_between="between",
            )
            path = destination / name
            between.to_parquet(path)

        return within, between

    def calculate_differences(
        self, destination: Path = None, force: bool = False
    ) -> pd.DataFrame:
        # Mediate the destination and validate existence.
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        destination.mkdir(parents=True, exist_ok=True)

        # Log start.
        start_message = logs.DIFFERENCES_START.format(
            n_protocols=len(self.results.protocol),
            n_configurations=len(self.results.configuration),
            n_runs=len(self.results.context),
        )
        logger.info(start_message)

        # Calculate differences per acquisition protocol and recon-all
        # execution configuration.
        for protocol in self.results.protocol:
            for configuration in self.results.configuration:
                self.calculate_filtered_differences(
                    protocol=protocol,
                    configuration=configuration,
                    destination=destination,
                    force=force,
                )

    def calculate_cosine_distance(
        self, values, run_id_1: int, run_id_2: int
    ) -> float:
        run_1 = values.loc[run_id_1]
        run_2 = values.loc[run_id_2]
        return 1 - (
            np.dot(run_1, run_2)
            / (np.linalg.norm(run_1) * np.linalg.norm(run_2))
        )

    def get_distances_name(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        within_or_between: str,
    ) -> Path:
        return self.DISTANCES_FILE_PATTERN.format(
            protocol=protocol.name,
            configuration=configuration.name,
            within_or_between=within_or_between,
        )

    def get_distance_path(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        within_or_between: str,
        destination: Path = None,
    ) -> Path:
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        return destination / self.get_distances_name(
            protocol, configuration, within_or_between
        )

    def read_existing_distance(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        within_or_between: str,
        destination: Path = None,
    ) -> pd.DataFrame:
        path = self.get_distance_path(
            protocol, configuration, within_or_between, destination
        )
        if path.exists():
            existing_found = logs.EXISTING_DISTANCES_FOUND.format(
                within_or_between=within_or_between,
                protocol=protocol,
                execution_configuration=configuration,
            )
            logger.info(existing_found)
            return pd.read_parquet(path)

    def calculate_filtered_distances(
        self,
        protocol: Protocol,
        configuration: ExecutionConfiguration,
        destination: Path = None,
        standardize: bool = True,
        subset: Iterable[str] = None,
        force: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Mediate destination and validate existence.
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        destination.mkdir(parents=True, exist_ok=True)

        # Read or calculate differences between result vectors.
        (
            within_differences,
            between_differences,
        ) = self.calculate_filtered_differences(
            protocol=protocol,
            configuration=configuration,
            destination=destination,
            force=force,
        )

        # Read or calculate distances between result vectors.
        both_distances = []
        values = self.results.filter_results(
            protocol=protocol, configuration=configuration
        )
        if subset is not None:
            values = self.results.select_metrics(subset, results=values)
        if standardize:
            logger.info(
                f"Standardizing {protocol} + {configuration} values..."
            )
            values.loc[:, :] = StandardScaler().fit_transform(values)
            logger.info(
                f"{len(values.index)} {protocol} + {configuration} values standardized."
            )
        for within_or_between in ["within", "between"]:
            existing = self.read_existing_distance(
                protocol=protocol,
                configuration=configuration,
                within_or_between=within_or_between,
                destination=destination,
            )
            if existing is not None and not force:
                both_distances.append(existing)
                continue
            differences = (
                within_differences
                if within_or_between == "within"
                else between_differences
            )
            if differences is None:
                continue
            distances = pd.DataFrame(
                columns=[
                    Metric.EUCLIDEAN.value,
                    Metric.COSINE.value,
                    Metric.MANHATTAN.value,
                ]
            )
            distances[Metric.COSINE.value] = differences.index.to_frame(
                index=False
            ).swifter.apply(
                lambda row: self.calculate_cosine_distance(
                    values, row["Run 1"], row["Run 2"]
                ),
                axis=1,
            )
            distances[Metric.EUCLIDEAN.value] = np.sqrt(
                (differences.values**2).sum(axis=1)
            )
            distances[Metric.MANHATTAN.value] = np.abs(differences.values).sum(
                axis=1
            )
            distances.index = differences.index
            path = self.get_distance_path(
                protocol, configuration, within_or_between, destination
            )
            distances.to_parquet(path)
            both_distances.append(distances)
        return tuple(both_distances)

    def calculate_distances(
        self,
        destination: Path = None,
        force: bool = False,
        subset: Iterable[str] = None,
    ) -> None:
        destination = (
            get_default_cache_dir() if destination is None else destination
        )
        start_message = logs.DISTANCES_START.format(
            n_runs=len(self.results.context),
            n_protocols=len(self.results.protocol),
            n_configurations=len(self.results.configuration),
        )
        logger.info(start_message)
        results = {}
        for protocol in self.results.protocol:
            for configuration in self.results.configuration:
                results[
                    (protocol, configuration)
                ] = self.calculate_filtered_distances(
                    protocol=protocol,
                    configuration=configuration,
                    destination=destination,
                    subset=subset,
                    force=force,
                )
        return results

    def plot_distances(
        self,
        metrics: Metric = None,
        within_or_between: str = "both",
        protocols: List[Protocol] = None,
        configurations: List[ExecutionConfiguration] = None,
        source: Path = None,
        binwidth: float = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        protocols = protocols or self.results.protocol
        metrics = metrics or [
            Metric.EUCLIDEAN,
            Metric.MANHATTAN,
            Metric.COSINE,
        ]
        configurations = configurations or self.results.configuration
        source = get_default_cache_dir() if source is None else source
        fig, ax = plt.subplots(
            figsize=(20, 15),
            nrows=len(configurations),
            ncols=len(metrics),
            sharex="col",
            sharey="col",
        )
        fig.suptitle(
            "Distance Distribution Density by Subject Identity\n",
            y=0.85,
            fontsize=24,
        )
        for protocol in protocols:
            for i_row, configuration in enumerate(configurations):
                for i_col, metric in enumerate(metrics):
                    current_axis = (
                        ax[i_row, i_col] if len(metrics) > 1 else ax[i_row]
                    )
                    within, between = self.calculate_filtered_distances(
                        protocol=protocol,
                        configuration=configuration,
                        destination=source,
                    )
                    if within_or_between in ["within", "both"]:
                        sns.histplot(
                            x=within[metric.value],
                            label="Within subject",
                            ax=current_axis,
                            kde=True,
                            stat="density",
                            color="orange",
                            binwidth=binwidth,
                        )
                        current_axis.axvline(
                            within[metric.value].max(),
                            color="orange",
                            linestyle="--",
                            label="Max. within subject",
                        )
                        min_between = between[metric.value].min()
                        misses = len(
                            within[within[metric.value] > min_between]
                        )
                        sensitivity = 1 - (misses / len(within))
                        current_axis.text(
                            0.6,
                            0.6,
                            f"Sensitivity: {sensitivity:.3f}",
                            transform=current_axis.transAxes,
                            fontsize=14,
                        )
                    if within_or_between in ["between", "both"]:
                        min_value = between[metric.value].min()
                        sns.histplot(
                            between[metric.value],
                            label="Between subject",
                            ax=current_axis,
                            kde=True,
                            stat="density",
                            color="gray",
                            binwidth=binwidth,
                        )
                        current_axis.axvline(
                            min_value,
                            color="gray",
                            linestyle="--",
                            label="Min. between subject",
                        )
                    current_axis.set(
                        xlabel=None,
                        ylabel=None,
                        **AXIS_KWARGS.get(metric, {}),
                    )
                    if i_row == len(configurations) - 1:
                        edges = (
                            min(
                                [
                                    between[metric.value].min(),
                                    within[metric.value].min(),
                                ]
                            ),
                            max(
                                [
                                    between[metric.value].max(),
                                    within[metric.value].max(),
                                ]
                            ),
                        )
                        edge_range = edges[1] - edges[0]
                        gap = edge_range * 0.05
                        current_axis.set_xlim(
                            np.array(edges) + np.array([-gap, gap])
                        )
                        current_axis.set_xlabel(metric.value, fontsize=20)
                    if i_col == 0:
                        current_axis.set_ylabel(
                            configuration.value,
                            rotation=75,
                            labelpad=12,
                            fontsize=14,
                        )
        fig.text(
            0,
            0.5,
            "Execution Configuration",
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        if len(metrics) > 1:
            ax[0, -1].legend(bbox_to_anchor=(0.85, 2.4), prop={"size": 16})
        else:
            ax[0].legend(bbox_to_anchor=(0.8, 2.2), prop={"size": 16})
        fig.tight_layout(pad=2)
        return fig, ax
