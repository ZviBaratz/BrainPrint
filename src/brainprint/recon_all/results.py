"""
Definition of the :class:`ReconAllResults` class.
"""
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import pingouin as pg
from brainprint.atlas import Atlas
from brainprint.protocol import Protocol, infer_protocol
from brainprint.recon_all import logs
from brainprint.recon_all.execution_configuration import (
    CONFIGURATIONS_LABELS,
    ExecutionConfiguration,
    infer_execution_configurations,
)
from brainprint.recon_all.read import (
    DEFAULT_CONFIGURATIONS_PATH,
    DEFAULT_CONTEXT_PATH,
    DEFAULT_QUESTIONNAIRE_PATH,
    DEFAULT_RESULTS_PATH,
    read_configurations,
    read_context,
    read_results,
)
from brainprint.recon_all.utils import (
    SURFACE_REGISTRATION,
    get_default_cache_dir,
    plot_nii,
    regional_stats_to_nifti,
)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# matplotlib configuations
SMALL_FONT_SIZE = 10
MEDIUM_FONT_SIZE = 12
BIG_FONT_SIZE = 16

plt.rc("font", size=SMALL_FONT_SIZE)
plt.rcParams["font.family"] = "Times New Roman"  # controls default text font
plt.rc("axes", titlesize=SMALL_FONT_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_FONT_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_FONT_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIG_FONT_SIZE)  # fontsize of the figure title


class ReconAllResults:
    """
    Represents the results of the recon-all pipeline, as exported from pylabber
    to three CSV files:
    - ``results.csv``: the raw results of the pipeline.
    - ``context.csv``: the research context of each run.
    - ``configurations.csv``: the execution configuration of each run.
    """

    def __init__(
        self,
        results_path: Path = DEFAULT_RESULTS_PATH,
        context_path: Path = DEFAULT_CONTEXT_PATH,
        configurations_path: Path = DEFAULT_CONFIGURATIONS_PATH,
        questionnaire_path: Path = DEFAULT_QUESTIONNAIRE_PATH,
        atlas: Atlas = None,
        protocol: Union[Protocol, Iterable[Protocol]] = None,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ] = None,
        completed_only: bool = True,
        multi_only: bool = True,
        questionnaire_only: bool = True,
    ) -> None:
        """
        Initialize a :class:`ReconAllResults` instance.

        Parameters
        ----------
        results_path : Path, optional
            Raw results CSV path, by default DEFAULT_RESULTS_PATH
        context_path : Path, optional
            Research context information CSV path, by default
            DEFAULT_CONTEXT_PATH
        configurations_path : Path, optional
            Execution configuration CSV path, by default
            DEFAULT_CONFIGURATIONS_PATH
        atlas : Atlas, optional
            Atlas to filter the results by, by default None
        protocol : Protocol, optional
            Protocols to filter the results by, by default None
        configuration : ExecutionConfiguration, optional
            Execution configurations to filter the results by, by default None
        completed_only : bool, optional
            Whether to filter the results by completed runs (i.e. runs for
            which all *configuration* executions exist), by default True
        multi_only : bool, optional
            Whether to filter the results by participants who have multiple
            acquisitions, by default True
        """
        # Mediate input arguments
        self.results_path = results_path
        self.context_path = context_path
        self.run_configurations_path = configurations_path
        self.questionnaire_path = questionnaire_path
        self.multi_only = multi_only
        self.completed_only = completed_only
        self.atlas: Atlas = atlas
        self.protocol: Iterable[Protocol] = (
            [protocol] if isinstance(protocol, Protocol) else protocol
        )
        self.configuration: Iterable[ExecutionConfiguration] = (
            [configuration]
            if isinstance(configuration, ExecutionConfiguration)
            else configuration
        )

        #
        # Read the results
        #
        # Read anatomical statistics.
        self.raw_results: pd.DataFrame = read_results(self.results_path)
        # Read execution configurations.
        self.run_configurations: pd.DataFrame = read_configurations(
            self.run_configurations_path
        )
        # Read the research context.
        self.raw_context: pd.DataFrame = read_context(self.context_path)
        # Read questionnaire scores.
        self.questionnaire = pd.read_csv(
            self.questionnaire_path,
            header=0,
            index_col=0,
            dtype={
                "Sex": "category",
                "Dominant Hand": "category",
            },
        )
        for column_name in ("Age (years)", "Height (cm)", "Weight (kg)"):
            self.questionnaire[column_name] = self.questionnaire[
                column_name
            ].astype(int)

        #
        # Infer the protocol and execution configuation name
        #
        #  Add "Protocol" column to the context dataframe.
        self.raw_context["Protocol"] = infer_protocol(self.raw_context)
        # Add "Configuration" column to the context dataframe.
        self.raw_context["Configuration"] = infer_execution_configurations(
            self.run_configurations
        )

        #
        # Filter the results according to the input arguments
        #
        self.clean: pd.DataFrame = self.filter_results(
            results=self.raw_results,
            context=self.raw_context,
            atlas=self.atlas,
            protocol=self.protocol,
            configuration=self.configuration,
            completed_only=self.completed_only,
            multi=self.multi_only,
            questionnaire_only=questionnaire_only,
        )
        self.context: pd.DataFrame = self.raw_context.loc[self.clean.index]

    def get_protocol_mask(
        self,
        protocol: Union[Protocol, Iterable[Protocol]],
        context: pd.DataFrame = None,
        premask: pd.Series = None,
    ) -> pd.Series:
        """
        Return a mask of the *context* indices (i.e. run IDs) that match the
        given *protocol*.

        Parameters
        ----------
        protocol : Union[Protocol, Iterable[Protocol]]
            Protocols to filter the results by
        context : pd.DataFrame, optional
            Context to filter, by default None

        Returns
        -------
        pd.Series
            Protocols mask
        """
        # Mediate input.
        protocol = (
            [protocol.value]
            if isinstance(protocol, Protocol)
            else [p.value for p in protocol]
        )
        context = self.context if context is None else context
        if premask is not None:
            context = context[premask.reindex(context.index, fill_value=False)]

        # Log start.
        start_message = logs.PROTOCOL_FILTER_START.format(
            n=len(context.index), protocol=protocol
        )
        logger.debug(start_message)

        # Create protocol mask.
        mask = context["Protocol"].isin(protocol)

        # Log end.
        end_message = logs.PROTOCOL_FILTER_END.format(
            n_match=mask.sum(), n_total=len(mask), protocol=protocol
        )
        logger.info(end_message)

        return mask

    def filter_by_protocol(
        self,
        protocol: Union[Protocol, Iterable[Protocol]],
        results: pd.DataFrame = None,
        context: pd.DataFrame = None,
    ) -> pd.DataFrame:
        results = self.clean if results is None else results
        context = self.context if context is None else context
        mask = self.get_protocol_mask(
            protocol, context=context.loc[results.index]
        )
        return results[mask]

    def select_atlas(
        self, atlas: Atlas, results: pd.DataFrame = None
    ) -> pd.DataFrame:
        results = self.clean if results is None else results

        # Log start.
        start_message = logs.ATLAS_FILTER_START.format(
            atlas=atlas, n=len(results.index)
        )
        logger.debug(start_message)

        # Select atlas results.
        results = results.xs(atlas.value, level="Atlas", axis=1)

        # Drop and report any rows with missing values.
        na_filtered = results.dropna(axis=0, how="any")
        na_found = len(results.index) - len(na_filtered.index)
        if na_found > 0:
            partial_results_message = logs.PARTIAL_ATLAS_RESULTS.format(
                atlas=atlas,
                n=len(results.index),
                n_filtered=len(na_filtered.index),
                n_dropped=na_found,
            )
            logger.info(partial_results_message)

        return na_filtered

    def get_configuration_mask(
        self,
        configuration: ExecutionConfiguration,
        context: pd.DataFrame = None,
        premask: pd.Series = None,
    ) -> pd.Series:
        """
        Return a mask of the *context* indices (i.e. run IDs) that match the
        given *configuration*.

        Parameters
        ----------
        configuration : ExecutionConfiguration
            Configuration to filter the results by
        context : pd.DataFrame, optional
            Context to create mask from, by default None

        Returns
        -------
        pd.Series
            Configuration mask
        """
        # Mediate input.
        context = self.context if context is None else context
        if premask is not None:
            context = context[premask.reindex(context.index, fill_value=False)]
        configuration = (
            [configuration.value]
            if isinstance(configuration, ExecutionConfiguration)
            else [c.value for c in configuration]
        )

        # Log start.
        start_message = logs.CONFIGURATION_FILTER_START.format(
            n=len(context), execution_configuration=configuration
        )
        logger.debug(start_message)

        # Create configuation mask.
        mask = context["Configuration"].isin(configuration)

        # Log end.
        end_message = logs.CONFIGURATION_FILTER_END.format(
            n_match=mask.sum(),
            n_total=mask.size,
            execution_configuration=configuration,
        )
        logger.info(end_message)

        return mask

    def filter_by_configuration(
        self,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ],
        results: pd.DataFrame = None,
        context: pd.DataFrame = None,
    ) -> pd.DataFrame:
        # Mediate inputs.
        results = self.clean if results is None else results
        context = self.context if context is None else context
        context = context.loc[results.index]
        mask = self.get_configuration_mask(
            configuration=configuration, context=context
        )
        return results[mask]

    def get_scan_id_mask(
        self,
        scan_ids: Iterable[int],
        context: pd.DataFrame = None,
        premask: pd.Series = None,
    ) -> pd.Series:
        # Mediate input.
        context = self.context if context is None else context
        if premask is not None:
            context = context[premask.reindex(context.index, fill_value=False)]

        # Log start.
        start_scan_id_message = logs.SCAN_ID_SELECTION_START.format(
            n_runs=len(context.index), n_scans=len(scan_ids)
        )
        logger.debug(start_scan_id_message)

        # Create scan ID mask.
        mask = context["Scan ID"].isin(scan_ids)

        # Log end.
        subject_ids = context.loc[mask, "Subject ID"].unique()
        end_scan_id_message = logs.SCAN_ID_SELECTION_END.format(
            n_runs_selected=mask.sum(),
            n_runs_total=len(context.index),
            n_scans=len(scan_ids),
            n_subjects=len(subject_ids),
        )
        logger.info(end_scan_id_message)

        return mask

    def get_mask(
        self,
        protocol: Union[Protocol, Iterable[Protocol]] = None,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ] = None,
        context: pd.DataFrame = None,
        completed_only: bool = False,
        questionnaire_only: bool = False,
        multi: bool = False,
        scan_ids: Iterable[int] = None,
    ):
        """
        Filter the results by the given parameters.

        Parameters
        ----------
        atlas : Atlas, optional
            Atlas to filter the results by, by default None
        protocol : Union[Protocol, Iterable[Protocol]], optional
            Protocol to filter the results by, by default None
        configuration : Union[ ExecutionConfiguration, Iterable[ExecutionConfiguration] ], optional
            Configuration to filter the results by, by default None
        context : pd.DataFrame, optional
            Context to filter the results by, by default None
        completed_only : bool, optional
            Whether to filter to only completed runs, by default False
        multi : bool, optional
            Whether to filter to only multi-run runs, by default False
        scan_ids : Iterable[int], optional
            Scan IDs to filter the results by, by default None

        Returns
        -------
        pd.DataFrame
            Filtered results
        """
        context = self.context if context is None else context
        # Log start.
        start_message = logs.FILTER_START.format(n=len(context.index))
        logger.info(start_message)

        # Calculate mask.
        mask = pd.Series(True, index=context.index)
        if protocol is not None:
            mask &= self.get_protocol_mask(protocol, context=context)
            context = context[mask]
        if configuration is not None:
            mask &= self.get_configuration_mask(
                configuration, context=context, premask=mask
            )
        if scan_ids is not None:
            mask &= self.get_scan_id_mask(
                scan_ids, context=context, premask=mask
            )
        if completed_only:
            mask &= self.get_completed_mask(context=context, premask=mask)
        if multi:
            mask &= self.get_multi_mask(context=context, premask=mask)
        if questionnaire_only:
            mask &= context["Subject ID"].isin(self.questionnaire.index)

        # Log end.
        end_message = logs.FILTER_END.format(
            n_total=mask.size, n_match=mask.sum()
        )
        logger.info(end_message)

        return mask

    def filter_results(
        self,
        atlas: Atlas = None,
        protocol: Union[Protocol, Iterable[Protocol]] = None,
        configuration: Union[
            ExecutionConfiguration, Iterable[ExecutionConfiguration]
        ] = None,
        results: pd.DataFrame = None,
        context: pd.DataFrame = None,
        completed_only: bool = False,
        multi: bool = False,
        questionnaire_only: bool = False,
        scan_ids: Iterable[int] = None,
    ):
        """
        Filter the results by the given parameters.

        Parameters
        ----------
        atlas : Atlas, optional
            Atlas to filter the results by, by default None
        protocol : Union[Protocol, Iterable[Protocol]], optional
            Protocol to filter the results by, by default None
        configuration : Union[ ExecutionConfiguration, Iterable[ExecutionConfiguration] ], optional
            Configuration to filter the results by, by default None
        results : pd.DataFrame, optional
            Results to filter, by default None
        context : pd.DataFrame, optional
            Context to filter the results by, by default None
        completed_only : bool, optional
            Whether to filter to only completed runs, by default False
        multi : bool, optional
            Whether to filter to only multi-run runs, by default False
        scan_ids : Iterable[int], optional
            Scan IDs to filter the results by, by default None

        Returns
        -------
        pd.DataFrame
            Filtered results
        """
        # Mediate input.
        results = self.clean if results is None else results
        context = self.context if context is None else context

        # Calculate mask.
        mask = self.get_mask(
            protocol=protocol,
            configuration=configuration,
            context=context,
            completed_only=completed_only,
            multi=multi,
            scan_ids=scan_ids,
            questionnaire_only=questionnaire_only,
        )

        # Select atlas statistics.
        filtered = results[mask]
        if atlas is not None:
            filtered = self.select_atlas(atlas, filtered)

        return filtered.copy()

    def check_same_subject(self, run_1: int, run_2: int) -> bool:
        """
        Checks whether two runs are derived from the same subject.

        Parameters
        ----------
        run_1 : int
            Run ID #1
        run_2 : int
            Run ID #2

        Returns
        -------
        bool
            Whether the two runs are derived from the same subject or not
        """
        subject_1 = self.context.loc[run_1, "Subject ID"]
        subject_2 = self.context.loc[run_2, "Subject ID"]
        return subject_1 == subject_2

    def check_same_protocol(self, run_1: int, run_2: int) -> bool:
        """
        Checks whether two runs are derived from the same protocol.

        Parameters
        ----------
        run_1 : int
            Run ID #1
        run_2 : int
            Run ID #2

        Returns
        -------
        bool
            Whether the two runs are derived from the same protocol or not
        """
        protocol_1 = self.context.loc[run_1, "Protocol"]
        protocol_2 = self.context.loc[run_2, "Protocol"]
        return protocol_1 == protocol_2

    def get_completed_mask(
        self, context: pd.DataFrame = None, premask: pd.Series = None
    ) -> pd.Series:
        """
        Returns a mask of completed runs, i.e. runs that belong to scans that
        have been analysed with all registered execution configuations.

        Parameters
        ----------
        context : pd.DataFrame, optional
            Context dataframe. If None, the instance's context dataframe is
            used

        Returns
        -------
        pd.Series
            Completed runs mask
        """
        # Mediate input.
        context = self.context if context is None else context
        if premask is not None:
            context = context[premask.reindex(context.index, fill_value=False)]

        if self.configuration is None:
            return pd.Series(True, index=context.index)

        # Log start.
        start_message = logs.COMPLETED_FILTER_START.format(
            n_configurations=len(self.configuration)
        )
        logger.debug(start_message)

        # Create mask by run counts per scan ID.
        run_counts = context.groupby("Scan ID")["Configuration"].count()
        completed_scan_ids = run_counts[
            run_counts >= len(self.configuration)
        ].index
        scan_id_mask = context["Scan ID"].isin(completed_scan_ids)

        # Log end.
        end_message = logs.COMPLETED_FILTER_END.format(
            n_runs=scan_id_mask.sum(),
            n_scans=len(completed_scan_ids),
            n_configurations=len(self.configuration),
        )
        logger.info(end_message)

        return scan_id_mask

    def filter_completed(
        self, results: pd.DataFrame = None, context: pd.DataFrame = None
    ) -> pd.DataFrame:
        results = self.clean if results is None else results
        context = self.context if context is None else context
        context = context.loc[results.index]
        scan_id_mask = self.get_completed_mask(context)
        return results[scan_id_mask]

    def select_metrics(
        self, metrics: List[str], results: pd.DataFrame = None
    ) -> pd.DataFrame:
        results = self.clean if results is None else results
        mask = results.columns.isin(metrics, level="Metric")
        return results.loc[:, mask]

    def sample_subject_scans(
        self, context: pd.DataFrame = None, mode: str = "last"
    ) -> pd.DataFrame:
        # Mediate input.
        context = self.context if context is None else context

        # Log start.
        n_scans = context["Scan ID"].unique().size
        start_message = logs.SINGLE_SCAN_SELECTION_START.format(
            n_runs=len(context.index), n_scans=n_scans
        )
        logger.debug(start_message)

        # Sample scans according to *mode*.
        sorted_context = context.reset_index().sort_values(
            ["Subject ID", "Session Time"]
        )
        grouped_context = sorted_context.groupby("Subject ID", as_index=False)
        if mode == "first":
            sample = grouped_context.first()
        elif mode == "last":
            sample = grouped_context.last()
        else:
            message = logs.INVALID_SAMPLE_MODE.format(mode=mode)
            raise ValueError(message)
        sample = sample.set_index("Run ID")
        context = context[context["Scan ID"].isin(sample["Scan ID"])]

        # Log end.
        n_scans = context["Scan ID"].unique().size
        n_subjects = context["Subject ID"].unique().size
        end_message = logs.SINGLE_SCAN_SELECTION_END.format(
            n_runs=len(context.index), n_scans=n_scans, n_subjects=n_subjects
        )
        logger.info(end_message)

        return context

    def split_scans(
        self,
        context: pd.DataFrame = None,
        single_mode: str = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the raw data into training and testing sets by scan ID.

        Parameters
        ----------
        context : pd.DataFrame, optional
            A context dataframe.
        single_mode : str, optional
            If True, only a single scan will be used per subject. Options are
            'first' or 'last'. If False, all scans will be used. Default is
            False

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Train scan IDs, Test scan IDs
        """
        # Mediate input.
        context = self.context if context is None else context

        # Sample one scan per subject.
        if single_mode:
            context = self.sample_subject_scans(
                context=context, mode=single_mode
            )

        # Create train/test split by scan IDs.
        scan_ids = context["Scan ID"].unique()
        return train_test_split(scan_ids, **kwargs)

    def split(
        self,
        execution_configuration: ExecutionConfiguration,
        target: str,
        single_mode: str = False,
        metrics: List[str] = None,
        context: pd.DataFrame = None,
        results: pd.DataFrame = None,
        **kwargs,
    ):
        """
        Split the clean data into training and testing sets by scan ID.

        Parameters
        ----------
        execution_configuration : ExecutionConfiguration
            The execution configuration to select from.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Train dataset, Test dataset
        """
        # Mediate input.
        context = self.context if context is None else context
        results = self.clean if results is None else results

        # Select metrics.
        if metrics is not None:
            results = self.select_metrics(metrics, results=results)

        # Split scan IDs to train and test sets.
        train_scan_ids, test_scan_ids = self.split_scans(
            context,
            single_mode=single_mode,
            **kwargs,
        )

        # Query train and test feature and target sets for the given scan IDs
        # and execution configuration.
        X_train = self.filter_results(
            configuration=execution_configuration,
            scan_ids=train_scan_ids,
        )
        X_test = self.filter_results(
            configuration=execution_configuration,
            scan_ids=test_scan_ids,
        )
        y_train = self.context.loc[X_train.index, target]
        y_test = self.context.loc[X_test.index, target]
        return X_train, X_test, y_train, y_test

    def get_multi_mask(
        self,
        context: pd.DataFrame = None,
        premask: pd.Series = None,
    ) -> pd.Series:
        """
        Returns a mask of runs derived from subjects with more than one
        acquisition.

        Parameters
        ----------
        context : pd.DataFrame, optional
            A context dataframe

        Returns
        -------
        pd.Series
            A mask of runs derived from subjects with more than one acquisition
        """
        # Mediate input.
        context = self.context if context is None else context
        if premask is not None:
            context = context[premask.reindex(context.index, fill_value=False)]

        # Log start.
        start_message = logs.MULTI_SUBJECTS_START.format(n_runs=len(context))
        logger.debug(start_message)

        # Calculate mask.
        n_scans_by_subject = (
            context.drop_duplicates(subset=["Subject ID", "Session ID"])
            .groupby("Subject ID")
            .size()
        )
        multi_mask = n_scans_by_subject > 1
        multi_run_subject_ids = multi_mask[multi_mask].index
        mask = context["Subject ID"].isin(multi_run_subject_ids)

        # Log end.
        end_message = logs.MULTI_SUBJECTS_END.format(
            n_subjects=len(multi_run_subject_ids),
            n_runs=mask.sum(),
            n_scans=len(context.loc[mask, "Scan ID"].unique()),
        )
        logger.info(end_message)

        return mask

    def unstack(self, results: pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a dataframe with the hemisphere and region column levels unstacked.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Anatomical statistics by run dataframe, by default None

        Returns
        -------
        pd.DataFrame
            Unstacked anatomical statistics dataframe
        """
        results = self.clean if results is None else results
        return (
            self.clean.unstack()
            .reset_index()
            .rename({0: "Value"}, axis="columns")
        )

    def plot_mean_pairplot(
        self,
        metric: str = "Average Thickness",
        results: pd.DataFrame = None,
        destination: Path = None,
        force_create: bool = False,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Path]:
        return_existing = (
            destination is not None
            and destination.exists()
            and not force_create
        )
        if return_existing:
            return destination
        results = self.clean if results is None else results
        unstacked_results = self.unstack(results)
        unstacked_results["Configuration"] = unstacked_results.apply(
            lambda row: self.context.loc[row["Run ID"], "Configuration"],
            axis=1,
        )
        avg_thickness_long = unstacked_results[
            unstacked_results["Metric"] == metric
        ].copy()
        order = [
            ExecutionConfiguration.MPRAGE_AND_3T_AND_T2.value,
            ExecutionConfiguration.MPRAGE_AND_T2.value,
            ExecutionConfiguration.T2.value,
            ExecutionConfiguration.DEFAULT.value,
            ExecutionConfiguration.FLAIR.value,
            ExecutionConfiguration.MPRAGE_AND_FLAIR.value,
            ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR.value,
        ]
        avg_thickness_long["Lateral Region Name"] = avg_thickness_long.apply(
            lambda row: f"{row['Hemisphere'][0]}_{row['Region Name']}", axis=1
        )
        fig, ax = plt.subplots(figsize=(15, 8))
        pg.plot_paired(
            data=avg_thickness_long,
            dv="Value",
            within="Configuration",
            subject="Lateral Region Name",
            dpi=150,
            ax=ax,
            order=order,
            boxplot_in_front=True,
        )
        fig.suptitle(
            f"Effect of FreeSurfer Execution Configuration on {metric} by Region"
        )
        xticklabels = [
            CONFIGURATIONS_LABELS[ExecutionConfiguration(x)] for x in order
        ]
        ax.set(ylabel=metric, xticklabels=xticklabels)
        plt.setp(ax.lines, alpha=0.5)
        plt.xticks(rotation=30)
        plt.tight_layout()
        if destination is not None:
            fig.savefig(destination)
        return fig, ax

    def calculate_difference_of_means(
        self,
        configuration: ExecutionConfiguration,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
        metrics: Optional[Iterable[str]] = None,
    ) -> pd.Series:
        configuration_results = self.filter_by_configuration(configuration)
        reference_results = self.filter_by_configuration(reference)
        if metrics is not None:
            configuration_results = configuration_results.loc[
                :, configuration_results.columns.isin(metrics, level="Metric")
            ]
            reference_results = reference_results.loc[
                :, reference_results.columns.isin(metrics, level="Metric")
            ]
        return configuration_results.mean() - reference_results.mean()

    HEMISPHERE_DIFFERENCE_NAME_PATTERN: str = (
        "{configuration}_vs_{reference}_{hemisphere}_{metric}.nii.gz"
    )
    BRAIN_DIFFERENCE_NAME_PATTERN: str = (
        "{configuration}_vs_{reference}_{metric}.nii.gz"
    )

    def get_difference_of_means_nifti_path(
        self,
        configuration: ExecutionConfiguration,
        metric: str,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
        hemisphere: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """
        Get the path to the difference of means nifti file for the given
        configuration, metric, reference, and hemisphere (optional).

        Parameters
        ----------
        configuration : ExecutionConfiguration
            Execution configuration used
        metric : str
            The compared metric
        reference : ExecutionConfiguration, optional
            The reference execution configuration, by default
            ExecutionConfiguration.DEFAULT
        hemisphere : str, optional
            'L' or 'R' if values represent only a single hemisphere, by default
            None
        cache_dir : Path, optional
            Root differences results directory, by default None

        Returns
        -------
        Path
            Destination path for the given parameters
        """
        cache_dir = get_default_cache_dir() if cache_dir is None else cache_dir
        configuration = configuration.name.lower()
        reference = reference.name.lower()
        metric = metric.lower().replace(" ", "_")
        if hemisphere is not None:
            name = self.HEMISPHERE_DIFFERENCE_NAME_PATTERN.format(
                configuration=configuration,
                reference=reference,
                metric=metric,
                hemisphere=hemisphere,
            )
        else:
            name = self.BRAIN_DIFFERENCE_NAME_PATTERN.format(
                configuration=configuration,
                reference=reference,
                metric=metric,
            )
        path = cache_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_difference_of_means_df(
        self,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
        configurations: Optional[Iterable[ExecutionConfiguration]] = None,
        metrics: Optional[Iterable[str]] = None,
    ):
        if configurations is None:
            configurations = [
                ExecutionConfiguration.MPRAGE_AND_3T_AND_T2,
                ExecutionConfiguration.T2,
                ExecutionConfiguration.FLAIR,
                ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR,
            ]
        if metrics is None:
            metrics = [
                "Average Thickness",
                "Thickness StdDev",
                "Gray Matter Volume",
            ]
        differences = {
            configuration.value: self.calculate_difference_of_means(
                configuration=configuration,
                metrics=metrics,
                reference=reference,
            )
            for configuration in self.configuration
            if configuration in configurations
        }
        return pd.DataFrame(differences)

    def plot_difference_of_means_table(
        self,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
        configurations: Optional[List[ExecutionConfiguration]] = None,
        metrics: Optional[List[str]] = None,
    ):
        df = self.get_difference_of_means_df(
            reference=reference, configurations=configurations, metrics=metrics
        ).unstack()
        df.columns = df.columns.swaplevel(0, 1)
        df = (
            df.sort_index(axis=1)
            .reindex(
                labels=[
                    "Average Thickness",
                    "Thickness StdDev",
                    "Gray Matter Volume",
                ],
                axis=1,
                level=0,
            )
            .reindex(
                labels=[
                    ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR.value,
                    ExecutionConfiguration.FLAIR.value,
                    ExecutionConfiguration.T2.value,
                    ExecutionConfiguration.MPRAGE_AND_3T_AND_T2.value,
                ],
                axis=1,
                level=1,
            )
        )
        df.columns.names = ["Metric", "Configuration"]
        table = df.style.set_table_styles(
            [
                {
                    "selector": "",
                    "props": [
                        ("font-family", "Times New Roman"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "12px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "12px"),
                        ("border-color", "black"),
                        ("border-style", "solid"),
                        ("border-width", "1px"),
                    ],
                },
                {
                    "selector": "th.col_heading",
                    "props": [
                        ("text-align", "center"),
                    ],
                },
                {"selector": "th.row_heading", "props": "text-align: left;"},
            ]
        )
        first_configuration = df.columns.levels[1][0]
        for metric in df.columns.levels[0]:
            table = table.set_table_styles(
                {
                    (metric, first_configuration): [
                        {
                            "selector": "",
                            "props": [("border-left", "3px solid black")],
                        }
                    ]
                },
                overwrite=False,
            )
        first_region = df.index.levels[1][0]
        for hemisphere in df.index.levels[0]:
            table = table.set_table_styles(
                {
                    (hemisphere, first_region): [
                        {
                            "selector": "",
                            "props": [("border-top", "3px solid black")],
                        }
                    ]
                },
                overwrite=False,
                axis=1,
            )
        for column_name in df.columns:
            vmax = df[column_name].abs().max()
            table = table.background_gradient(
                cmap="RdYlGn", subset=[column_name], vmin=-vmax, vmax=vmax
            )
        return table

    def create_difference_of_means_niftis(
        self,
        cache_dir: Path = None,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
    ) -> List[Path]:
        """
        Create difference of means niftis for all metrics and all
        configurations.

        Parameters
        ----------
        cache_dir : Path, optional
            Root differences results directory, by default None
        reference : ExecutionConfiguration, optional
            The reference execution configuration, by default
            ExecutionConfiguration.DEFAULT

        Returns
        -------
        List[Path]
            Created nifti files
        """
        cache_dir = get_default_cache_dir() if cache_dir is None else cache_dir

        # Iterate execution configurations.
        paths = []
        for configuration in self.configuration:

            # Skip comparison of the reference with itself.
            if configuration is reference:
                continue

            # Calculate difference of means for all metric.
            difference_of_means = self.calculate_difference_of_means(
                configuration, reference=reference
            )

            # Iterate metrics and create nifti files.
            metrics = difference_of_means.index.unique(level="Metric")
            for metric in metrics:

                # Select current metric differences.
                metric_values = difference_of_means.xs(metric, level="Metric")

                # Create a nifti for each hemisphere.
                hemispheres = metric_values.index.unique(level="Hemisphere")
                for hemisphere in hemispheres:
                    destination = self.get_difference_of_means_nifti_path(
                        configuration=configuration,
                        metric=metric,
                        hemisphere=hemisphere,
                        cache_dir=cache_dir,
                    )
                    regional_stats_to_nifti(
                        metric_values,
                        hemisphere=hemisphere,
                        destination=destination,
                    )
                    paths.append(destination)

                # Create whole-brain nifti.
                destination = self.get_difference_of_means_nifti_path(
                    configuration=configuration,
                    metric=metric,
                    cache_dir=cache_dir,
                )
                regional_stats_to_nifti(metric_values, destination=destination)

        return paths

    def plot_difference_surface(
        self,
        destination: Path,
        configuration: ExecutionConfiguration,
        reference: ExecutionConfiguration = ExecutionConfiguration.DEFAULT,
        metric: str = "Average Thickness",
        force_create: bool = False,
        cache_dir: Path = Optional[None],
        reg_file: Path = SURFACE_REGISTRATION,
    ):
        if destination.exists() and not force_create:
            return destination
        else:
            self.create_difference_of_means_niftis(
                cache_dir=cache_dir, reference=reference
            )
        path = self.get_difference_of_means_nifti_path(
            configuration=configuration,
            reference=reference,
            metric=metric,
            cache_dir=cache_dir,
        )
        title = f"FreeSurfer-derived {metric}"
        subtitle = f"{configuration.value} vs. {reference.value}"
        return plot_nii(
            path,
            destination,
            title=title,
            subtitle=subtitle,
            reg_file=reg_file,
        )


def load_results(**kwargs) -> ReconAllResults:
    return ReconAllResults(
        atlas=Atlas.DESTRIEUX,
        protocol=Protocol.BASE,
        configuration=(
            ExecutionConfiguration.DEFAULT,
            ExecutionConfiguration.T2,
            ExecutionConfiguration.FLAIR,
            ExecutionConfiguration.MPRAGE_AND_T2,
            ExecutionConfiguration.MPRAGE_AND_FLAIR,
            ExecutionConfiguration.MPRAGE_AND_3T_AND_T2,
            ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR,
        ),
        **kwargs,
    )
