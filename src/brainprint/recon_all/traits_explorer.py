import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from brainprint.atlas import Atlas
from brainprint.protocol import Protocol
from brainprint.recon_all.execution_configuration import ExecutionConfiguration
from brainprint.recon_all.results import ReconAllResults
from brainprint.recon_all.utils import (
    SUBJECT_CATEGORICAL_TRAITS,
    SUBJECT_NUMERICAL_TRAITS,
    SUBJECT_TRAITS,
    get_default_exports_destination,
    plot_nii,
    regional_stats_to_nifti,
)
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


class TraitsExploration:
    PAIRPLOT_KWARGS: dict = {"diag_kind": "kde", "plot_kws": {"alpha": 0}}
    PAIRPLOT_TITLE: str = "Target Subject Attributes Pairwise Relationships"
    DECOMPOSITION_GRID_KWARGS: Dict[str, Any] = {
        "color": "grey",
        "alpha": 0.4,
    }
    DECOMPOSITION_CIRCLE_KWARGS: Dict[str, Any] = {
        "edgecolor": "grey",
        "fill": False,
        "alpha": 0.8,
    }
    TRAITS: List[str] = SUBJECT_TRAITS
    NUMERICAL_TRAITS: List[str] = SUBJECT_NUMERICAL_TRAITS
    CATEGORICAL_TRAITS: List[str] = SUBJECT_CATEGORICAL_TRAITS
    TRAIT_CATEGORY: Dict[str, str] = {
        "Physiological": [
            "Sex_M",
            "Age (years)",
            "Weight (kg)",
            "Height (cm)",
            "Dominant Hand_Left",
            "Dominant Hand_Right",
        ],
        "BFI": [
            "Agreeableness",
            "Conscientiousness",
            "Extraversion",
            "Neuroticism",
            "Openness to Experience",
        ],
    }

    def __init__(
        self,
        standardization: BaseEstimator = RobustScaler,
        dummy_encode: bool = True,
    ) -> None:
        # Read recon-all results.
        self.results = ReconAllResults(
            questionnaire_only=True,
            protocol=Protocol.BASE,
            atlas=Atlas.DESTRIEUX,
            configuration=ExecutionConfiguration.DEFAULT,
            multi_only=False,
        )

        # Select only the last scan for each subject.
        self.run_ids = (
            self.results.context.reset_index()
            .groupby("Subject ID")
            .last()["Run ID"]
        )
        self.raw_anatomical_statistics = self.results.clean.loc[self.run_ids]

        # Extract target features for the selected subjects.
        self.subject_ids = self.results.context.loc[
            self.raw_anatomical_statistics.index, "Subject ID"
        ]
        self.raw_subject_traits = self.results.questionnaire.loc[
            self.subject_ids, self.TRAITS
        ]
        self.subject_traits = self.raw_subject_traits.set_index(
            self.raw_anatomical_statistics.index
        )

        # Standardize numerical attributes.
        self.standardization = standardization
        self.standardized_anatomical_statistics = None
        self.standardized_subjects_traits = None
        if self.standardization is not None:
            self.standardized_anatomical_statistics = (
                self.raw_anatomical_statistics.copy()
            )
            self.standardized_subjects_traits = self.subject_traits.copy()
            self.standardized_anatomical_statistics.loc[
                :, :
            ] = standardization().fit_transform(self.raw_anatomical_statistics)
            self.standardized_subjects_traits.loc[
                :, self.NUMERICAL_TRAITS
            ] = standardization().fit_transform(
                self.raw_subject_traits.loc[:, self.NUMERICAL_TRAITS]
            )

        self.anatomical_statistics = (
            self.standardized_anatomical_statistics
            if self.standardization
            else self.raw_anatomical_statistics
        ).copy()
        self.subject_traits = (
            self.standardized_subjects_traits
            if self.standardization
            else self.raw_subject_traits
        ).copy()

        # Dummy encode categorical attributes.
        self.dummy_encode = dummy_encode
        if self.dummy_encode:
            self.subject_traits = pd.get_dummies(
                self.subject_traits,
                drop_first=True,
            )

    def plot_attributes_scatter_matrix(self, **kwargs) -> sns.PairGrid:
        kwargs = {**self.PAIRPLOT_KWARGS, **kwargs}
        numerical_traits = self.subject_traits[self.NUMERICAL_TRAITS]
        pairplot = sns.pairplot(numerical_traits, **kwargs)
        pairplot.map_lower(sns.kdeplot, levels=5, color="black")
        pairplot.map_lower(sns.scatterplot, color="blue")
        pairplot.map_upper(sns.histplot)
        # Add title
        pairplot.fig.subplots_adjust(top=0.95)
        title = kwargs.get("title", self.PAIRPLOT_TITLE)
        pairplot.fig.suptitle(title)
        return pairplot

    def plot_decomposition(
        self,
        estimator,
        circle: bool = True,
        circle_kwargs: Dict[str, Any] = None,
        circle_radius: float = 1,
        grid: bool = True,
        grid_kwargs: Dict[str, Any] = None,
        feature_arrow_color: str = "blue",
        target_arrow_color: str = "red",
        feature_labels: bool = False,
        feature_label_color: str = "blue",
        target_labels: bool = True,
        target_label_color: str = "red",
        figure_size: Tuple[int, int] = (12, 12),
        axis_limit: float = 1.1,
        **estimator_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:

        # Decompose.
        decomposition = estimator(n_components=2, **estimator_kwargs)
        numerical_traits = self.subject_traits[self.NUMERICAL_TRAITS]
        decomposition.fit(self.anatomical_statistics, numerical_traits)

        # Plot the decomposition in 2D.
        fig, ax = plt.subplots(figsize=figure_size)

        ax.set(xlim=(-axis_limit, axis_limit), ylim=(-axis_limit, axis_limit))

        # Plot features.
        for i in range(len(decomposition.x_rotations_)):
            x = decomposition.x_rotations_[i, 0]
            y = decomposition.x_rotations_[i, 1]
            ax.arrow(0, 0, x, y, color=feature_arrow_color)
            if feature_labels:
                label_text = decomposition.x_names_[i]
                ax.text(x, y, label_text, color=feature_label_color)
        # Plot target.
        for i in range(len(decomposition.y_rotations_)):
            x = decomposition.y_rotations_[i, 0]
            y = decomposition.y_rotations_[i, 1]
            ax.arrow(
                0,
                0,
                x,
                y,
                color=target_arrow_color,
            )
            if target_labels:
                label_text = self.subject_traits.columns[i]
                ax.text(x, y, label_text, color=target_label_color)

        # Style the plot.
        if circle:
            circle_kwargs = {} if circle_kwargs is None else circle_kwargs
            circle_kwargs = {
                **self.DECOMPOSITION_CIRCLE_KWARGS,
                **circle_kwargs,
            }
            circle_patch = plt.Circle((0, 0), circle_radius, **circle_kwargs)
            ax.add_patch(circle_patch)
        if grid:
            grid_kwargs = {} if grid_kwargs is None else grid_kwargs
            grid_kwargs = {**self.DECOMPOSITION_GRID_KWARGS, **grid_kwargs}
            ax.grid(**grid_kwargs)

        return fig, ax

    def reduce_traits(
        self, method: BaseEstimator = PCA, n_components: int = 1
    ) -> pd.DataFrame:
        """
        Reduce the regions to a lower dimensionality.
        """
        reduced = pd.DataFrame(
            index=self.subject_traits.index,
            columns=self.TRAIT_CATEGORY.keys(),
        )
        for category, traits in self.TRAIT_CATEGORY.items():
            category_data = self.subject_traits.loc[:, traits]
            reduced[category] = method(
                n_components=n_components
            ).fit_transform(category_data)
        return reduced

    def reduce_metrics(
        self, method: BaseEstimator = PCA, n_components: int = 1
    ) -> pd.DataFrame:
        regional_stats = self.anatomical_statistics.stack(
            ["Hemisphere", "Region Name"]
        )
        reduced_values = method(n_components=n_components).fit_transform(
            regional_stats
        )
        columns = ["PC{}".format(i) for i in range(1, n_components + 1)]
        return pd.DataFrame(
            index=regional_stats.index,
            columns=columns,
            data=reduced_values,
        )

    def calculate_reduced_correlation(
        self, destination: Path = Path("./corr.nii")
    ):
        reduced_stats = self.reduce_metrics()
        bfi = self.reduce_traits()["BFI"]
        hemispheres = reduced_stats.index.levels[-2]
        region_names = reduced_stats.index.levels[-1]
        correlations_index = pd.MultiIndex.from_product(
            [hemispheres, region_names]
        )
        correlations = pd.Series(
            index=correlations_index,
            name="Correlations",
            dtype=float,
        )
        for hemisphere in hemispheres:
            for region_name in region_names:
                regional_stats = reduced_stats.xs(
                    (hemisphere, region_name),
                    level=("Hemisphere", "Region Name"),
                )
                regional_stats = regional_stats.set_index(
                    self.results.context.loc[
                        regional_stats.index, "Subject ID"
                    ]
                ).squeeze()
                correlations.loc[
                    (hemisphere, region_name)
                ] = regional_stats.corr(bfi)
        if destination:
            regional_stats_to_nifti(correlations, destination=destination)
            return destination
        return correlations

    def calculate_pearson_r(
        self, correction_method: Optional[str] = "fdr_bh"
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        pairs = itertools.product(
            self.NUMERICAL_TRAITS, self.anatomical_statistics.columns
        )
        # Create dataframes to store results.
        correlation_coefficients = pd.DataFrame(
            index=self.anatomical_statistics.columns,
            columns=self.NUMERICAL_TRAITS,
            dtype=float,
        )
        p_values = correlation_coefficients.copy()

        # Calculate correlations.
        for trait, metric in tqdm(pairs, total=p_values.size):
            trait_values = self.subject_traits[trait]
            metric_values = self.anatomical_statistics[metric]
            statistic, p_value = pearsonr(trait_values, metric_values)
            correlation_coefficients.loc[metric, trait] = statistic
            p_values.loc[metric, trait] = p_value

        # Correct for multiple comparisons.
        if correction_method is not None:
            # Single correction method.
            if isinstance(correction_method, str):
                _, corrected_values, _, _ = multipletests(
                    p_values.stack(), method=correction_method
                )
                p_values = pd.Series(
                    corrected_values, index=p_values.stack().index
                ).unstack()

            # Multiple correction methods.
            elif isinstance(correction_method, Iterable):
                raw_values = p_values.copy()
                p_values = {
                    method: p_values.copy() for method in correction_method
                }
                for method in correction_method:
                    _, corrected_values, _, _ = multipletests(
                        raw_values.stack(), method=method
                    )
                    p_values[method] = pd.Series(
                        corrected_values, index=raw_values.stack().index
                    ).unstack()

        return correlation_coefficients, p_values

    def create_correlation_figures(self, alpha: float = 0.05):
        exports = get_default_exports_destination() / "correlations"
        coefficients, p_values = self.calculate_pearson_r()
        for metric in coefficients.index.levels[-1]:
            for trait in coefficients.columns:
                significance_mask = (
                    p_values.xs(metric, level="Metric")[trait] < alpha
                )
                values = coefficients.xs(metric, level="Metric")[trait].copy()
                values[~significance_mask] = 0
                if values[significance_mask].size > 0:
                    name = f"{trait.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{metric}"
                    png_destination = exports / f"{name}.png"
                    nii_destination = exports / f"{name}.nii"
                    title = f"{metric} Correlation with {trait}"
                    regional_stats_to_nifti(
                        values, destination=nii_destination
                    )
                    try:
                        plot_nii(
                            nii_destination,
                            destination=png_destination,
                            title=title,
                        )
                    except ValueError as e:
                        print(
                            f"Failed to plot {trait} and {metric} correlation with the following exception:\n{e}"
                        )
                        continue

    def plot_corr_surface(
        self,
        destination: Path = None,
    ):
        correlation_nii = self.calculate_reduced_correlation()
        plot_nii(
            correlation_nii,
            destination,
            title="PC Correlations",
            subtitle="sMRI / Personality",
        )
