import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import imageio as iio
import nibabel as nib
import numpy as np
import pandas as pd
from mayavi import mlab
from nilearn import datasets
from PIL import Image, ImageDraw, ImageFont
from surfer import Brain, project_volume_data

ATLAS_FETCHERS: Dict[str, Callable] = {
    "destrieux": datasets.fetch_atlas_destrieux_2009,
}

TIMES_NEW_ROMAN: Path = Path(__file__).parent.parent.parent / "fonts/times.ttf"
BLACK = (0, 0, 0)


def flatten_label(hemisphere: str, region_name: str) -> str:
    region_name = region_name.replace("&", "_and_")
    return f"{hemisphere[0]} {region_name}"


def regional_stats_to_nifti(
    stats: pd.Series,
    atlas: str = "destrieux",
    destination: Path = None,
    hemisphere: str = None,
) -> nib.Nifti1Image:
    # Mediate hemisphere input.
    hemisphere = hemisphere if hemisphere is None else hemisphere.lower()[0]

    # Fix multi-indexing by hemisphere and region name.
    if isinstance(stats.index, pd.MultiIndex) and stats.index.nlevels == 2:
        stats.index = [
            flatten_label(hemi, region_name)
            for hemi, region_name in stats.index
        ]

    # Fetch atlas information.
    fetcher = ATLAS_FETCHERS[atlas]
    atlas_dict = fetcher()
    atlas_nii_path, atlas_labels = atlas_dict["maps"], atlas_dict["labels"]
    atlas_nii = nib.load(atlas_nii_path)
    atlas_data = atlas_nii.get_fdata()

    if hemisphere == "b":
        return [
            regional_stats_to_nifti(stats, atlas, hemisphere)
            for hemisphere in ["l", "r"]
        ]

    # Create projection.
    projection = np.zeros_like(atlas_data)
    for label_value, label_name in atlas_labels:

        is_other_hemisphere = False
        if hemisphere:
            is_other_hemisphere = not label_name.lower().startswith(hemisphere)
        is_background = label_name == "Background"
        is_medial_wall = "Medial_wall" in label_name
        if is_background or is_medial_wall or is_other_hemisphere:
            continue

        region_mask = atlas_data == label_value
        if region_mask.sum() == 0:
            print(f"Warning: {label_name} is empty.")
            continue

        try:
            projection[region_mask] = stats[label_name]
        except KeyError:
            print(
                f"Label {label_name} could not be found in the provided series!"
            )

    # Create NIfTI instance and save if destination is provided.
    nii = nib.Nifti1Image(projection, atlas_nii.affine)
    if destination is not None:
        nib.save(nii, destination)

    return nii


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    BOLD_OFF = "\033[22m"
    ITALIC = "\033[3m"
    ITALIC_OFF = "\033[23m"
    UNDERLINE = "\033[4m"
    UNDERLINE_OFF = "\033[24m"
    PINK_BACK = "\33[45m"
    PINK_FORE = "\33[35m"
    RED_BACK = "\33[41m"
    RED_FORE = "\33[31m"
    GREEN_BACK = "\33[42m"
    GREEN_FORE = "\33[32m"
    YELLOW_BACK = "\33[43m"
    YELLOW_FORE = "\33[33m"
    BLUE_BACK = "\33[44m"
    BLUE_FORE = "\33[34m"
    LIGHT_PINK_BACK = "\33[105m"
    LIGHT_PINK_FORE = "\33[95m"
    INVERSE = "\033[7m"
    INVERSE_OFF = "\033[27m"


def defaultdict_factory():
    return defaultdict(defaultdict_factory)


def get_default_exports_destination() -> Path:
    return Path(__file__).parent / "exports"


def get_default_cache_dir() -> Path:
    exports_destination = get_default_exports_destination()
    return exports_destination / "differences"


SUBJECT_TRAITS: List[str] = [
    # Physiological
    "Sex",
    "Age (years)",
    "Weight (kg)",
    "Height (cm)",
    "Dominant Hand",
    # Personality
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness to Experience",
    # # Lifestyle
    # "Years in Current Relationship",
    # "Number of Children",
    # "Do you have a pet?",
    # "Weekly workout hours",
    # "Cups of Non-diet Beverages per Day",
    # "Alcohol Consumption (days per week)",
    # "Social Meetings per Week",
    # "Frequency of Feeling Stress or Anxiety",
    # "Frequency of Nicotine Consumption",
    # "Frequency of Cannabis Consumption",
    # # Sleep
    # "PSQI",
    # # Heritage
    # "Cultural Descent",
    # "First Language",
    # "Are you 2nd or 3rd generation to holocaust survivors?",
    # # Education
    # "Psychometric Test Score",
    # "Learning difficulties",
]
SUBJECT_NUMERICAL_TRAITS: List[str] = [
    "Age (years)",
    "Height (cm)",
    "Weight (kg)",
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness to Experience",
]
SUBJECT_CATEGORICAL_TRAITS: List[str] = ["Sex", "Dominant Hand"]

SURFACE_REGISTRATION = None
try:
    SURFACE_REGISTRATION = (
        Path(os.environ["FREESURFER_HOME"]) / "average/mni152.register.dat"
    )
except KeyError:
    pass


def plot_nii(
    nii_path: Path,
    destination: Path,
    colormap: str = "RdYlGn",
    size: int = (1200, 1200),
    title: Optional[str] = None,
    title_size: int = 56,
    subtitle: Optional[str] = None,
    subtitle_size: int = 46,
    subtitle_padding: int = 5,
    reg_file: Path = SURFACE_REGISTRATION,
):
    if SURFACE_REGISTRATION is None:
        raise ValueError("No registration file provided or detected!")
    surf_data_lh = np.nan_to_num(
        project_volume_data(nii_path, hemi="lh", reg_file=str(reg_file))
    )
    surf_data_rh = np.nan_to_num(
        project_volume_data(nii_path, hemi="rh", reg_file=str(reg_file))
    )
    fig = [mlab.figure(size=size) for i in range(4)]
    brain = Brain(
        subject_id="fsaverage",
        hemi="split",
        surf="pial",
        views=["lat", "med"],
        background="white",
        figure=fig,
    )
    brain.add_data(surf_data_lh, hemi="lh", colormap=colormap)
    brain.add_data(surf_data_rh, hemi="rh", colormap=colormap)

    vmax = np.nanmax(np.abs(nib.load(nii_path).get_fdata()))
    brain.scale_data_colormap(0, vmax / 2, vmax, transparent=False, center=0)

    if destination is not None:
        # Export mayavi figure to png.
        brain.save_image(destination, mode="rgba")

        # Fix image spacing and multiple colorbars.
        image = iio.imread(destination)

        # Remove central colorbars.
        image = np.delete(image, range(size[0] - 300, size[0] - 75), axis=0)

        # Decrease vertical middle spacing.
        image = np.delete(image, range(size[1] - 50, size[1] + 50), axis=1)

        # Delete bottom-right colobar.
        image[-150:, size[1] :, :] = 0

        # Roll bottom left colorbar to center.
        image[-150:, :, :] = np.roll(
            image[-150:, :, :], size[0] // 2, axis=(0, 1)
        )

        # Override original file.
        iio.imsave(destination, image)

        # Add title and subtitle.
        image = Image.open(destination)
        draw = ImageDraw.Draw(image)

        if title is not None:
            title_font = ImageFont.truetype(str(TIMES_NEW_ROMAN), title_size)
            title_width, title_height = draw.textsize(title, font=title_font)
            figure_width, figure_height = image.size
            title_x = (figure_width - title_width) // 2
            title_y = figure_height // 30
            title_coords = title_x, title_y
            draw.text(xy=title_coords, text=title, fill=BLACK, font=title_font)

        if subtitle is not None:
            subtitle_font = ImageFont.truetype(
                str(TIMES_NEW_ROMAN), subtitle_size
            )
            subtitle_width, _ = draw.textsize(subtitle, font=subtitle_font)
            subtitle_x = (figure_width - subtitle_width) // 2
            subtitle_y = title_y + title_height + subtitle_padding
            subtitle_coords = subtitle_x, subtitle_y
            draw.text(
                xy=subtitle_coords,
                text=subtitle,
                fill=BLACK,
                font=subtitle_font,
            )

        image.save(destination)
        return fig
