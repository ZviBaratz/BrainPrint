import matplotlib.pyplot as plt
import pandas as pd
from nilearn import datasets, plotting

# from sklearn.preprocessing import StandardScaler

#: Nilearn Destrieux region name with no represenation in FreeSurfer
MEDIAL_WALL: str = "Medial_wall"


# def parse_destrieux_label(label: bytes) -> str:
#     return label.decode().replace("_and_", "&")


def plot_destrieux_hemisphere_surface(
    stats: pd.DataFrame,
    hemisphere: str = "Left",
    metric: str = "Surface Area",
    std: bool = False,
    standardize: bool = True,
    title: str = None,
    symmetric_cmap: bool = False,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    factor: int = 1,
) -> plt.Figure:
    title = f"{metric} ({hemisphere})"
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    fsaverage = datasets.fetch_surf_fsaverage()
    destrieux_labels = [
        label.decode() for label in destrieux_atlas["labels"][1:]
    ]
    metric_data = stats.xs(metric, level="Metric", axis=1)
    if std:
        data = metric_data.std().xs(hemisphere, level="Hemisphere")
        title = f"{metric} Standard Deviation ({hemisphere})"
        vmin = 0
    else:
        data = metric_data.mean().xs(hemisphere, level="Hemisphere")
        title = f"Average {title}"
    # if standardize:
    #     data[:] = StandardScaler().fit_transform(data.to_numpy().reshape((-1, 1)))
    #     title = f"Standardized {title}"
    #     symmetric_cmap = True
    #     cmap = cmap if cmap is not None else "coolwarm"
    cmap = cmap if cmap is not None else "Reds"
    destrieux_projection = destrieux_atlas[f"map_{hemisphere.lower()}"].copy()
    region_ids = sorted(set(destrieux_projection))
    for i, region_id in enumerate(region_ids):
        label = destrieux_labels[i]
        if label == MEDIAL_WALL:
            value = 0
        else:
            value = data.loc[label]
        region_mask = destrieux_projection == region_id
        destrieux_projection[region_mask] = value * factor
    surface = plotting.view_surf(
        fsaverage[f"infl_{hemisphere.lower()}"],
        destrieux_projection,
        bg_map=fsaverage[f"sulc_{hemisphere.lower()}"],
        cmap=cmap,
        title=title,
        symmetric_cmap=symmetric_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    surface.resize(900, 600)
    return surface
