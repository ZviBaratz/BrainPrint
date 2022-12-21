import ast
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

# Data extraction pattern.
DATA_DIR = Path(__file__).parent / "data"
NII_PATTERN = "*.nii.gz"
FILE_PATTERN = DATA_DIR / NII_PATTERN

# Contextual info CSV.
INFO_CSV = DATA_DIR / "info.csv"
INFO = pd.read_csv(INFO_CSV, index_col=0)
INFO["Scan Date"] = pd.to_datetime(INFO["Scan Date"])
INFO["Date of Birth"] = pd.to_datetime(INFO["Date of Birth"])
INFO["Age"] = (
    (INFO["Scan Date"] - INFO["Date of Birth"]) / np.timedelta64(1, "Y")
).astype("float16")
TARGETS = {"M": 0, "F": 1}

# Shape information.
VOLUME_SHAPE = (91, 109, 91)
INFO_SHAPE = (7,)
N_CLASSES = 2

# Set tensorflow seed for reproducibility.
tf.random.set_seed(42)


def get_nii_paths(source: Path = DATA_DIR) -> List[Path]:
    return list(source.glob(NII_PATTERN))


def get_scan_info(path: Path, as_tensor: bool = True):
    scan_id = int(path.name.split(".")[0])
    tr = INFO.loc[scan_id, "TR"] / 5000
    te = INFO.loc[scan_id, "TE"] / 4
    ti = INFO.loc[scan_id, "TI"] / 3000
    flip_angle = INFO.loc[scan_id, "Flip Angle"] / 10
    spatial_resolution = (
        np.array(ast.literal_eval(INFO.loc[scan_id, "Spatial Resolution"])) / 4
    )
    x_res, y_res, z_res = spatial_resolution
    info = [tr, te, ti, flip_angle, x_res, y_res, z_res]
    if not as_tensor:
        return info
    info = tf.convert_to_tensor(info, dtype="float32")
    return info


def get_sex_label(
    path: Path, as_tensor: bool = True, one_hot_encode: bool = True
) -> str:
    scan_id = int(path.name.split(".")[0])
    label = [TARGETS[INFO.loc[scan_id, "Sex"]]]
    if not as_tensor:
        return label
    if one_hot_encode:
        return tf.one_hot(label, depth=2)
    label = tf.convert_to_tensor(label, dtype="uint8")
    return label


def read_volume(path: Path, as_tensor: bool = True) -> np.ndarray:
    volume = nib.load(path).get_fdata()
    volume /= volume.max()
    if not as_tensor:
        return volume
    volume = tf.convert_to_tensor(volume, dtype="float32")
    return volume


def get_age_label(path: Path, as_tensor: bool = True) -> str:
    scan_id = int(path.name.split(".")[0])
    label = [INFO.loc[scan_id, "Age"]]
    if not as_tensor:
        return label
    label = tf.convert_to_tensor(label, dtype="float16")
    return label


def generate_dataset(
    as_tensor: bool = True,
    include_info: bool = True,
    one_hot_encode: bool = False,
    target: str = "sex",
):
    nii_paths = get_nii_paths()
    if isinstance(target, bytes):
        target = target.decode("utf-8")
    for path in nii_paths:
        volume = read_volume(path, as_tensor=as_tensor)
        if target == "sex":
            label = get_sex_label(
                path, as_tensor=as_tensor, one_hot_encode=one_hot_encode
            )
        elif target == "age":
            label = get_age_label(path, as_tensor=as_tensor)
        else:
            raise ValueError("Invalid target.")
        if include_info:
            info = get_scan_info(path, as_tensor=as_tensor)
            yield (volume, info), label
        else:
            yield volume, label


def read_dataset(
    validation_size: float = 0.2,
    include_info: bool = True,
    one_hot_encode: bool = False,
    batch_size: int = 16,
    target: str = "sex",
):
    output_shape = (1, N_CLASSES) if one_hot_encode else (1,)
    output_spec = (
        tf.TensorSpec(shape=output_shape, dtype=tf.uint8)
        if target == "sex"
        else tf.TensorSpec(shape=(1,), dtype=tf.float16)
    )
    output_signature = (
        (
            tf.TensorSpec(shape=VOLUME_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=INFO_SHAPE, dtype=tf.float32),
        ),
        output_spec,
    )

    ds = tf.data.Dataset.from_generator(
        generate_dataset,
        output_signature=output_signature,
        args=(True, include_info, one_hot_encode, target),
    )
    nii_paths = get_nii_paths()
    validation_size = int(validation_size * len(nii_paths))
    train = (
        ds.skip(validation_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    test = (
        ds.take(validation_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    return train, test
