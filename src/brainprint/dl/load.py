import ast
import random
from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

# Data extraction pattern.
DATA_DIR = Path(__file__).parent / "data"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
QUESTIONNAIRE_DATA_DIR = DATA_DIR / "questionnaire"
QUESTIONNAIRE_TRAIN_DIR = QUESTIONNAIRE_DATA_DIR / "train"
QUESTIONNAIRE_VALIDATION_DIR = QUESTIONNAIRE_DATA_DIR / "validation"
QUESTIONNAIRE_TEST_DIR = QUESTIONNAIRE_DATA_DIR / "test"
NII_PATTERN = "*.nii.gz"

# Contextual info CSV.
INFO_CSV = DATA_DIR / "info.csv"
INFO = pd.read_csv(INFO_CSV, index_col=0)
INFO["Scan Date"] = pd.to_datetime(INFO["Scan Date"])
INFO["Date of Birth"] = pd.to_datetime(INFO["Date of Birth"])
INFO["Age"] = (
    (INFO["Scan Date"] - INFO["Date of Birth"]) / np.timedelta64(1, "Y")
).astype("float16")
SEX_TARGETS = {"M": 0, "F": 1}

# Questionnaire CSV.
QUESTIONNAIRE_CSV = DATA_DIR / "questionnaire.csv"
QUESTIONNAIRE = pd.read_csv(QUESTIONNAIRE_CSV)
PERSONALITY_TRAITS = QUESTIONNAIRE.iloc[:, -7:-2]

# Shape information.
VOLUME_SHAPE = (91, 109, 91)
INFO_SHAPE = (7,)
N_CLASSES = 2
N_PERSONALITY_TRAITS = 5

# Set tensorflow seed for reproducibility.
tf.random.set_seed(42)


def get_nii_paths(
    source: Union[bytes, str, Path] = TRAIN_DIR,
    shuffle: bool = True,
    seed: int = 0,
) -> List[Path]:
    source = source.decode() if isinstance(source, bytes) else source
    source = Path(source)
    paths = list(source.glob(NII_PATTERN))
    if shuffle:
        random.Random(seed).shuffle(paths)
    return paths


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
    assert not np.any(np.isnan(info))
    if not as_tensor:
        return info
    return tf.convert_to_tensor(info, dtype="float32")


def get_sex_label(
    path: Path, as_tensor: bool = True, one_hot_encode: bool = True
) -> str:
    scan_id = int(path.name.split(".")[0])
    label = [SEX_TARGETS[INFO.loc[scan_id, "Sex"]]]
    assert not np.any(np.isnan(label))
    if not as_tensor:
        return label
    if one_hot_encode:
        return tf.one_hot(label, depth=2)
    return tf.convert_to_tensor(label, dtype="uint8")


def get_personality_scores(path: Path, as_tensor: bool = True) -> str:
    scan_id = int(path.name.split(".")[0])
    subject_id = INFO.loc[scan_id, "Subject ID"]
    scores = PERSONALITY_TRAITS[
        QUESTIONNAIRE["Subject ID"] == subject_id
    ].values.squeeze()
    assert not np.any(np.isnan(scores))
    if not as_tensor:
        return scores
    return tf.convert_to_tensor(scores, dtype="float32")


def read_volume(path: Path, as_tensor: bool = True) -> np.ndarray:
    volume = nib.load(path).get_fdata()
    assert not np.any(np.isnan(volume))
    volume /= volume.max()
    if not as_tensor:
        return volume
    volume = tf.convert_to_tensor(volume, dtype="float32")
    return volume


def get_age_label(path: Path, as_tensor: bool = True) -> str:
    scan_id = int(path.name.split(".")[0])
    label = [INFO.loc[scan_id, "Age"]]
    assert not np.any(np.isnan(label))
    if not as_tensor:
        return label
    label = tf.convert_to_tensor(label, dtype="float16")
    return label


def generate_dataset(
    source: Path = TRAIN_DIR,
    as_tensor: bool = True,
    include_info: bool = True,
    one_hot_encode: bool = False,
    target: str = "sex",
):
    nii_paths = get_nii_paths(source)
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
        elif target == "personality":
            label = get_personality_scores(path, as_tensor=as_tensor)
        else:
            raise ValueError("Invalid target.")
        if include_info:
            info = get_scan_info(path, as_tensor=as_tensor)
            yield (volume, info), label
        else:
            yield volume, label


def read_dataset(
    include_info: bool = True,
    one_hot_encode: bool = False,
    batch_size: int = 8,
    target: str = "sex",
):
    output_shape = (1,)
    if target == "sex":
        if one_hot_encode:
            output_shape = (1, N_CLASSES)
        output_dtype = tf.uint8
    elif target == "age":
        output_dtype = tf.float16
    elif target == "personality":
        output_shape = (N_PERSONALITY_TRAITS,)
        output_dtype = tf.float32
    output_spec = tf.TensorSpec(shape=output_shape, dtype=output_dtype)
    volume_spec = tf.TensorSpec(shape=VOLUME_SHAPE, dtype=tf.float32)
    scan_info_spec = tf.TensorSpec(shape=INFO_SHAPE, dtype=tf.float32)
    output_signature = ((volume_spec, scan_info_spec), output_spec)

    train_dir = str(
        QUESTIONNAIRE_TRAIN_DIR if target == "personality" else TRAIN_DIR
    )
    validation_dir = str(
        QUESTIONNAIRE_VALIDATION_DIR
        if target == "personality"
        else VALIDATION_DIR
    )
    train = tf.data.Dataset.from_generator(
        generate_dataset,
        output_signature=output_signature,
        args=(train_dir, True, include_info, one_hot_encode, target),
    )
    validation = tf.data.Dataset.from_generator(
        generate_dataset,
        output_signature=output_signature,
        args=(validation_dir, True, include_info, one_hot_encode, target),
    )
    train = (
        train.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation = (
        validation.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    )
    if target != "personality":
        return train, validation
    test = tf.data.Dataset.from_generator(
        generate_dataset,
        output_signature=output_signature,
        args=(
            str(QUESTIONNAIRE_TEST_DIR),
            True,
            include_info,
            one_hot_encode,
            target,
        ),
    )
    test = test.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return train, validation, test
