from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

# Data extraction pattern.
DATA_DIR = Path(__file__).parent / "data"
FILE_PATTERN = DATA_DIR / "*.nii.gz"

# Contextual info CSV.
INFO_CSV = DATA_DIR / "info.csv"
INFO = pd.read_csv(INFO_CSV, index_col=0)
TARGETS = {"M": 0, "F": 1}

# Set tensorflow seed for reproducibility.
tf.random.set_seed(42)


def get_label(path: Path) -> str:
    scan_id = int(path.name.split(".")[0])
    label = TARGETS[INFO.loc[scan_id, "Sex"]]
    return tf.convert_to_tensor([label], dtype="uint8")


def read_volume(path: Path) -> np.ndarray:
    volume = np.expand_dims(nib.load(path).get_fdata(), axis=0)
    return tf.convert_to_tensor(volume, dtype="float32")


def process_volume(path: tf.Tensor) -> Tuple[np.ndarray, str]:
    path = Path(path.numpy().decode())
    volume = read_volume(path)
    label = get_label(path)
    return volume, label


def tf_process_volume(path: tf.Tensor):
    volume, label = tf.py_function(
        func=process_volume, inp=[path], Tout=[tf.float32, tf.uint8]
    )
    volume.set_shape((1, 91, 109, 91))
    label.set_shape((1,))
    return volume, label


def read_dataset():
    ds = tf.data.Dataset.list_files(str(FILE_PATTERN), shuffle=True)
    validation_size = int(0.2 * len(ds))
    train = ds.skip(validation_size).map(
        tf_process_volume, num_parallel_calls=tf.data.AUTOTUNE
    )
    test = ds.take(validation_size).map(
        tf_process_volume, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train, test
