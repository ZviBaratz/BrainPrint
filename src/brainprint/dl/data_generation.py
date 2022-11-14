import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from tensorflow.keras.utils import Sequence

MNI_TEMPLATE: nib.Nifti1Image = load_mni152_template()


class AnatomicalScanGenerator(Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size=4,
        template: nib.Nifti1Image = MNI_TEMPLATE,
    ):
        "Initialization"
        self.batch_size = batch_size
        self.df = df
        self.template = template

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, index) -> np.ndarray:
        "Generate one batch of data."
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.paths))
        indices = self.indexes[start_index:end_index]
        return self.__data_generation(indices)

    def generate_epoch_data(self, indices) -> np.ndarray:
        "Generates data containing batch_size samples."
        X = np.empty((self.batch_size, *self.template.shape, 1))
        for i, index in enumerate(indices):
            volume_path = self.df.loc[index, "Path"]
            X[i, :, :, :, :] = self.get_image_data(volume_path)
        sex = self.df.loc[indices, "Sex"].values
        age = self.df.loc[indices, "Age"].values
        return [X, sex], [age]

    def get_image_data(
        self, path: Path, template: nib.Nifti1Image = MNI_TEMPLATE
    ) -> np.ndarray:
        image = nib.load(path)
        data = resample_to_img(image, template).get_fdata()
        return data.expand_dims(axis=-1)
