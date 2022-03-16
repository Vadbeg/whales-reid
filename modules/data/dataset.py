"""Module with classification dataset"""


from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from modules.help import to_tensor


class ClassificationDataset(Dataset):
    IMAGE_FILENAME_COLUMN = 'image'
    IMAGE_SPECIES_COLUMN = 'species'
    INDIVIDUAL_ID_COLUMN = 'individual_id'

    def __init__(
        self,
        folder: Path,
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int] = (50, 50),
        transform_to_tensor: bool = True,
        augmentations: Optional[albu.Compose] = None,
    ):
        self.dataframe = dataframe
        self.folder = folder

        self.image_size = image_size
        self.transform_to_tensor = transform_to_tensor
        self.augmentations = augmentations

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, int],]:
        dataframe_row = self.dataframe.iloc[idx]

        image_filename = dataframe_row[self.IMAGE_FILENAME_COLUMN]
        individual_id = dataframe_row[self.INDIVIDUAL_ID_COLUMN]

        image_path = self.folder.joinpath(image_filename)

        image = self._load_image(image_path=image_path)

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        if self.transform_to_tensor:
            image = to_tensor(image=image) / 255

        return image

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _load_image(
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
