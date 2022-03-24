"""Module with base dataset"""


import abc
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from modules.help import DATAFRAME_BBOX_COLUMN


class BaseDataset(abc.ABC, Dataset):
    def __init__(
        self,
        folder: Path,
        image_size: Tuple[int, int] = (250, 250),
        transform_to_tensor: bool = True,
        augmentations: Optional[albu.Compose] = None,
    ):
        self.folder = folder

        self.image_size = image_size
        self.transform_to_tensor = transform_to_tensor
        self.augmentations = augmentations

    @abc.abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Union[torch.Tensor, np.ndarray], Union[str, int]],
        Union[torch.Tensor, np.ndarray],
    ]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def _load_image(
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def _get_coords_from_row(
        dataframe_row: pd.Series,
    ) -> Optional[Tuple[int, int, int, int]]:
        value_raw = dataframe_row[DATAFRAME_BBOX_COLUMN]
        if not isinstance(value_raw, str) and math.isnan(value_raw):
            return None

        coords = eval(value_raw)[0]

        return (
            coords[0],
            coords[1],
            coords[2],
            coords[3],
        )

    @staticmethod
    def _get_crop_from_image(
        image: np.ndarray, coords: Tuple[int, int, int, int]
    ) -> np.ndarray:
        image_crop = image[
            coords[1] : coords[3],
            coords[0] : coords[2],
        ]

        return image_crop
