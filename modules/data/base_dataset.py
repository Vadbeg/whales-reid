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
    def _transform_to_gray(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
        image: np.ndarray, coords: Tuple[int, int, int, int], border: float = 0.0
    ) -> np.ndarray:
        """
        Cuts out box out of the image

        :param image: original image
        :param coords: box coords (x, y, w, h)
        :param border: border around box to save
        :return: result area
        """

        x_border = int((coords[3] - coords[1]) * border)
        y_border = int((coords[2] - coords[0]) * border)

        x_start = int(coords[1]) - x_border
        x_start = x_start if x_start > 0 else 0
        x_end = int(coords[3]) + x_border
        x_end = x_end if x_end < image.shape[0] else image.shape[0]

        y_start = int(coords[0]) - y_border
        y_start = y_start if y_start > 0 else 0
        y_end = int(coords[2]) + y_border
        y_end = y_end if y_end < image.shape[1] else image.shape[1]

        crop = image[
            x_start:x_end,
            y_start:y_end,
        ]

        return crop
