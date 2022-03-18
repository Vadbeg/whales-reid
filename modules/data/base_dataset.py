"""Module with base dataset"""


import abc
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as albu
import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import Dataset


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
        Union[Tuple[torch.Tensor, str], Tuple[np.ndarray, str]],
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
