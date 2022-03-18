"""Module with help functions"""

from typing import Tuple

import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from cv2 import cv2
from torch.utils.data import DataLoader, Dataset


def to_tensor(image: np.ndarray) -> torch.Tensor:
    to_tensor_func = ToTensorV2(always_apply=True)
    image = to_tensor_func(image=image)['image']

    return image


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = cv2.resize(image, dsize=size)

    return image


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader
