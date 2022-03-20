"""Module with help functions"""

from typing import Tuple

import albumentations as albu
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from cv2 import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATAFRAME_IMAGE_FILENAME_COLUMN = 'image'
DATAFRAME_IMAGE_SPECIES_COLUMN = 'species'
DATAFRAME_INDIVIDUAL_ID_COLUMN = 'individual_id'
DATAFRAME_CLASS_ID_COLUMN = 'class_id'


def to_tensor(image: np.ndarray) -> torch.Tensor:
    to_tensor_func = ToTensorV2(always_apply=True)
    image = to_tensor_func(image=image)['image']

    return image


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    normalize_func = albu.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    image = normalize_func(image=image)['image']

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


def get_train_augmentations() -> albu.Compose:
    augs = albu.Compose(
        [
            albu.ShiftScaleRotate(
                shift_limit=(-0.0625, 0.0625),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-20, 20),
                p=0.5,
            ),
            albu.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
            ),
            albu.Blur(p=0.2),
            albu.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.5,
            ),
            albu.JpegCompression(quality_lower=80, quality_upper=100, p=0.1),
            albu.ToGray(p=0.05),
        ]
    )

    return augs


def split_dataframe(
    dataframe: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    individual_ids = dataframe['individual_id']

    dataframe_train, dataframe_test = train_test_split(
        dataframe, test_size=test_size, stratify=individual_ids
    )

    return dataframe_train, dataframe_test
