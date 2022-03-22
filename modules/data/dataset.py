"""Module with classification dataset"""


from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torch

from modules.data.base_dataset import BaseDataset
from modules.help import (
    DATAFRAME_CLASS_ID_COLUMN,
    DATAFRAME_IMAGE_FILENAME_COLUMN,
    DATAFRAME_INDIVIDUAL_ID_COLUMN,
    normalize_imagenet,
    resize_image,
    to_tensor,
)


class FolderDataset(BaseDataset):
    def __init__(
        self,
        folder: Path,
        image_size: Tuple[int, int] = (250, 250),
        transform_to_tensor: bool = True,
        augmentations: Optional[albu.Compose] = None,
        limit: Optional[int] = None,
    ):
        super().__init__(
            folder=folder,
            image_size=image_size,
            transform_to_tensor=transform_to_tensor,
            augmentations=augmentations,
        )
        self.image_paths = list(folder.glob(pattern='**/*.*'))

        if limit:
            self.image_paths = self.image_paths[:limit]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, np.ndarray]:
        image_path = self.image_paths[idx]

        image = self._load_image(image_path=image_path)
        image = resize_image(image, size=self.image_size)

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        image = normalize_imagenet(image=image)
        if self.transform_to_tensor:
            image = to_tensor(image=image)

        return image

    def __len__(self) -> int:
        return len(self.image_paths)


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        folder: Path,
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int] = (250, 250),
        transform_to_tensor: bool = True,
        transform_label: bool = False,
        augmentations: Optional[albu.Compose] = None,
    ):
        super().__init__(
            folder=folder,
            image_size=image_size,
            transform_to_tensor=transform_to_tensor,
            augmentations=augmentations,
        )

        self.dataframe = dataframe
        self.transform_label = transform_label

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[str, int]]:
        dataframe_row = self.dataframe.iloc[idx]

        image_filename = dataframe_row[DATAFRAME_IMAGE_FILENAME_COLUMN]

        image_path = self.folder.joinpath(image_filename)

        image = self._load_image(image_path=image_path)
        image = resize_image(image, size=self.image_size)

        if self.augmentations:
            image = np.uint8(image)
            image = self.augmentations(image=image)['image']
        image = normalize_imagenet(image=image)

        if self.transform_to_tensor:
            image = to_tensor(image=image)

        if self.transform_label:
            label: Union[int, str] = dataframe_row[DATAFRAME_CLASS_ID_COLUMN]
        else:
            label = dataframe_row[DATAFRAME_INDIVIDUAL_ID_COLUMN]

        return image, label

    def __len__(self) -> int:
        return len(self.dataframe)
