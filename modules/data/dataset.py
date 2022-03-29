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
        dataframe: Optional[pd.DataFrame] = None,
        image_size: Tuple[int, int] = (250, 250),
        transform_to_tensor: bool = True,
        augmentations: Optional[albu.Compose] = None,
        limit: Optional[int] = None,
        use_boxes: bool = False,
    ):
        super().__init__(
            folder=folder,
            image_size=image_size,
            transform_to_tensor=transform_to_tensor,
            augmentations=augmentations,
        )
        self.image_paths = list(folder.glob(pattern='**/*.*'))

        self.dataframe = dataframe
        self.use_boxes = use_boxes
        if limit:
            self.image_paths = self.image_paths[:limit]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, np.ndarray]:
        image_path = self.image_paths[idx]

        image = self._load_image(image_path=image_path)

        if self.use_boxes and self.dataframe is not None:
            dataframe_row = self.dataframe.loc[
                self.dataframe[DATAFRAME_IMAGE_FILENAME_COLUMN] == image_path.name
            ].squeeze()
            coords = self._get_coords_from_row(dataframe_row=dataframe_row)
            if coords:
                image = self._get_crop_from_image(image=image, coords=coords)

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
        use_boxes: bool = False,
        to_gray: bool = False,
    ):
        super().__init__(
            folder=folder,
            image_size=image_size,
            transform_to_tensor=transform_to_tensor,
            augmentations=augmentations,
        )

        self.dataframe = dataframe
        self.transform_label = transform_label
        self.use_boxes = use_boxes
        self.to_gray = to_gray

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[str, int]]:
        dataframe_row = self.dataframe.iloc[idx]

        image_filename = dataframe_row[DATAFRAME_IMAGE_FILENAME_COLUMN]

        image_path = self.folder.joinpath(image_filename)

        image = self._load_image(image_path=image_path)

        if self.to_gray:
            image = self._transform_to_gray(image=image)

        if self.use_boxes:
            coords = self._get_coords_from_row(dataframe_row=dataframe_row)
            if coords:
                image = self._get_crop_from_image(image=image, coords=coords)

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
