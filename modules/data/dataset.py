"""Module with classification dataset"""

import random
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
    DATAFRAME_IS_HORIZONTAL_COLUMN,
    horizontal_flip,
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
        to_gray: bool = False,
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
        self.to_gray = to_gray

        if limit:
            self.image_paths = self.image_paths[:limit]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, np.ndarray]:
        image_path = self.image_paths[idx]

        image = self._load_image(image_path=image_path)
        if self.to_gray:
            image = self._transform_to_gray(image=image)

        if self.use_boxes and self.dataframe is not None:
            dataframe_row = self.dataframe.loc[
                self.dataframe[DATAFRAME_IMAGE_FILENAME_COLUMN] == image_path.name
            ].squeeze()
            coords = self._get_coords_from_row(dataframe_row=dataframe_row)
            if coords:
                image = self._get_crop_from_image(
                    image=image, coords=coords, border=0.15
                )

        image = resize_image(image, size=self.image_size)

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        if self.to_gray:
            image = image / 255.0
        else:
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
        with_horizontal: bool = False,
        use_boxes_for_augmentations_chance: float = 0.0,
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
        self.with_horizontal = with_horizontal
        self.use_boxes_for_augmentations_chance = use_boxes_for_augmentations_chance
        self.to_gray = to_gray

        if (
            with_horizontal
            and DATAFRAME_IS_HORIZONTAL_COLUMN not in self.dataframe.columns
        ):
            raise ValueError('No horizontal labels in dataset')

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[str, int]]:
        dataframe_row = self.dataframe.iloc[idx]

        image_filename = dataframe_row[DATAFRAME_IMAGE_FILENAME_COLUMN]

        image_path = self.folder.joinpath(image_filename)

        image = self._load_image(image_path=image_path)

        if self.to_gray:
            image = self._transform_to_gray(image=image)
        if self.with_horizontal and dataframe_row[DATAFRAME_IS_HORIZONTAL_COLUMN]:
            image = horizontal_flip(image=image)

        if self.use_boxes or self.use_boxes_for_augmentations_chance > random.uniform(
            0, 1
        ):
            coords = self._get_coords_from_row(dataframe_row=dataframe_row)
            if coords:
                image = self._get_crop_from_image(
                    image=image, coords=coords, border=0.1
                )

        image = resize_image(image, size=self.image_size)

        if self.augmentations:
            image = np.uint8(image)
            image = self.augmentations(image=image)['image']

        if self.to_gray:
            image = image / 255.0
        else:
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


class TripletDataset(BaseDataset):
    def __init__(
        self,
        folder: Path,
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int] = (250, 250),
        transform_to_tensor: bool = True,
        transform_label: bool = False,
        augmentations: Optional[albu.Compose] = None,
        use_boxes: bool = False,
        with_horizontal: bool = False,
        use_boxes_for_augmentations_chance: float = 0.0,
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
        self.with_horizontal = with_horizontal
        self.use_boxes_for_augmentations_chance = use_boxes_for_augmentations_chance
        self.to_gray = to_gray

        if (
            with_horizontal
            and DATAFRAME_IS_HORIZONTAL_COLUMN not in self.dataframe.columns
        ):
            raise ValueError('No horizontal labels in dataset')

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        dataframe_row = self.dataframe.iloc[idx]

        same_whales_dataframe = self._get_same_whale_dataframe(
            dataframe_row=dataframe_row
        )
        different_whales_dataframe = self._get_different_whale_dataframe(
            dataframe_row=dataframe_row
        )

        image_path = self._get_image_path(dataframe_row=dataframe_row)

        image = self._load_image(image_path=image_path)

        if len(same_whales_dataframe) > 0:
            same_whale_row = same_whales_dataframe.sample(1).squeeze()
            pos_image_path = self._get_image_path(dataframe_row=same_whale_row)

            pos_image = self._load_image(image_path=pos_image_path)
            pos_image = self._process_image(
                image=pos_image, dataframe_row=same_whale_row
            )
        else:
            pos_image = np.copy(image)
            pos_image = self._process_image(
                image=pos_image, dataframe_row=dataframe_row
            )

        image = self._process_image(image=image, dataframe_row=dataframe_row)

        diff_whale_row = different_whales_dataframe.sample(1).squeeze()
        neg_image_path = self._get_image_path(dataframe_row=diff_whale_row)

        neg_image = self._load_image(image_path=neg_image_path)
        neg_image = self._process_image(image=neg_image, dataframe_row=diff_whale_row)

        return image, pos_image, neg_image

    def _get_image_path(self, dataframe_row: pd.Series) -> Path:
        image_filename = dataframe_row[DATAFRAME_IMAGE_FILENAME_COLUMN]
        image_path = self.folder.joinpath(image_filename)

        return image_path

    def _process_image(
        self, image: np.ndarray, dataframe_row: pd.Series
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.to_gray:
            image = self._transform_to_gray(image=image)
        if self.with_horizontal and dataframe_row[DATAFRAME_IS_HORIZONTAL_COLUMN]:
            image = horizontal_flip(image=image)

        if self.use_boxes or self.use_boxes_for_augmentations_chance > random.uniform(
            0, 1
        ):
            coords = self._get_coords_from_row(dataframe_row=dataframe_row)
            if coords:
                image = self._get_crop_from_image(
                    image=image, coords=coords, border=0.1
                )

        image = resize_image(image, size=self.image_size)

        if self.augmentations:
            image = np.uint8(image)
            image = self.augmentations(image=image)['image']

        if self.to_gray:
            image = image / 255.0
        else:
            image = normalize_imagenet(image=image)

        if self.transform_to_tensor:
            image = to_tensor(image=image)

        return image

    def _get_same_whale_dataframe(self, dataframe_row: pd.Series) -> pd.DataFrame:
        same_whales_dataframe = self.dataframe.loc[
            (
                self.dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN]
                == dataframe_row[DATAFRAME_INDIVIDUAL_ID_COLUMN]
            )
            & (self.dataframe.index != dataframe_row.name)
        ]

        return same_whales_dataframe

    def _get_different_whale_dataframe(self, dataframe_row: pd.Series) -> pd.DataFrame:
        different_whales_dataframe = self.dataframe.loc[
            (
                self.dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN]
                != dataframe_row[DATAFRAME_INDIVIDUAL_ID_COLUMN]
            )
            & (self.dataframe.index != dataframe_row.name)
        ]

        return different_whales_dataframe

    def __len__(self) -> int:
        return len(self.dataframe)
