"""Module with classification training class"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.utils.data import DataLoader

from modules.data.dataset import ClassificationDataset
from modules.help import (
    DATAFRAME_CLASS_ID_COLUMN,
    DATAFRAME_INDIVIDUAL_ID_COLUMN,
    create_data_loader,
    get_train_augmentations,
    split_dataframe,
)
from modules.models.efficient_net import EfficientNetModel
from modules.models.margins import ArcMarginProduct
from modules.training.base import BaseLightningModel


@MODEL_REGISTRY
class ClassificationLightningModel(BaseLightningModel):
    def __init__(
        self,
        dataset_folder: Union[str, Path],
        dataframe_path: Union[str, Path],
        model_type: str = 'efficientnet-b0',
        shuffle: bool = True,
        size: Tuple[int, int] = (250, 250),
        batch_size: int = 2,
        num_processes: int = 1,
        learning_rate: float = 0.001,
        classes: int = 2,
        test_size: float = 0.22,
    ):
        super().__init__(learning_rate=learning_rate, classes=classes)
        self.save_hyperparameters()

        self.dataset_folder = Path(dataset_folder)
        (train_dataframe, val_dataframe,) = self._get_train_val_dataframes(
            path=Path(dataframe_path),
            test_size=test_size,
        )

        self.train_dataframe = train_dataframe
        self.val_dataframe = val_dataframe

        self.shuffle = shuffle

        self.size = size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        self.model = EfficientNetModel(
            model_type=model_type,
        )
        self.margin = ArcMarginProduct(in_features=1280, out_features=classes)

        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(
        self, batch: Dict, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, label = batch

        features = self.model.extract_features(batch=image.float())
        result = self.margin(features, label, self.device)

        loss = self.loss(result, label)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='train')

        return {'loss': loss, 'pred': result, 'label': label}

    def validation_step(
        self, batch: Dict, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, label = batch

        features = self.model.extract_features(batch=image.float())
        result = self.margin(features, label, self.device)

        loss = self.loss(result, label)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='val')

        return {'loss': loss, 'pred': result, 'label': label}

    def train_dataloader(self) -> DataLoader:
        train_augmentations = get_train_augmentations()
        train_brain_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            augmentations=train_augmentations,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
        )

        val_brain_dataloader = create_data_loader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_processes,
        )

        return val_brain_dataloader

    @staticmethod
    def _get_train_val_dataframes(
        path: Path, test_size: float = 0.22
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataframe = pd.read_csv(filepath_or_buffer=path)
        dataframe[DATAFRAME_CLASS_ID_COLUMN] = pd.factorize(
            values=dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN]
        )[0]

        train_dataframe, val_dataframe = split_dataframe(
            dataframe=dataframe, test_size=test_size
        )

        return train_dataframe, val_dataframe
