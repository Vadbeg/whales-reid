"""Module with classification training class"""

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.utils.data import DataLoader

from modules.data.dataset import ClassificationDataset
from modules.help import (
    DATAFRAME_CLASS_ID_COLUMN,
    DATAFRAME_INDIVIDUAL_ID_COLUMN,
    create_data_loader,
    get_gray_train_augmentations,
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
        model_type: str = 'efficientnet_b0',
        shuffle: bool = True,
        size: Tuple[int, int] = (250, 250),
        batch_size: int = 2,
        num_processes: int = 1,
        learning_rate: float = 0.001,
        classes: int = 2,
        num_folds: int = 4,
        dataset_part: float = 1.0,
        use_boxes: bool = False,
        label_smoothing: float = 0.1,
        in_channels: int = 1,
    ):
        self.save_hyperparameters()

        self.dataset_folder = Path(dataset_folder)
        (train_dataframe, val_dataframe,) = self._get_train_val_dataframes(
            path=Path(dataframe_path), num_folds=num_folds, dataset_part=dataset_part
        )
        if dataset_part < 1.0:
            classes = self._get_num_classes(
                train_dataframe=train_dataframe, val_dataframe=val_dataframe
            )

        super().__init__(learning_rate=learning_rate, classes=classes)

        self.train_dataframe = train_dataframe
        self.val_dataframe = val_dataframe

        self.shuffle = shuffle
        self.use_boxes = use_boxes

        self.size = size
        self.to_gray = True if in_channels == 1 else False
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        num_features = 1000
        self.model = EfficientNetModel(
            model_type=model_type, num_features=num_features, in_channels=in_channels
        )
        self.margin = ArcMarginProduct(in_features=num_features, out_features=classes)

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._num_of_train_batches = 100

    def configure_callbacks(self) -> List[pl.callbacks.Callback]:
        return [
            pl.callbacks.EarlyStopping(
                monitor='val_loss', patience=30, min_delta=0.001, verbose=True
            ),
            pl.callbacks.LearningRateMonitor(),
        ]

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.learning_rate,
            total_steps=self._num_of_train_batches,
            epochs=self.trainer.max_epochs,
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    def training_step(
        self, batch: Tuple, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, label = batch

        features = self.model(image.float())
        result = self.margin(features, label, self.device)

        loss = self.loss(result, label)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        self._log_metrics(preds=torch.sigmoid(result), target=label, prefix='train')

        return {'loss': loss, 'pred': result, 'label': label}

    def validation_step(
        self, batch: Tuple, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        self.trainer.callbacks[3].patience = 30

        image, label = batch

        features = self.model(image.float())
        result = self.margin(features, label, self.device)

        loss = self.loss(result, label)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        self._log_metrics(preds=torch.sigmoid(result), target=label, prefix='val')

        return {'loss': loss, 'pred': result, 'label': label}

    def train_dataloader(self) -> DataLoader:
        train_augmentations = get_gray_train_augmentations()
        train_brain_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            augmentations=train_augmentations,
            use_boxes=self.use_boxes,
            to_gray=self.to_gray,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )
        self._num_of_train_batches = len(train_brain_dataloader)

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            use_boxes=self.use_boxes,
            to_gray=self.to_gray,
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
        path: Path, num_folds: int = 5, dataset_part: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataframe = pd.read_csv(filepath_or_buffer=path)

        if dataset_part < 1.0:
            individual_ids = dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN].tolist()
            individual_ids_index = int(len(set(individual_ids)) * dataset_part)

            random.shuffle(individual_ids)
            individual_ids_part = individual_ids[:individual_ids_index]

            dataframe = dataframe.loc[
                dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN].isin(individual_ids_part)
            ]

        dataframe[DATAFRAME_CLASS_ID_COLUMN] = pd.factorize(
            values=dataframe[DATAFRAME_INDIVIDUAL_ID_COLUMN]
        )[0]

        train_dataframe, val_dataframe = split_dataframe(
            dataframe=dataframe, num_folds=num_folds
        )

        return train_dataframe, val_dataframe

    @staticmethod
    def _get_num_classes(
        train_dataframe: pd.DataFrame, val_dataframe: pd.DataFrame
    ) -> int:
        train_class_ids = train_dataframe[DATAFRAME_CLASS_ID_COLUMN].tolist()
        val_class_ids = val_dataframe[DATAFRAME_CLASS_ID_COLUMN].tolist()

        class_ids = set(train_class_ids + val_class_ids)
        num_classes = len(class_ids)

        return num_classes
