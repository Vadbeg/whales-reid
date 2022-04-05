"""Module with classification training class"""

import random
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.utils.data import DataLoader

from modules.data.dataset import ClassificationDataset, TripletDataset
from modules.evaluation.evaluate import Evaluation
from modules.help import (
    DATAFRAME_CLASS_ID_COLUMN,
    DATAFRAME_INDIVIDUAL_ID_COLUMN,
    DATAFRAME_KFOLD_COLUMN,
    create_data_loader,
    get_train_augmentations,
    split_by_fold_from_dataframe,
    split_dataframe,
)
from modules.models.efficient_net import EfficientNetModel
from modules.training.base import BaseLightningModel


@MODEL_REGISTRY
class TripletLightningModel(BaseLightningModel):
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
        test_fold: int = 0,
        choose_fold_from_dataframe: bool = False,
        dataset_part: float = 1.0,
        use_boxes: bool = False,
        with_horizontal: bool = True,
        use_boxes_for_augmentations_chance: float = 0.0,
        in_channels: int = 1,
    ):
        self.save_hyperparameters()

        self.dataset_folder = Path(dataset_folder)
        (train_dataframe, val_dataframe,) = self._get_train_val_dataframes(
            path=Path(dataframe_path),
            num_folds=num_folds,
            test_fold=test_fold,
            choose_fold_from_dataframe=choose_fold_from_dataframe,
            dataset_part=dataset_part,
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
        self.with_horizontal = with_horizontal
        self.use_boxes_for_augmentations_chance = use_boxes_for_augmentations_chance

        self.size = size
        self.to_gray = True if in_channels == 1 else False
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        num_features = 512
        self.model = EfficientNetModel(
            model_type=model_type, num_features=num_features, in_channels=in_channels
        )
        self.loss = torch.nn.TripletMarginLoss()

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=4,
            mode='min',
            threshold=0.1,
            verbose=True,
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    def on_validation_epoch_end(self) -> None:
        if self.global_step > 0:
            self._log_evaluation_map()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model(image=image.float())

        return features

    def training_step(
        self, batch: Tuple, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, pos_image, neg_image = batch

        features = self.model(image.float())
        pos_features = self.model(pos_image.float())
        neg_features = self.model(neg_image.float())

        loss = self.loss(features, pos_features, neg_features)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return {'loss': loss}

    def validation_step(
        self, batch: Tuple, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, pos_image, neg_image = batch

        features = self.model(image.float())
        pos_features = self.model(pos_image.float())
        neg_features = self.model(neg_image.float())

        loss = self.loss(features, pos_features, neg_features)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return {'loss': loss}

    def train_dataloader(self) -> DataLoader:
        train_augmentations = get_train_augmentations()

        train_brain_dataset = TripletDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            augmentations=train_augmentations,
            use_boxes=self.use_boxes,
            use_boxes_for_augmentations_chance=self.use_boxes_for_augmentations_chance,
            to_gray=self.to_gray,
            with_horizontal=self.with_horizontal,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = TripletDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            use_boxes=self.use_boxes,
            to_gray=self.to_gray,
            with_horizontal=self.with_horizontal,
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
        path: Path,
        num_folds: int = 5,
        test_fold: int = 0,
        choose_fold_from_dataframe: bool = False,
        dataset_part: float = 1.0,
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

        if choose_fold_from_dataframe and DATAFRAME_KFOLD_COLUMN in dataframe.columns:
            train_dataframe, val_dataframe = split_by_fold_from_dataframe(
                dataframe=dataframe, test_fold=test_fold
            )
        else:
            train_dataframe, val_dataframe = split_dataframe(
                dataframe=dataframe, num_folds=num_folds, test_fold=test_fold
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

    def _log_evaluation_map(self):
        self.print('Calculating MAP')

        train_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.train_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            augmentations=None,
            use_boxes=self.use_boxes,
            use_boxes_for_augmentations_chance=self.use_boxes_for_augmentations_chance,
            to_gray=self.to_gray,
            with_horizontal=self.with_horizontal,
        )
        val_dataset = ClassificationDataset(
            folder=self.dataset_folder,
            dataframe=self.val_dataframe,
            image_size=self.size,
            transform_to_tensor=True,
            transform_label=True,
            use_boxes=self.use_boxes,
            to_gray=self.to_gray,
            with_horizontal=self.with_horizontal,
        )

        evaluation = Evaluation(
            model=self.model,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            verbose=True,
            device=self.device,
        )
        _map_value = evaluation.evaluate_metric()
        self.model.train()

        del train_dataset
        del val_dataset
        del evaluation

        self.log(
            name='val_map',
            value=_map_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.print(f'Finished calculating MAP: {_map_value}')
