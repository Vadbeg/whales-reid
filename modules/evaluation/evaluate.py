"""Module with evaluation"""

from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from modules.data.base_dataset import BaseDataset
from modules.data.dataset import ClassificationDataset, FolderDataset
from modules.help import create_data_loader
from modules.models.base_model import BaseModel
from modules.prediction.knn_prediction import SimpleKNNPredictor


class Evaluation:
    def __init__(
        self,
        model: BaseModel,
        train_dataset: BaseDataset,
        valid_dataset: BaseDataset,
        batch_size: int = 8,
        num_workers: int = 4,
        device: torch.device = torch.device('cpu'),
        verbose: bool = False,
    ):
        self._model = model
        self._predictor = SimpleKNNPredictor()

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._device = device
        self._verbose = verbose

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

        self._model.eval()
        self._model.to(device)

    def evaluate_metric(self):
        if not (
            isinstance(self._train_dataset, ClassificationDataset)
            and isinstance(self._valid_dataset, ClassificationDataset)
        ):
            raise ValueError(
                'Train and val datasets needs to be '
                'ClassificationDataset to calculate metrics'
            )

        (
            train_features,
            train_individual_ids,
        ) = self._prepare_features_and_individual_ids(dataset=self._train_dataset)
        (
            valid_features,
            valid_individual_ids,
        ) = self._prepare_features_and_individual_ids(dataset=self._valid_dataset)

        self._predictor.train(embeddings=train_features)
        indexes = self._predictor.predict(embeddings=valid_features)

        return indexes

    def evaluate_submission(self):
        if not (
            isinstance(self._train_dataset, ClassificationDataset)
            and isinstance(self._valid_dataset, FolderDataset)
        ):
            raise ValueError(
                'Train needs to be ClassificationDataset '
                'and val needs to be FolderDataset'
            )

        (
            train_features,
            train_individual_ids,
        ) = self._prepare_features_and_individual_ids(dataset=self._train_dataset)
        valid_features = self._prepare_features(dataset=self._valid_dataset)

        self._predictor.train(embeddings=train_features)
        indexes = self._predictor.predict(embeddings=valid_features)

        pred_individual_ids = self._get_individual_ids_from_indexes(
            indexes=indexes, train_individual_ids=train_individual_ids
        )

        return pred_individual_ids

    @staticmethod
    def _get_individual_ids_from_indexes(
        indexes: np.ndarray, train_individual_ids: List[str]
    ) -> List[List[str]]:
        pred_ids = []

        for curr_indexes in list(indexes):
            curr_ids = [train_individual_ids[index] for index in curr_indexes]
            pred_ids.append(curr_ids)

        return pred_ids

    def _prepare_features_and_individual_ids(
        self, dataset: BaseDataset
    ) -> Tuple[np.ndarray, List[str]]:
        dataloader = create_data_loader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        feature_batches = []
        all_individual_ids = []

        for curr_batch in tqdm(
            dataloader, postfix='Calculating embeddings', disable=not self._verbose
        ):
            image_tensors, individual_ids = curr_batch
            features = self._model.extract_features(
                batch=image_tensors.to(self._device)
            )
            feature_batches.append(features.detach().cpu().numpy())

            all_individual_ids.extend(individual_ids)

        all_features = np.concatenate(feature_batches)

        return all_features, all_individual_ids

    def _prepare_features(self, dataset: BaseDataset) -> np.ndarray:
        dataloader = create_data_loader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        feature_batches = []

        for curr_batch in tqdm(
            dataloader, postfix='Calculating embeddings', disable=not self._verbose
        ):
            features = self._model.extract_features(batch=curr_batch.to(self._device))
            feature_batches.append(features.detach().cpu().numpy())

        all_features = np.concatenate(feature_batches)

        return all_features
