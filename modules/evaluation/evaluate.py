"""Module with evaluation"""

from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from modules.data.dataset import ClassificationDataset
from modules.help import create_data_loader
from modules.models.base_model import BaseModel
from modules.prediction.knn_prediction import SimpleKNNPredictor


class Evaluation:
    def __init__(
        self,
        model: BaseModel,
        train_dataset: ClassificationDataset,
        valid_dataset: ClassificationDataset,
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

    def evaluate(self):
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

    def _prepare_features_and_individual_ids(
        self, dataset: ClassificationDataset
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
