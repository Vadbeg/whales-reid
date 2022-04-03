"""Module with evaluation"""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm

from modules.data.base_dataset import BaseDataset
from modules.data.dataset import ClassificationDataset, FolderDataset
from modules.help import create_data_loader, get_gray_train_augmentations
from modules.metrics import map_all_images
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
        new_individual_threshold: float = 1.0,
        use_tta: bool = False,
    ):
        self._model = model
        self._predictor = SimpleKNNPredictor()

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._device = device
        self._verbose = verbose
        self._new_individual_threshold = new_individual_threshold
        self._use_tta = use_tta

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

        self._model.eval()
        self._model.to(device)

    def evaluate_metric(self) -> float:
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
        indexes, distances = self._predictor.predict(embeddings=valid_features)

        pred_individual_ids = self._get_individual_ids_from_indexes(
            indexes=indexes,
            distances=distances,
            train_individual_ids=train_individual_ids,
        )

        avg_map = map_all_images(
            labels=valid_individual_ids,
            all_predictions=pred_individual_ids,
        )

        return avg_map

    def evaluate_submission(self) -> List[List[str]]:
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
        if self._use_tta:
            tta_features, _ = self._prepare_features_and_individual_ids_tta(
                dataset=self._train_dataset
            )
            train_features = self._merge_features(train_features, tta_features)

        valid_features = self._prepare_features(dataset=self._valid_dataset)

        train_features = normalize(train_features, axis=1, norm='l2')
        valid_features = normalize(valid_features, axis=1, norm='l2')

        self._predictor.train(embeddings=train_features)
        indexes, distances = self._predictor.predict(
            embeddings=valid_features, n_neighbors=50
        )

        pred_individual_ids = self._get_individual_ids_from_indexes(
            indexes=indexes,
            distances=distances,
            train_individual_ids=train_individual_ids,
            top_k=5,
            new_individual_threshold=self._new_individual_threshold,
        )

        return pred_individual_ids

    @staticmethod
    def _get_individual_ids_from_indexes(
        indexes: np.ndarray,
        distances: np.ndarray,
        train_individual_ids: List[str],
        top_k: int = 5,
        new_individual_threshold: float = 1.0,
    ) -> List[List[str]]:
        pred_ids = []
        new_individual_id = 'new_individual'

        for curr_indexes, curr_distances in zip(list(indexes), list(distances)):
            curr_unique_ids: List[str] = []
            for index, distance in zip(curr_indexes, curr_distances):
                if len(curr_unique_ids) == top_k:
                    break

                if (
                    new_individual_id not in curr_unique_ids
                    and distance > new_individual_threshold
                ):
                    curr_unique_ids.append(new_individual_id)

                unique_id = train_individual_ids[index]
                if unique_id not in curr_unique_ids:
                    curr_unique_ids.append(unique_id)

            pred_ids.append(curr_unique_ids[:top_k])

        return pred_ids

    @staticmethod
    def _merge_features(*features: np.ndarray) -> np.ndarray:
        res_features = sum(features) / len(features)

        return res_features

    def _prepare_features_and_individual_ids_tta(
        self, dataset: BaseDataset
    ) -> Tuple[np.ndarray, List[str]]:
        dataset.augmentations = get_gray_train_augmentations()

        tta_features, tta_individual_ids = self._prepare_features_and_individual_ids(
            dataset=dataset
        )

        return tta_features, tta_individual_ids

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
            features = self._model(batch=image_tensors.to(self._device, torch.float))
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
            features = self._model(
                batch=curr_batch.to(self._device).to(self._device, torch.float)
            )
            feature_batches.append(features.detach().cpu().numpy())

        all_features = np.concatenate(feature_batches)

        return all_features
