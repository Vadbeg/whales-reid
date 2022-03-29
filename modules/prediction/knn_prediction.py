"""Module with KNN predictor"""

from typing import Tuple

import numpy as np
import sklearn.exceptions
from sklearn.neighbors import NearestNeighbors


class SimpleKNNPredictor:
    def __init__(self, n_neighbors: int = 10, metric: str = 'cosine'):
        self.nearest_neighbours = NearestNeighbors(
            n_neighbors=n_neighbors, metric=metric
        )

    def train(self, embeddings: np.ndarray) -> None:
        """
        Trains KNN alg

        :param embeddings: embeddings with shape (n_samples, n_features)
        """

        self.nearest_neighbours.fit(X=embeddings)

    def predict(
        self, embeddings: np.ndarray, n_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:

        try:
            distances, indexes = self.nearest_neighbours.kneighbors(
                X=embeddings, n_neighbors=n_neighbors
            )
            print(distances)
        except sklearn.exceptions.NotFittedError as error:
            raise NotTrainedPredictor(
                'This NearestNeighbors instance is not fitted yet. '
                'Call "train" with appropriate arguments before using this predictor.'
            ) from error

        return indexes, distances


class NotTrainedPredictor(Exception):
    pass
