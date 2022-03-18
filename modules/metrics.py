"""Module with metrics"""

import statistics
from typing import List


def map_all_images(labels: List[str], all_predictions: List[List[str]]) -> float:
    all_map_values = []

    for curr_labels, curr_predictions in zip(labels, all_predictions):
        map_value = map_per_image(label=curr_labels, predictions=curr_predictions)
        all_map_values.append(map_value)

    return statistics.fmean(all_map_values)


def map_per_image(label: str, predictions: List[str]) -> float:
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0
