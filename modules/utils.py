"""Module with utils for whole project"""


from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import torch
import yaml


def load_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)

    with yaml_path.open(mode='r') as file:
        result = yaml.safe_load(file)

    return result


def prepare_submission_file(
    individual_ids: List[List[str]], image_paths: List[Path]
) -> pd.DataFrame:
    assert len(individual_ids) == len(image_paths), 'Must to be the same length'

    individual_ids_string = [' '.join(curr_ids) for curr_ids in individual_ids]
    image_filenames = [curr_image_path.name for curr_image_path in image_paths]

    dataframe = pd.DataFrame(
        columns=['image', 'predictions'],
        data={'image': image_filenames, 'predictions': individual_ids_string},
    )

    return dataframe


def reformat_checkpoint(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_checkpoint = OrderedDict()

    for key, item in checkpoint.items():
        if key.startswith('model.'):
            new_checkpoint[key[6:]] = item

    return new_checkpoint


def add_prefix_checkpoint_name(
    checkpoint: Dict[str, torch.Tensor], prefix: str = '_eff_net_model.'
) -> Dict[str, torch.Tensor]:
    new_checkpoint = OrderedDict()

    for key, item in checkpoint.items():
        new_checkpoint[prefix + key] = item

    return new_checkpoint
