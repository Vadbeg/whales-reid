from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from modules.data.dataset import ClassificationDataset, FolderDataset
from modules.evaluation.evaluate import Evaluation
from modules.models.efficient_net import EfficientNetModel
from modules.utils import load_yaml


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
            new_checkpoint[key.removeprefix('model.')] = item

    return new_checkpoint


if __name__ == '__main__':
    DATA_CONFIG_PATH = 'configs/data.yaml'
    CHECKPOINT_PATH = (
        '/home/vadbeg/Projects/kaggle/happy-whale-and-dolphin/'
        'whales-reid/logs/lightning_logs/version_8/checkpoints/epoch=11-step=503.ckpt'
    )

    _data_config = load_yaml(yaml_path=DATA_CONFIG_PATH)
    _model = EfficientNetModel(model_type='efficientnet-b0')
    _checkpoint = torch.load(CHECKPOINT_PATH)['state_dict']
    _checkpoint = reformat_checkpoint(checkpoint=_checkpoint)

    _model.load_state_dict(_checkpoint)

    _images_folder_path = Path(_data_config['train_images_folder'])
    _test_images_folder_path = Path(_data_config['test_images_folder'])
    _dataframe_path = Path(_data_config['train_meta'])

    _train_dataframe = pd.read_csv(filepath_or_buffer=_dataframe_path)

    batch_size = 32
    num_processes = 10
    train_num = 100
    valid_num = 50

    device = torch.device('cuda:0')

    _train_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_train_dataframe,
        transform_to_tensor=True,
    )
    # _valid_dataset = ClassificationDataset(
    #     folder=_images_folder_path,
    #     dataframe=_train_dataframe.iloc[train_num : train_num + valid_num],
    #     transform_to_tensor=True,
    # )
    _valid_dataset = FolderDataset(
        folder=_test_images_folder_path,
        transform_to_tensor=True,
        limit=None,
    )

    evaluation = Evaluation(
        model=_model,
        train_dataset=_train_dataset,
        valid_dataset=_valid_dataset,
        batch_size=batch_size,
        num_workers=num_processes,
        verbose=True,
        device=device,
    )
    _individual_ids = evaluation.evaluate_submission()

    _pred_dataframe = prepare_submission_file(
        individual_ids=_individual_ids, image_paths=_valid_dataset.image_paths
    )

    _pred_dataframe.to_csv(path_or_buf='results/submission.csv', index=False)
