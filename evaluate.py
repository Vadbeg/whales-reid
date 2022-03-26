from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything

from modules.data.dataset import ClassificationDataset
from modules.evaluation.evaluate import Evaluation
from modules.help import split_dataframe
from modules.models.efficient_net import EfficientNetModel
from modules.utils import load_yaml, reformat_checkpoint

if __name__ == '__main__':
    seed_everything(seed=27)

    DATA_CONFIG_PATH = 'configs/data_118_server.yaml'
    CHECKPOINT_PATH = (
        '/home/vadim-tsitko/Projects/SERVER/'
        'whales-reid/logs/HappyWhale/36kk3uxu/checkpoints/epoch=25-step=29509_copy.ckpt'
    )

    _data_config = load_yaml(yaml_path=DATA_CONFIG_PATH)
    _model = EfficientNetModel(model_type='tf_efficientnet_b6')
    _checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))[
        'state_dict'
    ]
    _checkpoint = reformat_checkpoint(checkpoint=_checkpoint)

    _model.load_state_dict(_checkpoint)

    _images_folder_path = Path(_data_config['train_images_folder'])
    _test_images_folder_path = Path(_data_config['test_images_folder'])
    _dataframe_path = Path(_data_config['train_meta'])

    batch_size = 16
    num_folds = 5
    num_processes = 16
    image_size = (384, 384)

    _dataframe = pd.read_csv(filepath_or_buffer=_dataframe_path)
    _train_dataframe, _test_dataframe = split_dataframe(
        dataframe=_dataframe, num_folds=num_folds
    )

    device = torch.device('cuda:2')

    _train_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_train_dataframe,
        transform_to_tensor=True,
        image_size=image_size,
        use_boxes=True,
    )
    _test_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_test_dataframe,
        transform_to_tensor=True,
        image_size=image_size,
        use_boxes=True,
    )

    evaluation = Evaluation(
        model=_model,
        train_dataset=_train_dataset,
        valid_dataset=_test_dataset,
        batch_size=batch_size,
        num_workers=num_processes,
        verbose=True,
        device=device,
    )
    _map_value = evaluation.evaluate_metric()

    print(_map_value)
