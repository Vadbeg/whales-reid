from pathlib import Path

import pandas as pd

from modules.data.dataset import ClassificationDataset
from modules.evaluation.evaluate import Evaluation
from modules.models.efficient_net import EfficientNetModel
from modules.utils import load_yaml

if __name__ == '__main__':
    DATA_CONFIG_PATH = 'configs/data.yaml'

    _data_config = load_yaml(yaml_path=DATA_CONFIG_PATH)
    _model = EfficientNetModel(model_type='efficientnet-b0')

    _images_folder_path = Path(_data_config['train_images_folder'])
    _dataframe_path = Path(_data_config['train_meta'])

    _train_dataframe = pd.read_csv(filepath_or_buffer=_dataframe_path)

    train_num = 100
    valid_num = 50
    _train_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_train_dataframe.iloc[:train_num],
        transform_to_tensor=True,
    )
    _valid_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_train_dataframe.iloc[train_num : train_num + valid_num],
        transform_to_tensor=True,
    )

    evaluation = Evaluation(
        model=_model, train_dataset=_train_dataset, valid_dataset=_valid_dataset
    )
    indexes = evaluation.evaluate()

    print(indexes)
