from pathlib import Path

import pandas as pd
from tqdm import tqdm

from modules.data.dataset import ClassificationDataset
from modules.utils import load_yaml

if __name__ == '__main__':
    DATA_CONFIG_PATH = 'configs/data.yaml'

    data_config = load_yaml(yaml_path=DATA_CONFIG_PATH)

    images_folder_path = Path(data_config['train_images_folder'])
    dataframe_path = Path(data_config['train_meta'])

    train_dataframe = pd.read_csv(filepath_or_buffer=dataframe_path)

    dataset = ClassificationDataset(
        folder=images_folder_path, dataframe=train_dataframe, transform_to_tensor=False
    )

    for curr_item in tqdm(dataset):
        pass
