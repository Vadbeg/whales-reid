from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from timm.data import create_transform
from timm.optim import create_optimizer_v2

from modules.data.dataset import ClassificationDataset, FolderDataset
from modules.evaluation.evaluate import Evaluation
from modules.models.efficient_net import EfficientNetModel
from modules.utils import (
    add_prefix_checkpoint_name,
    load_yaml,
    prepare_submission_file,
    reformat_checkpoint,
)

if __name__ == '__main__':
    seed_everything(seed=27)

    DATA_CONFIG_PATH = 'configs/data_118_server.yaml'
    CHECKPOINT_PATH = (
        '/home/vadim-tsitko/Projects/SERVER/whales-reid/logs/'
        'HappyWhale/zx7e96qk/checkpoints/epoch=49-step=60049.ckpt'
    )

    _data_config = load_yaml(yaml_path=DATA_CONFIG_PATH)
    _model = EfficientNetModel(
        model_type='tf_efficientnet_b4', with_embeddings_layer=True, in_channels=3
    )
    _checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))[
        'state_dict'
    ]
    _checkpoint = reformat_checkpoint(checkpoint=_checkpoint)
    # _checkpoint = add_prefix_checkpoint_name(
    #     checkpoint=_checkpoint,
    #     prefix='_eff_net_model.'
    # )

    _model.load_state_dict(_checkpoint)

    _images_folder_path = Path(_data_config['train_images_folder'])
    _test_images_folder_path = Path(_data_config['test_images_folder'])
    _dataframe_path = Path(_data_config['train_meta'])
    _test_dataframe_path = Path(_data_config['test_meta'])

    _train_dataframe = pd.read_csv(filepath_or_buffer=_dataframe_path)
    _test_dataframe = pd.read_csv(filepath_or_buffer=_test_dataframe_path)

    batch_size = 16
    num_processes = 20
    image_size = (512, 512)

    device = torch.device('cuda:2')

    _train_dataset = ClassificationDataset(
        folder=_images_folder_path,
        dataframe=_train_dataframe,
        transform_to_tensor=True,
        image_size=image_size,
        use_boxes=False,
        to_gray=False,
    )
    _valid_dataset = FolderDataset(
        folder=_test_images_folder_path,
        dataframe=_test_dataframe,
        transform_to_tensor=True,
        limit=None,
        image_size=image_size,
        use_boxes=False,
        to_gray=False,
    )

    evaluation = Evaluation(
        model=_model,
        train_dataset=_train_dataset,
        valid_dataset=_valid_dataset,
        batch_size=batch_size,
        num_workers=num_processes,
        verbose=True,
        device=device,
        new_individual_threshold=0.5,
        use_tta=False,
    )
    _individual_ids = evaluation.evaluate_submission()

    _pred_dataframe = prepare_submission_file(
        individual_ids=_individual_ids, image_paths=_valid_dataset.image_paths
    )

    _pred_dataframe.to_csv(path_or_buf='results/submission.csv', index=False)
