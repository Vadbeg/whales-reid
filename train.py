"""Script for model training"""

import warnings

import flash.image
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, LightningCLI

from modules.training.arc_face_train import ClassificationLightningModel
from modules.training.triplet_train import TripletLightningModel

if __name__ == '__main__':
    MODEL_REGISTRY.register_classes(flash.image, LightningModule)
    warnings.filterwarnings('ignore')
    cli = LightningCLI(save_config_callback=None)
