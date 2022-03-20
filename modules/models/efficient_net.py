"""Module with effnet models"""

import timm
import torch

from modules.models.base_model import BaseModel


class EfficientNetModel(BaseModel):
    def __init__(self, model_type: str = 'efficientnet_b0', num_classes: int = 1000):
        super().__init__()

        self._eff_net_model = timm.create_model(model_type, pretrained=True)
        self.out_features = self._eff_net_model.get_classifier().in_features

        self._eff_net_model.reset_classifier(num_classes=0, global_pool="avg")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        prediction = self._eff_net_model(batch)

        return prediction
