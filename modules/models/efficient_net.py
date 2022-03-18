"""Module with effnet models"""

from typing import Tuple

import torch
from efficientnet_pytorch import EfficientNet

from modules.models.base_model import BaseModel


class EfficientNetModel(BaseModel):
    def __init__(self, model_type: str = 'efficientnet-b0', num_classes: int = 1000):
        super().__init__()

        self._eff_net_model = EfficientNet.from_pretrained(
            model_name=model_type, num_classes=num_classes
        )

    def forward(self, batch: Tuple[torch.Tensor, str]) -> torch.Tensor:
        prediction = self._eff_net_model(batch)

        return prediction

    def extract_features(self, batch: Tuple[torch.Tensor, str]) -> torch.Tensor:
        features = self._eff_net_model.extract_features(inputs=batch)
        features = torch.mean(input=features, dim=[2, 3])

        return features
