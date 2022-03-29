"""Module with effnet models"""

import timm
import torch

from modules.models.base_model import BaseModel


class EfficientNetModel(BaseModel):
    def __init__(
        self,
        model_type: str = 'efficientnet_b0',
        in_channels: int = 3,
        num_features: int = 1000,
    ):
        super().__init__()

        self._eff_net_model = timm.create_model(
            model_type,
            pretrained=True,
            in_chans=in_channels,
        )
        out_features = self._eff_net_model.get_classifier().in_features

        self._eff_net_model.reset_classifier(num_classes=0, global_pool="avg")
        self._embeddings = torch.nn.Linear(
            in_features=out_features,
            out_features=num_features,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        prediction = self._eff_net_model(batch)
        features = self._embeddings(prediction)

        return features
