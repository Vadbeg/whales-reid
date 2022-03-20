"""Module with base model abstract class"""


import abc
from typing import Tuple

import torch


class BaseModel(abc.ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch: Tuple[torch.Tensor, str]) -> torch.Tensor:
        pass
