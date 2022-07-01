from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Optional

import torch

from lunar_encoder.models.base_encoder import BaseEncoder


class DistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    DOT = "dot"

    def __call__(self, x, y):
        if self.name == "EUCLIDEAN":
            return torch.pairwise_distance(x, y, p=2)
        elif self.name == "MANHATTAN":
            return torch.pairwise_distance(x, y, p=1)
        elif self.name == "COSINE":
            return 1 - torch.cosine_similarity(x, y)
        else:
            return torch.matmul(x, torch.transpose(y, 0, 1))


class BaseLoss(ABC):
    def __init__(self, reduction: Optional[str] = "mean"):
        self._reduction = reduction

    @property
    def reduction(self):
        return self._reduction

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(
        self,
        model: BaseEncoder,
        evaluation_data: Iterable,
        batch_size: int = 32,
        as_tensor: bool = False,
        show_dot: bool = False,
    ):
        pass
