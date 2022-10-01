from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

import torch


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
        elif self.name == "DOT":
            return torch.matmul(x, torch.transpose(y, 0, 1))
        else:
            raise ValueError(
                "Unknown distance metric type {}. Accepted types are {}.".format(
                    self.name, DistanceMetric.__dict__.keys()
                )
            )


class BaseLoss(ABC):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.DOT,
        reduction: Optional[str] = "mean",
    ):

        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric[distance_metric.upper()]

        self._distance_metric = distance_metric
        self._reduction = reduction

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def reduction(self):
        return self._reduction

    @abstractmethod
    def __call__(
        self,
        anchors: torch.Tensor,
        examples: torch.Tensor,
        num_positives: Optional[int] = None,
    ):
        pass
