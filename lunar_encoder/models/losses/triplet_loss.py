import logging
from typing import Optional, Union

import torch

from lunar_encoder.models.losses.base_loss import BaseLoss
from lunar_encoder.typing import DistanceMetric
from lunar_encoder.utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


class TripletLoss(BaseLoss):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
        margin: float = 1.0,
        reduction: Optional[str] = "mean",
    ):
        super().__init__(reduction)

        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric[distance_metric.upper()]

        self._distance_metric = distance_metric
        self._margin = margin

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def margin(self):
        return self._margin

    def __call__(
        self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ):
        positive_distance_values = self._distance_metric(anchors, positives)
        negative_distance_values = self._distance_metric(anchors, negatives)

        loss_values = torch.relu(
            positive_distance_values - negative_distance_values + self._margin
        )
        if self._reduction is None:
            return loss_values
        elif self._reduction == "sum":
            return loss_values.sum()
        return loss_values.mean()
