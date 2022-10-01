from typing import Optional, Union

import torch

from lunar_encoder.models.losses.base_loss import BaseLoss, DistanceMetric


class TripletLoss(BaseLoss):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
        margin: float = 1.0,
        reduction: Optional[str] = "mean",
    ):
        super().__init__(distance_metric=distance_metric, reduction=reduction)
        self._margin = margin

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def margin(self):
        return self._margin

    def __call__(
        self,
        anchors: torch.Tensor,
        examples: torch.Tensor,
        num_positives: Optional[int] = None,
    ):
        positive_distance_values = self._distance_metric(
            anchors, examples[:num_positives]
        )
        negative_distance_values = self._distance_metric(
            anchors, examples[num_positives:]
        )

        loss_values = torch.relu(
            positive_distance_values - negative_distance_values + self._margin
        )
        if self._reduction is None:
            return loss_values
        elif self._reduction == "sum":
            return loss_values.sum()
        return loss_values.mean()
