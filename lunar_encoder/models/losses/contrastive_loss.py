import logging
from typing import Optional, Union

import torch

from lunar_encoder.models.losses.base_loss import BaseLoss
from lunar_encoder.typing import DistanceMetric
from lunar_encoder.utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


class ContrastiveLoss(BaseLoss):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
        margin: float = 0.5,
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
        """
        Inspired by
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py
        Parameters
        ----------
        anchors
        positives
        negatives

        Returns
        -------

        """
        positive_distance_values = self._distance_metric(anchors, positives)
        positive_labels = torch.ones(
            positives.shape[0], dtype=torch.float16, device=anchors.device
        )

        negative_distance_values = self._distance_metric(anchors, negatives)
        negative_labels = torch.zeros(
            negatives.shape[0], dtype=torch.float16, device=anchors.device
        )

        distance_values = torch.concat(
            [positive_distance_values, negative_distance_values], dim=0
        )
        labels = torch.concat([positive_labels, negative_labels], dim=0)
        loss_values = 0.5 * (
            labels.float() * distance_values.pow(2)
            + (1 - labels).float() * torch.relu(self._margin - distance_values).pow(2)
        )
        if self._reduction is None:
            return loss_values
        elif self._reduction == "sum":
            return loss_values.sum()
        return loss_values.mean()
