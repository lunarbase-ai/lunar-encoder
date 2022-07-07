import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F

from lunar_encoder.models.losses.base_loss import BaseLoss
from lunar_encoder.typing import DistanceMetric
from lunar_encoder.utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


class PNLLLoss(BaseLoss):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.DOT,
        reduction: Optional[str] = "mean",
    ):
        super().__init__(reduction)

        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric[distance_metric.upper()]

        self._distance_metric = distance_metric

    @property
    def distance_metric(self):
        return self._distance_metric

    def __call__(self, anchors: torch.Tensor, examples: torch.Tensor):
        """
        Inspired by
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py
        Parameters
        ----------
        anchors
        examples

        Returns
        -------

        """
        distance_values = self._distance_metric(anchors, examples)
        if len(anchors.size()) > 1:
            a_num = anchors.size(0)
            distance_values = anchors.view(a_num, -1)
        softmax_scores = F.log_softmax(distance_values, dim=1)
        loss_values = F.nll_loss(
            softmax_scores,
            torch.arange(0, softmax_scores.shape[0]).to(softmax_scores.device),
            reduction=self._reduction,
        )
        return loss_values
