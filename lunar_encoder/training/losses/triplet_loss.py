import logging
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.training.losses.base_loss import BaseLoss, DistanceMetric
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

    def evaluate(
        self,
        model: BaseEncoder,
        evaluation_data: Iterable[Tuple[Tuple[str], int]],
        batch_size: int = 32,
        as_tensor: bool = False,
        show_dot: bool = False,
    ):

        num_texts = len(evaluation_data[0][0])
        texts = [[] for _ in range(num_texts)]
        labels = []
        for examples, label in evaluation_data:
            for idx, text in enumerate(examples):
                texts[idx].append(text)
            labels.append(label)

        anchors, positives, negatives = texts

        all_embeddings = model.encode(
            anchors + positives + negatives,
            batch_size=batch_size,
            show_progress_bar=False,
            as_tensor=as_tensor,
            normalize_embeddings=False,
        )

        anchor_embeddings, positive_embeddings, negative_embeddings = (
            all_embeddings[: len(anchors), :],
            all_embeddings[len(anchors) : (len(anchors) + len(positives)), :],
            all_embeddings[(len(anchors) + len(positives)) :, :],
        )

        pos_distances = self._distance_metric(anchor_embeddings, positive_embeddings)
        neg_distances = self._distance_metric(anchor_embeddings, negative_embeddings)
        correct_triplets = np.array(
            [
                pos_distances[idx] < neg_distances[idx]
                for idx in range(len(pos_distances))
            ],
            dtype=np.int,
        )

        accuracy = np.mean(correct_triplets)
        logger.info(
            "Distance metric {} :\tAverage accuracy: {:.4f}".format(
                self._distance_metric.name, accuracy
            )
        )
