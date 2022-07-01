import logging
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.training.losses.base_loss import BaseLoss, DistanceMetric
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
        self,
        anchors: torch.Tensor,
        examples: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Inspired by
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py
        Parameters
        ----------
        anchors
        examples
        labels

        Returns
        -------

        """
        distance_values = self._distance_metric(anchors, examples)
        loss_values = 0.5 * (
            labels.float() * distance_values.pow(2)
            + (1 - labels).float() * torch.relu(self._margin - distance_values).pow(2)
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
        # TODO: needs rethinking...

        num_texts = len(evaluation_data[0][0])
        texts = [[] for _ in range(num_texts)]
        labels = []
        for examples, label in evaluation_data:
            for idx, text in enumerate(examples):
                texts[idx].append(text)
            labels.append(label)

        anchors, positives = texts
        all_embeddings = model.encode(
            anchors + positives,
            batch_size=batch_size,
            show_progress_bar=False,
            as_tensor=as_tensor,
            normalize_embeddings=False,
            use16=True,
        )
        anchor_embeddings, positive_embeddings = (
            all_embeddings[: len(anchors), :],
            all_embeddings[len(anchors), :],
        )
        labels = np.array(labels)
        distances = self._distance_metric(anchor_embeddings, positive_embeddings)
        eval_pearson, _ = pearsonr(labels, distances)
        eval_spearman, _ = spearmanr(labels, distances)

        logger.info(
            "Distance metric {} :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                self._distance_metric.name, eval_pearson, eval_spearman
            )
        )

        return eval_pearson, eval_spearman
