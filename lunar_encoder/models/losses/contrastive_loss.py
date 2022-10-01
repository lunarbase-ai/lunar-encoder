from typing import Optional, Union

import torch

from lunar_encoder.models.losses.base_loss import BaseLoss, DistanceMetric


class ContrastiveLoss(BaseLoss):
    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.COSINE,
        margin: float = 0.5,
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
        """

        Parameters
        ----------
        anchors
        examples

        Returns
        -------

        """

        if num_positives is None:
            raise ValueError("Contrastive loss requires a known number of positives.")

        positive_distance_values = self._distance_metric(
            anchors[:num_positives], examples[:num_positives]
        )
        positive_labels = torch.ones(
            num_positives, device=anchors.device
        )

        negative_distance_values = self._distance_metric(
            anchors[num_positives:], examples[num_positives:]
        )
        negative_labels = torch.zeros(
            examples.shape[0] - num_positives,
            device=anchors.device,
        )

        distance_values = torch.cat(
            [positive_distance_values, negative_distance_values], dim=0
        )
        labels = torch.cat([positive_labels, negative_labels], dim=0)
        loss_values = 0.5 * (
            labels.float() * distance_values.pow(2)
            + (1 - labels).float() * torch.relu(self._margin - distance_values).pow(2)
        )
        if self._reduction is None:
            return loss_values
        elif self._reduction == "sum":
            return loss_values.sum()
        return loss_values.mean()
