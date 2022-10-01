from dataclasses import dataclass, field
from typing import Optional, Union, Dict

import torch
from lunar_encoder.models.lunar_typing.enums import Loss, Scheduler, Optimizer

from lunar_encoder.models.losses.base_loss import DistanceMetric


@dataclass
class PassageTrainer:
    batch_size: int = field(default=16)
    epochs: int = field(default=1)
    loss: Union[str, Loss] = field(default=Loss.PNLL)
    loss_args: Dict = field(default_factory=dict)
    scheduler: Optional[Union[str, Scheduler]] = field(default=None)
    scheduler_args: Dict = field(default_factory=dict)
    optimizer: Union[str, Optimizer] = field(default=Optimizer.ADAM)
    optimizer_args: Dict = field(default_factory=dict)
    distance_metric: DistanceMetric = field(default=DistanceMetric.DOT)
    scaler: Optional[torch.cuda.amp.GradScaler] = field(default=None)
    max_grad_norm: Optional[float] = field(default=1.0)
    grad_accumulation: int = field(default=0)
    checkpoint_path: Optional[str] = field(default=None)
    checkpoint_steps: int = field(default=1000)

    def __post_init__(self):
        if isinstance(self.loss, str):
            self.loss = Loss[self.loss.upper()]
        if isinstance(self.optimizer, str):
            self.optimizer = Optimizer[self.optimizer.upper()]
        if isinstance(self.scheduler, str):
            self.scheduler = Scheduler[self.scheduler.upper()]

        if isinstance(self.distance_metric, str):
            self.distance_metric = DistanceMetric[self.distance_metric.upper()]

        if self.loss != Loss.PNLL and self.distance_metric == DistanceMetric.DOT:
            raise ValueError(
                "DOT product as a distance metric is currently not supported for contrastive or triplet learning!"
                "Please use another metric, e.g., COSINE."
            )

        self.loss_args.update({"distance_metric": self.distance_metric})
        self.loss = self.loss(**self.loss_args)

        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer, **self.scheduler_args)
