from typing import Dict

import torch

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.training.config import TrainerConfig

from lunar_encoder.training.typing import Loss, Scheduler, Optimizer


class Trainer:
    def __init__(self, config: TrainerConfig, model: BaseEncoder):
        self._config = config
        self._model = model

        # Instantiate loss
        if isinstance(self._config.loss, str):
            self._config.loss = Loss[self._config.loss.upper()]
        self._config.loss_args.update({"distance_metric": self._config.distance_metric})
        self._loss = self._config.loss(**self._config.loss_args)

        # Instantiate optimizer
        if isinstance(self._config.optimizer, str):
            self._config.optimizer = Optimizer[self._config.optimizer.upper()]
        named_params = list(self._model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self._config.optimizer_args.get(
                    "weight_decay", self._config.default_weight_decay
                ),
            },
            {
                "params": [
                    p for n, p in named_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self._optimizer = self._config.optimizer(
            optimizer_grouped_parameters, **self._config.optimizer_args
        )

        # Instantiate scheduler
        if isinstance(self._config.scheduler, str):
            self._config.scheduler = Scheduler[self._config.scheduler.upper()]
        self._scheduler = self._config.scheduler(
            self._optimizer, **self._config.scheduler_args
        )

    def training_step(self, batch_data: Dict[str, torch.Tensor]):
        with torch.autocast(
            device_type=self._model.device,
            enabled=self._config.use16,
            dtype=torch.bfloat16 if self._model.device == "cpu" else torch.float16,
        ):
            pass
            # with torch.set_grad_enabled(True):
