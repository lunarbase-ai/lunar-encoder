from dataclasses import dataclass, field
from typing import Optional

import torch

from lunar_encoder.models.losses.base_loss import BaseLoss


@dataclass
class PassageTrainer:
    loss: Optional[BaseLoss] = field(default=None)
    optimizer: Optional[torch.optim.Optimizer] = field(default=None)
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = field(default=None)
    scaler: Optional[torch.cuda.amp.GradScaler] = field(default=None)
