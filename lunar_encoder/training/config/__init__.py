from dataclasses import dataclass, field
from typing import Union, Optional, Dict

from lunar_encoder.training.typing import Scheduler, Optimizer, Loss, DistanceMetric


@dataclass
class TrainerConfig:
    default_epochs: int = field(default=1)
    loss: Union[str, Loss] = field(default=Loss.CONTRASTIVE)
    loss_args: Dict = field(default_factory=dict)
    scheduler: Union[str, Scheduler] = field(default=Scheduler.WARMUPLINEAR)
    scheduler_args: Dict = field(default_factory=dict)
    optimizer: Union[str, Optimizer] = field(default=Optimizer.ADAM)
    optimizer_args: Dict = field(default_factory=dict)
    distance_metric: DistanceMetric = field(default=DistanceMetric.COSINE)
    default_weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    use16: bool = field(default=True)
    checkpoint_path: Optional[str] = field(default=None)
    checkpoint_save_steps: int = field(default=500)
    checkpoint_save_total_limit: int = field(default=0)
