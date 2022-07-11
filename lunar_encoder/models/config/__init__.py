from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from lunar_encoder.models.losses.base_loss import DistanceMetric
from lunar_encoder.models.lunar_typing.enums import Loss, Scheduler, Optimizer


@dataclass
class EncoderConfig:
    base_transformer_name: str = field(
        default="sentence-transformers/all-mpnet-base-v2"
    )
    base_transformer_args: Dict = field(default_factory=dict)
    base_tokenizer_args: Dict = field(default_factory=dict)
    max_seq_length: int = field(default=256)
    amp: bool = field(default=True)
    cache_folder: str = field(default=None)
    pooling_method: str = field(default="cls")
    dense_hidden_dims: Optional[List[int]] = field(default=None)
    dense_output_dim: Optional[int] = field(default=None)
    dense_activation: str = field(default="tanh")
    pooled_embedding_name: str = field(default="pooled_embedding")
    pooled_attention_params: Dict = field(default_factory=dict)
    device: str = field(default="cpu")

    # Training config
    loss: Union[str, Loss] = field(default=Loss.PNLL)
    loss_args: Dict = field(default_factory=dict)
    scheduler: Union[str, Scheduler] = field(default=Scheduler.WARMUPLINEAR)
    scheduler_args: Dict = field(default_factory=dict)
    optimizer: Union[str, Optimizer] = field(default=Optimizer.ADAM)
    optimizer_args: Dict = field(default_factory=dict)
    distance_metric: DistanceMetric = field(default=DistanceMetric.DOT)
    default_weight_decay: float = field(default=0.01)
    max_grad_norm: Optional[float] = field(default=1.0)
    grad_accumulation: int = field(default=0)
    eval_callback: callable = field(default=None)
    checkpoint_path: Optional[str] = field(default=None)
    checkpoint_steps: int = field(default=1000)
    num_checkpoints: int = 1
