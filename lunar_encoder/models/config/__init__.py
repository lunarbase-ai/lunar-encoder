import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from lunar_encoder.models.losses.base_loss import DistanceMetric
from lunar_encoder.models.lunar_typing.enums import (
    Activation,
    Loss,
    Optimizer,
    Scheduler,
)


@dataclass
class EncoderConfig:
    base_transformer_name: str = field(
        default="sentence-transformers/all-mpnet-base-v2"
    )
    base_transformer_args: Dict = field(default_factory=dict)
    base_tokenizer_args: Dict = field(default_factory=dict)
    max_seq_length: int = field(default=256)
    amp: bool = field(default=True)
    cache_folder: str = field(default="/tmp/.cache/lunar/")
    pooling_method: str = field(default="mean")
    dense_hidden_dims: Optional[List[int]] = field(default=None)
    dense_output_dim: Optional[int] = field(default=None)
    dense_activation: str = field(default="tanh")
    pooled_embedding_name: str = field(default="pooled_embedding")
    pooled_attention_params: Dict = field(default_factory=dict)
    device: str = field(default="cpu")
    num_checkpoints: int = 1


class ConfigSerializationEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        enums = {Optimizer, Scheduler, Loss, Activation, DistanceMetric}
        if type(obj) in enums:
            return obj.name.upper()
        return json.JSONEncoder.default(self, obj)
