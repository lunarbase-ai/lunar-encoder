from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EncoderConfig:
    base_transformer_name: str = field(
        default="sentence-transformers/all-mpnet-base-v2"
    )
    base_transformer_args: Dict = field(default_factory=dict)
    base_tokenizer_args: Dict = field(default_factory=dict)
    max_seq_length: int = field(default=256)
    use16: bool = field(default=True)
    projection_dims: List[int] = field(default=None)
    cache_folder: str = field(default=None)
    pooling_method: str = field(default="cls")
    dense_hidden_dims: Optional[List[int]] = field(default=None)
    dense_output_dim: Optional[int] = field(default=None)
    dense_activation: str = field(default="tanh")
    pooled_embedding_name: str = field(default="pooled_embedding")
    pooled_attention_params: Dict = field(default_factory=dict)
    device: str = field(default="cpu")