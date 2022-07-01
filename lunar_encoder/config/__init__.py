from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EncoderConfig:
    base_transformer_name: str = field(default="bert-base-uncased")
    base_transformer_args: Dict = field(default_factory=dict)
    base_tokenizer_args: Dict = field(default_factory=dict)
    base_tokenizer_lowercase: bool = field(default=False)
    max_seq_len: int = field(default=512)
    projection_dims: List[int] = field(default=None)
    cache_folder: str = field(default=None)
    pooling_method: str = field(default="cls")
    add_dense: bool = field(default=False)
    dense_hidden_dims: Optional[List[int]] = field(default=None)
    dense_output_dim: Optional[int] = field(default=None)
    dense_activation: str = field(default="tanh")
    pooled_embedding_name: str = field(default="pooled_embedding")
    pooled_attention_params: Dict = field(default={})
    device: str = field(default="cpu")
