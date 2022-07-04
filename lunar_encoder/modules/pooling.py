"""
Inspired by https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
"""
from abc import abstractmethod, ABC
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn


class PoolingMethod(Enum):
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    MEAN_SQRT = "mean_sqrt"
    ATTN = "attention"


class PoolingModel(ABC, nn.Module):
    def __init__(self):
        super(PoolingModel, self).__init__()

    @abstractmethod
    def forward(self, input_features: Tensor, input_attention: Optional[Tensor] = None):
        pass


class AttentionPooling(PoolingModel):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int = 2,
            num_seeds: int = 1,
            dropout: float = 0.0,
    ):
        super(AttentionPooling, self).__init__()

        self.seed = nn.Parameter(torch.Tensor(1, num_seeds, embedding_dim))
        nn.init.xavier_uniform_(self.seed)
        self.cross_encoder = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, input_features: Tensor, input_attention: Optional[Tensor] = None):
        cross_attention = (1 - input_attention).bool()
        embeddings, _ = self.cross_encoder(
            query=self.seed.repeat(input_features.size(0), 1, 1),
            key=input_features,
            value=input_features,
            key_padding_mask=cross_attention,
        )
        return embeddings.squeeze(1)


class MeanPooling(PoolingModel):
    def __init__(self, with_sqrt: bool = False):
        super(MeanPooling, self).__init__()

        self._with_sqrt = with_sqrt

    @property
    def with_sqrt(self):
        return self._with_sqrt

    def forward(self, input_features: Tensor, input_attention: Optional[Tensor] = None):
        input_mask_expanded = (
            input_attention.unsqueeze(-1).expand(input_features.size()).float()
        )
        sum_embeddings = torch.sum(input_features * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        if self._with_sqrt:
            pooled_output = sum_embeddings / torch.sqrt(sum_mask)
        else:
            pooled_output = sum_embeddings / sum_mask

        return pooled_output


class MaxPooling(PoolingModel):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, input_features: Tensor, input_attention: Optional[Tensor] = None):
        input_mask_expanded = (
            input_attention.unsqueeze(-1).expand(input_features.size()).float()
        )
        input_features[input_mask_expanded == 0] = -1e9
        pooled_output = torch.max(input_features, 1)[0]

        return pooled_output


class ClsPooling(PoolingModel):
    def __init__(self):
        super(ClsPooling, self).__init__()

    def forward(self, input_features: Tensor, input_attention: Optional[Tensor] = None):
        pooled_output = input_features[:, 0, :]
        return pooled_output


class Pooling(nn.Module):
    """Performs pooling on the token embeddings.
    It generates from a variable sized sentence a fixed sized sentence embedding.
    You can concatenate multiple poolings together.
    """

    def __init__(
            self,
            pooling_method: Union[PoolingMethod, str] = PoolingMethod.CLS,
            pooled_embedding_name: str = "pooled_embedding",
            **attention_pooling_params: Optional[Any]
    ):
        super(Pooling, self).__init__()

        self._pooling_method = (
            PoolingMethod(pooling_method.lower())
            if isinstance(pooling_method, str)
            else pooling_method
        )
        self._pooled_embedding_name = pooled_embedding_name

        if self._pooling_method == PoolingMethod.MEAN:
            self._pooler = MeanPooling(with_sqrt=False)

        elif self._pooling_method == PoolingMethod.MEAN_SQRT:
            self._pooler = MeanPooling(with_sqrt=True)

        elif self._pooling_method == PoolingMethod.MAX:
            self._pooler = MaxPooling()

        elif self.pooling_method == PoolingMethod.ATTN:
            self._pooler = AttentionPooling(**attention_pooling_params)

        else:
            self._pooler = ClsPooling()

    @property
    def pooling_method(self):
        return self._pooling_method

    @property
    def pooled_embedding_name(self):
        return self._pooled_embedding_name

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]
        pooled_output = self._pooler(token_embeddings, attention_mask)

        features.update({self._pooled_embedding_name: pooled_output})
        return features
