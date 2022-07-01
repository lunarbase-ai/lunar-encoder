from typing import Dict, List, Optional, Union

from torch import Tensor, nn

from lunar_encoder import Activation


class Dense(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Union[Activation, str] = Activation.TANH,
        pooled_embedding_name: str = "pooled_embedding",
    ):
        super(Dense, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._pooled_embedding_name = pooled_embedding_name

        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self._activation = activation.value

        layer_dims = self._hidden_dims + [self._output_dim]
        in_dim = self._input_dim
        seq = []
        for dim in layer_dims:
            seq += [nn.Linear(in_features=in_dim, out_features=dim), self._activation]
            in_dim = dim
        self._dense = nn.Sequential(*seq)

    @property
    def pooled_embedding_name(self):
        return self._pooled_embedding_name

    def forward(self, features: Dict[str, Tensor]):
        features.update(
            {
                self._pooled_embedding_name: self._dense(
                    features[self._pooled_embedding_name]
                )
            }
        )
        return features
