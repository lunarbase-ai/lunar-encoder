from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterable
import numpy as np
from torch import nn, Tensor

from lunar_encoder.config import EncoderConfig


class BaseEncoder(ABC, nn.Module):
    def __init__(self, config: Optional[EncoderConfig] = None):
        super(BaseEncoder, self).__init__()

        self._config = config if config is not None else EncoderConfig()

    @property
    def config(self):
        return self._config

    @abstractmethod
    def forward(self, features: Dict[str, Tensor]):
        pass

    @abstractmethod
    def encode(
        self,
        input_instances: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        as_tensor: bool = False,
        normalize_embeddings: bool = False,
        use16: Optional[bool] = None,
    ) -> Union[np.ndarray, Tensor]:
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @staticmethod
    @abstractmethod
    def load(model_path: str):
        pass
