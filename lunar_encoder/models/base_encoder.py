import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterable
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from lunar_encoder.models.config import EncoderConfig
from lunar_encoder.utils import setup_logger


class BaseEncoder(ABC, nn.Module):
    def __init__(self, config: Optional[EncoderConfig] = None):
        super(BaseEncoder, self).__init__()
        self.logger = logging.getLogger()
        setup_logger(self.logger)

        self._config = config if config is not None else EncoderConfig()

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._config.device

    @abstractmethod
    def forward(self, features: Dict[str, Tensor], min_out: bool = True):
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
    def fit(self,
            data_loader: DataLoader,
            batch_size: int = 32,
            num_epochs: int = 1):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass
