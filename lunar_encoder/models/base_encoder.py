import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterable
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from lunar_encoder.models.config import EncoderConfig
from lunar_encoder.utils import setup_logger


class BaseEncoder(ABC, nn.Module):
    def __init__(self, config: Optional[Union[str, EncoderConfig]] = None):
        super(BaseEncoder, self).__init__()
        self.logger = logging.getLogger()
        setup_logger(self.logger)

        if config is None:
            self._config = EncoderConfig()
        elif isinstance(config, str):
            self._config = self.load_json_config(config)
        else:
            self._config = config

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._config.device

    @staticmethod
    def load_json_config(config_file_path: str):
        with open(config_file_path) as fIn:
            config_dict = json.load(fIn)
        config = EncoderConfig()
        config.__dict__.update(config_dict)
        return config

    def _save_checkpoint(
        self, current_step: int, checkpoint_path: Optional[str] = None
    ):

        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(current_step)))

        # Delete old checkpoints
        old_checkpoints = []
        for subdir in os.listdir(checkpoint_path):
            if subdir.isdigit():
                old_checkpoints.append(
                    {"step": int(subdir), "path": os.path.join(checkpoint_path, subdir)}
                )

        if len(old_checkpoints) > self.config.num_checkpoints:
            old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
            shutil.rmtree(old_checkpoints[0]["path"])

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
    def fit(self, data_loader: DataLoader, batch_size: int = 32, num_epochs: int = 1):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass
