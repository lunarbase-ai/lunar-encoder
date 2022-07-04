from enum import Enum
from torch import nn


class Activation(Enum):
    TANH = nn.Tanh()
    RELU = nn.ReLU()
    GELU = nn.GELU()