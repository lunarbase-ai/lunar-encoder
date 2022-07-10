from enum import Enum

import torch
import transformers
from torch import nn

from lunar_encoder.models.losses.contrastive_loss import ContrastiveLoss
from lunar_encoder.models.losses.pnll_loss import PNLLLoss
from lunar_encoder.models.losses.triplet_loss import TripletLoss


class DistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    DOT = "dot"

    def __call__(self, x, y):
        if self.name == "EUCLIDEAN":
            return torch.pairwise_distance(x, y, p=2)
        elif self.name == "MANHATTAN":
            return torch.pairwise_distance(x, y, p=1)
        elif self.name == "COSINE":
            return 1 - torch.cosine_similarity(x, y)
        elif self.name == "DOT":
            return torch.matmul(x, torch.transpose(y, 0, 1))
        else:
            raise ValueError(
                "Unknown distance metric type {}. Accepted types are {}.".format(
                    self.name, DistanceMetric.__dict__.keys()
                )
            )


class Optimizer(Enum):
    ADADELTA = "adadelta"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAM_W = "adam_w"
    SGD = "sgd"

    def __call__(self, params, **kwargs):
        if self.name == "ADADELTA":
            return torch.optim.Adadelta(params, **kwargs)
        elif self.name == "ADAGRAD":
            return torch.optim.Adagrad(params, **kwargs)
        elif self.name == "ADAM":
            return torch.optim.Adam(params, **kwargs)
        elif self.name == "ADAM_W":
            return torch.optim.AdamW(params, **kwargs)
        elif self.name == "SGD":
            return torch.optim.SGD(params, **kwargs)
        else:
            raise ValueError(
                "Unknown optimizer type {}. Accepted types are {}.".format(
                    self.name, Optimizer.__dict__.keys()
                )
            )


class Scheduler(Enum):
    """
    Inspired by SBERT's https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L850
    """

    CONSTANTLR = "constantlr"
    WARMUPCONSTANT = "warmupconstant"
    WARMUPLINEAR = "warmuplinear"
    WARMUPCOSINE = "warmupcosine"
    WARMUPCOSINE_HR = "warmupcosine_hr"

    def __call__(self, optimizer: torch.optim.Optimizer, **kwargs):
        if self.name == "CONSTANTLR":
            return transformers.get_constant_schedule(optimizer, **kwargs)
        elif self.name == "WARMUPCONSTANT":
            return transformers.get_constant_schedule_with_warmup(optimizer, **kwargs)
        elif self.name == "WARMUPLINEAR":
            return transformers.get_linear_schedule_with_warmup(optimizer, **kwargs)
        elif self.name == "WARMUPCOSINE":
            return transformers.get_cosine_schedule_with_warmup(optimizer, **kwargs)
        elif self.name == "WARMUPCOSINE_HR":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, **kwargs
            )
        else:
            raise ValueError(
                "Unknown scheduler type {}. Accepted types are {}.".format(
                    self.name, Scheduler.__dict__.keys()
                )
            )


class Loss(Enum):
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    PNLL = "pnll"

    def __call__(self, **kwargs):
        if self.name == "CONTRASTIVE":
            return ContrastiveLoss(**kwargs)
        elif self.name == "TRIPLET":
            return TripletLoss(**kwargs)
        elif self.name == "PNLL":
            return PNLLLoss(**kwargs)
        else:
            raise ValueError(
                "Unknown loss type {}. Accepted types are {}.".format(
                    self.name, Loss.__dict__.keys()
                )
            )


class Activation(Enum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"

    def __call__(self):
        if self.name == "RELU":
            return nn.ReLU()
        elif self.name == "GELU":
            return nn.GELU()
        elif self.name == "TANH":
            return nn.Tanh()
        else:
            raise ValueError(
                "Unknown activation type {}. Accepted types are {}.".format(
                    self.name, Activation.__dict__.keys()
                )
            )
