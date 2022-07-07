from abc import ABC, abstractmethod
from typing import Optional


class BaseLoss(ABC):
    def __init__(self, reduction: Optional[str] = "mean"):
        self._reduction = reduction

    @property
    def reduction(self):
        return self._reduction

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
