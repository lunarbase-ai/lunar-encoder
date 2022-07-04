"""
Copyright (C) 2022 - the LunarBase team.
This file is part of the LunarBase Framework.
Unauthorized copying of this file, via any medium is strictly prohibited.

Notes
-----
This module defines utility function for creating and querying indexes.

[1] https://github.com/spotify/annoy
"""
import pickle
from typing import List, Iterator, Tuple, Union
from torch import nn, Tensor
import logging
import torch

from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger()


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(
        "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)


setup_logger(logger)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def iterate_embedding_files(
    vector_files: list, path_id_prefixes: List = None
) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc


def dict_batch_to_device(batch, target_device: str):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def get_linear_warmup_scheduler(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Based on https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py#L75
    Parameters
    ----------
    optimizer
    num_warmup_steps
    num_training_steps
    last_epoch

    Returns
    -------

    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def text_object_length(text: Union[List, List[List]]):
    """
    Help function to get the length for the input text. Text can be either
    a list of ints (which means a single text as input), or a tuple of list of ints
    (representing several text inputs to the model).
    """

    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, "__len__"):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
        return len(text)
    else:
        return sum([len(t) for t in text])  # Sum of length of individual strings


def save_module(module: nn.Module, module_path: str):
    module_scripted = torch.jit.script(module)  # Export to TorchScript
    module_scripted.save(module_path)  # Save


def load_module(module_path: str):
    module = torch.jit.load(module_path)
    module.eval()
    return module
