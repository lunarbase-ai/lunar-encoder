"""
Copyright (C) 2022 - the LunarBase team.
This file is part of the LunarBase Framework.

Notes
-----

"""
import logging
import os
import pickle
from typing import Iterator, List, Tuple, Union, Dict

import torch
from torch import Tensor, nn

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


def pack_batch(unpacked_batch: List[Dict[str, torch.Tensor]]):
    """
    Each batch dict needs to have the same keys as the anchor.
    """
    packed_batch = {
        k: torch.stack([component[k] for component in unpacked_batch])
        for k in unpacked_batch[0].keys()
    }
    return packed_batch


def unpack_batch(packed_batch: Dict[str, torch.Tensor], num_components: int):
    """
    Returns equally sized *num_components* chunks of the original tensors as **tensor views**
    """
    unpacked_batch = {
        k: torch.vsplit(packed_batch[k], num_components) for k in packed_batch.keys()
    }
    return unpacked_batch


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


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def save_module(module: nn.Module, module_path: str):
    # module_scripted = torch.jit.script(module)  # Export to TorchScript
    # module_scripted.save(module_path)  # Save
    dir_name = os.path.dirname((os.path.abspath(module_path)))
    os.makedirs(dir_name, exist_ok=True)

    torch.save(module, module_path)


def load_module(module_path: str):
    # module = torch.jit.load(module_path)
    module = torch.load(module_path)
    module.eval()
    return module
