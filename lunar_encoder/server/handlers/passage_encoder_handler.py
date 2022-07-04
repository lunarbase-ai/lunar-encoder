from typing import Union, Iterable, Any

from ts.torch_handler.base_handler import BaseHandler
import torch
import os
import logging

from lunar_encoder.models.passage_encoder import PassageEncoder
from lunar_encoder.utils import setup_logger

logger = logging.getLogger()
setup_logger(logger)


class PassageEncoderHandler(BaseHandler):
    """
    Main Passage Encoder handler class.
    This handler receives a collection of passages/sentences as input and returns
    the corresponding encodings based on the encoder checkpoint.
    """

    def __init__(self):
        super(PassageEncoderHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        if not os.path.isdir(model_dir):
            raise FileNotFoundError("{} : no such directory!".format(model_dir))

        self.model = PassageEncoder.load(model_dir)
        self.model.device = self.device
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def encode(
        self, input_instances: Union[str, Iterable[str]], **encoding_kwargs: Any
    ):
        return self.model.encode(input_instances=input_instances, **encoding_kwargs)

    def preprocess(self, data):
        return data

    def inference(self, data, *args, **kwargs):
        return self.encode(data, **kwargs)

    def postprocess(self, data):
        return data
