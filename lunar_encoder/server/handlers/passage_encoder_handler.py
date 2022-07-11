import logging
import os
from typing import Any, Iterable, Union

from ts.torch_handler.base_handler import BaseHandler

from lunar_encoder.models.passage_encoder import PassageEncoder
from lunar_encoder.utils import setup_logger


class PassageEncoderHandler(BaseHandler):
    """
    Main Passage Encoder handler class.
    This handler receives a collection of passages/sentences as input and returns
    the corresponding encodings based on the encoder checkpoint.
    """

    def __init__(self):
        super(PassageEncoderHandler, self).__init__()

        self.logger = logging.getLogger()
        setup_logger(self.logger)

        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError("{} : no such directory!".format(model_dir))

        self.model = PassageEncoder.load(model_dir)
        self.model.eval()
        self.initialized = True

    def encode(
        self, input_instances: Union[str, Iterable[str]], **encoding_kwargs: Any
    ):
        return self.model.encode(input_instances=input_instances, **encoding_kwargs)

    def preprocess(self, requests):
        inputs = []
        self.logger.info("Received raw data {}".format(requests))
        for idx, data in enumerate(requests):
            input_texts = data.get("data")
            if input_texts is None:
                input_texts = data.get("body")
            for it in input_texts:
                if isinstance(it, (bytes, bytearray)):
                    it = it.decode("utf-8")
                inputs.append(it)

        return inputs

    def inference(self, data, *args, **kwargs):
        return self.encode(data, **kwargs)

    def postprocess(self, data):
        embeddings = data.tolist()
        return [embeddings]
