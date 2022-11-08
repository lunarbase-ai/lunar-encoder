import logging
import os

from pydantic import ValidationError
from ts.torch_handler.base_handler import BaseHandler

from lunar_encoder.models.passage_encoder import PassageEncoder
from lunar_encoder.typing.encoder_io import (
    DefaultPassageEncoderInput,
    DefaultPassageEncoderOutput,
)
from lunar_encoder.utils import setup_logger

# logger = logging.getLogger(__name__)


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

    def preprocess(self, requests):
        encoder_input = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            encoder_input.extend(input_text)

        try:
            encoder_input = DefaultPassageEncoderInput(passages=encoder_input)
            return encoder_input
        except ValidationError:
            raise ValidationError(f"Received invalid data: {encoder_input}")
        except Exception as e:
            raise e

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.model.encode(input_instances=model_input.passages)
        model_output = self.postprocess(model_output)
        return model_output

    def postprocess(self, data):
        encoder_output = DefaultPassageEncoderOutput(embeddings=data.tolist())
        return [
            encoder_output.embeddings
        ]  # Needed when TorchServe's bach_size is fixed to 1.


if __name__ == "__main__":
    data = ["Portuguese query"]
    model_input = DefaultPassageEncoderInput(passages=data)
    print(model_input)
