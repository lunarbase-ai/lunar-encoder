from lunar_encoder.generic_transformer_encoder.generic_transformer_encoder import GenericTransformerEncoder
from lunar_encoder.universal_sentence_encoder.universal_sentence_encoder import UniversalSentenceEncoder
from typing import List, Optional
from enum import Enum


class ModelType(Enum):
    GENERIC_TRANSFORMER_ENCODER = 1
    UNIVERSAL_SENTENCE_ENCODER = 2


def select_model_type(model_name: Optional[str]) -> ModelType:
    if model_name and model_name.startswith("https://tfhub.dev"):
        return ModelType.UNIVERSAL_SENTENCE_ENCODER
    else:
        return ModelType.GENERIC_TRANSFORMER_ENCODER


class EncoderProxy:
    def __init__(self):
        self.generic_transformer_encoder = GenericTransformerEncoder()
        self.universal_sentence_encoder = UniversalSentenceEncoder()

    def encode(self, sentences: List[str], tokenizer_name: Optional[str], model_name: Optional[str]):
        model_type = select_model_type(model_name)
        if model_type == ModelType.GENERIC_TRANSFORMER_ENCODER:
            encoder = self.generic_transformer_encoder
            encoder.load(tokenizer_name, model_name)
            return encoder.encode(
                sentences=sentences
            )
        elif model_type == ModelType.UNIVERSAL_SENTENCE_ENCODER:
            encoder = self.universal_sentence_encoder
            encoder.load(model_name)
            return encoder.encode(
                sentences=sentences
            )
