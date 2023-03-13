from pydantic import BaseSettings, Field


class GenericTransformerEncoderSettings(BaseSettings):
    default_tokenizer_name: str = Field(env="DEFAULT_TOKENIZER_NAME")
    default_model_name: str = Field(env="DEFAULT_MODEL_NAME")
