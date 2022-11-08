from typing import List, Union

from pydantic import BaseModel, Field, validator


class DefaultPassageEncoderInput(BaseModel):
    passages: List[str] = Field(default_factory=list)

    @validator("passages", each_item=True)
    def passages_as_strings(cls, v):
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8")

        if len(v) == 0:
            raise ValueError("Encountered empty string - not allowed!")

        return v


class DefaultPassageEncoderOutput(BaseModel):
    embeddings: List[Union[float, List[float]]]
