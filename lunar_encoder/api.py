from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from lunar_encoder.proxy.proxy import EncoderProxy

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

proxy = EncoderProxy()


class EncodeInput(BaseModel):
    sentences: List[str]
    modelName: Optional[str]
    tokenizerName: Optional[str]


@app.post("/encode")
def encode(body: EncodeInput):

    return proxy.encode(
        sentences=body.sentences,
        tokenizer_name=body.tokenizerName,
        model_name=body.modelName,
    )
