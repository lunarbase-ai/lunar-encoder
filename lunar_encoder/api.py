import os
from http.client import HTTPException

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from lunar_encoder.encoders.huggingface import HuggingFaceEncoder

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

model_name = os.getenv("MODEL_NAME")
model_kwargs = {'device': os.getenv("DEVICE")}
encode_kwargs = {'normalize_embeddings': os.getenv("NORMALIZE_EMBEDDINGS")}
huggingface_encoder = HuggingFaceEncoder(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


class EncodeInput(BaseModel):
    sentences: List[str]


@app.post("/encode")
def encode(body: EncodeInput):
    try:
        if os.getenv("PROVIDER") == "huggingface":
            return huggingface_encoder.embed_documents(body.sentences)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
