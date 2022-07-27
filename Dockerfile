# syntax=docker/dockerfile:1
FROM python:3.8-slim

ARG MODEL_STORE="/tmp/lunar_encoder"
ARG MODEL_NAME="lunarenc"
ARG HANDLER="./lunar_encoder/server/handlers/passage_encoder_handler.py"
ARG TORCH_SERVE_CONFIG="./resources/config/torchserve.properties"

ENV LOG_LOCATION="/var/log"

RUN groupadd --gid 1001 --system app && \
    useradd -d /app -m --shell /bin/false --uid 1001 --system --gid app
USER app

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .
RUN python setup.py install

RUN lunar-encoder package --model-store $MODEL_STORE --model-name $MODEL_NAME --handler $HANDLER
RUN lunar-encoder deploy --model-store $MODEL_STORE --model-name $MODEL_NAME --config-file $TORCH_SERVE_CONFIG
