# syntax=docker/dockerfile:1
FROM python:3.8-slim

ARG MODEL_STORE="/tmp/lunar_encoder"
ARG MODEL_NAME="lunarenc"
ARG HANDLER="./lunar_encoder/server/handlers/passage_encoder_handler.py"
ARG TORCH_SERVE_CONFIG="./resources/configs/torchserve.properties"

ENV LOG_LOCATION="/var/log"

# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

RUN addgroup --gid 1001 --system app && \
    adduser --home /app --shell /bin/false --disabled-password --uid 1001 --system --group app
USER app

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-warn-script-location --upgrade pip && \
    pip3 install --no-warn-script-location -r requirements.txt

COPY . .
RUN python setup.py install --user

ENV PATH="${PATH}:./.local/bin/"

RUN lunar-encoder package --model-store $MODEL_STORE --model-name $MODEL_NAME --handler $HANDLER
RUN lunar-encoder deploy --model-store $MODEL_STORE --model-name $MODEL_NAME --config-file $TORCH_SERVE_CONFIG
