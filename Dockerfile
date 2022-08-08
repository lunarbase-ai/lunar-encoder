# syntax=docker/dockerfile:1
#FROM python:3.8-slim AS python
FROM nvidia/cuda:11.0.3-base-ubuntu18.04 AS nvidia

# Install Python 3.9
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install --no-install-recommends -y python3.9 python3-pip python3.9-dev python3-wheel python3.9-distutils build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install OpenJDK-11
RUN add-apt-repository ppa:openjdk-r/ppa && \
    apt-get update -y && \
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

COPY . /app

WORKDIR /app

RUN chown -R app:app /app

USER app

ARG MODEL_STORE="/app/store/"
ARG MODEL_NAME="lunarenc"
ARG HANDLER="/app/lunar_encoder/server/handlers/passage_encoder_handler.py"
ARG TORCH_SERVE_CONFIG="/app/resources/configs/torchserve.properties"
ARG ENCODER_CONFIG="/app/resources/configs/passage_encoder.json"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV MODEL_STORE ${MODEL_STORE}
ENV MODEL_NAME ${MODEL_NAME}
ENV HANDLER ${HANDLER}
ENV TORCH_SERVE_CONFIG ${TORCH_SERVE_CONFIG}
ENV ENCODER_CONFIG ${ENCODER_CONFIG}

RUN export MODEL_STORE
RUN export MODEL_NAME
RUN export HANDLER
RUN export TORCH_SERVE_CONFIG
RUN export ENCODER_CONFIG

ENV PATH="${PATH}:/app/.local/bin/"
ENV LOG_LOCATION="/app/var/log"

RUN export PATH
RUN export LOG_LOCATION

RUN python3.9 -m pip install --user --upgrade pip && \
    python3.9 -m pip install --user -r requirements.txt

RUN python3.9 setup.py install --user
RUN lunar-encoder package --model-store ${MODEL_STORE} --model-name ${MODEL_NAME} --handler ${HANDLER} --model-config ${ENCODER_CONFIG}

CMD lunar-encoder deploy --model-store ${MODEL_STORE} --model-name ${MODEL_NAME} --config-file ${TORCH_SERVE_CONFIG}