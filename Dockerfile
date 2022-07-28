# syntax=docker/dockerfile:1
FROM python:3.8-slim

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

ARG MODEL_STORE="/app/store/"
ARG MODEL_NAME="lunarenc"
ARG HANDLER="/app/lunar_encoder/server/handlers/passage_encoder_handler.py"
ARG TORCH_SERVE_CONFIG="/app/resources/configs/torchserve.properties"

ENV MODEL_STORE ${MODEL_STORE}
ENV MODEL_NAME ${MODEL_NAME}
ENV HANDLER ${HANDLER}
ENV TORCH_SERVE_CONFIG ${TORCH_SERVE_CONFIG}

RUN export MODEL_STORE
RUN export MODEL_NAME
RUN export HANDLER
RUN export TORCH_SERVE_CONFIG

ENV PATH="${PATH}:/app/.local/bin/"
ENV LOG_LOCATION="/app/var/log"

RUN export PATH
RUN export LOG_LOCATION

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-warn-script-location --upgrade pip && \
    pip3 install --no-warn-script-location -r requirements.txt

COPY . /app
RUN python setup.py install
RUN lunar-encoder package --model-store ${MODEL_STORE} --model-name ${MODEL_NAME} --handler ${HANDLER}

CMD lunar-encoder deploy --model-store ${MODEL_STORE} --model-name ${MODEL_NAME} --config-file ${TORCH_SERVE_CONFIG}
