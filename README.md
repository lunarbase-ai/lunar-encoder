# Lunar Encoder
Home of Lunar Encoder - A transformer-based sentence/passage encoder that can be used **as a service** using [TorchServe](https://pytorch.org/serve/index.html). At the moment only Huggingface models are supported.

## Getting Started
Using Lunar Encoder requires three main steps:
1. Installation
2. Model packaging
3. Model deployment

### 1. Installation

The package requires `Python 3.7+` and `Java 11+`.

For the time being we recommend to **clone this repository** and, in the project's directory, run:

`python setup.py install` (or install using `pip` directly from GitHub)

or build and install using Docker:

`docker build --tag lunar-encoder`
`docker run [--net=<some_net>] [--gpus all] lunar-encoder`

For the Docker option, `--net` options specifies the network the container will use and the `--gpus` option will give access to the host GPU. However, for GPU, additional configuration is necessary beforehand. [This documentation page](https://docs.docker.com/compose/gpu-support/) explains how to run Docker with GPU support. However, Lunar Encoder requires build time GPU access as well because of the `package` command. Accessing GPU at runtime is documented [here](https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup).

### 2. Model packaging

Once the requirements have been installed run:

`lunar-encoder package --model-store {MODEL_STORE} --model-name {MODEL_NAME} --handler {HANDLER} --model-config {ENCODER_CONFIG}`
where:

- MODEL_STORE: a path where the Transformer model is saved - this can be the result of [Huggingface's `save_pretrained` function] (https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained).
- MODEL_NAME: the entrypoint used later with each client call, e.g., *lunarenc*.
- HANDLER: a path to a Python module that defines the behaviour of the module at inference time. [TorchServe executes this code when it runs] (https://pytorch.org/serve/custom_service.html). Unless you have a good reason to use a custom handler, we recommend using `./lunar_encoder/server/handlers/passage_encoder_handler.py`.
- ENCODER_CONFIG: a path to a JSON configuration file used by the Huggingface model. An example is given in `./resources/configs/passage_encoder.json`. Make sure you configure the exemplified parameters for your system.

### 3. Model deployment

`lunar-encoder deploy --model-store {MODEL_STORE} --model-name {MODEL_NAME} --config-file {TORCH_SERVE_CONFIG}`
where:

- MODEL_STORE: the same as above.
- MODEL_NAME: the same as above.
- TORCH_SERVE_CONFIG: a path to a `.properties` configuration file used by TorchServe. An example is given in `./resources/configs/torchserve.properties`. Make sure you configure the exemplified parameters for your system.

More information about how Lunar Encoder servers pre-trained Huggingface models as a service is provided by [TorchServe's documentation](https://pytorch.org/serve/index.html).
