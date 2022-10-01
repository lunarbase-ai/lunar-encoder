import logging
import os
import subprocess

import typer

from lunar_encoder.models.config import EncoderConfig
from lunar_encoder.models.passage_encoder import PassageEncoder
from lunar_encoder.utils import setup_logger

logger = logging.getLogger()
setup_logger(logger)

archiver = "torch-model-archiver"
torchserve = "torchserve"
app = typer.Typer()


@app.callback()
def run():
    """
    LunarEncoder CLI 🚀
    """


@app.command()
def package(
    model_store: str = typer.Option(..., "--model-store"),
    model_name: str = typer.Option(..., "--model-name"),
    torchserve_handler_path: str = typer.Option(..., "--handler"),
    model_config: str = typer.Option(..., "--model-config"),
    version: str = typer.Option("1.0", "--version", help="Model's version"),
):
    model_store = os.path.abspath(model_store)
    if not os.path.isdir(model_store):
        logger.info(f"{model_store} does not exist. Packaging new model.")
        if os.path.isfile(os.path.abspath(model_config)):
            logger.info(
                f"Reading model configuration from {os.path.abspath(model_config)}"
            )
            encoder = PassageEncoder(config=os.path.abspath(model_config))
        else:
            encoder = PassageEncoder(config=EncoderConfig())
        encoder.save(model_store)

    logger.info(f"Packaging model files from {model_store}.")
    arch_proc = subprocess.run(
        [
            archiver,
            "--model-name",
            model_name,
            "-v",
            version,
            "--handler",
            torchserve_handler_path,
            "--export-path",
            model_store,
            "--extra-files",
            model_store,
        ]
    )
    if arch_proc.returncode == 0:
        logger.info(f"Packaging model files from {model_store} finished successfully.")
    else:
        raise RuntimeError(
            f"Failed to package files. Exit code is {arch_proc.returncode}."
        )


@app.command()
def deploy(
    model_store: str = typer.Option(..., "--model-store"),
    model_name: str = typer.Option(..., "--model-name"),
    model_mar: str = typer.Option(None, "--model-mar"),
    config_file: str = typer.Option(None, "--config-file"),
    snapshots: bool = typer.Option(False, "--snapshots"),
    background: bool = typer.Option(
        False, "--background", help="Run server in background"
    ),
):
    model_store = os.path.abspath(model_store)
    if not os.path.isdir(model_store):
        raise RuntimeError(f"Model store {model_store} does not exist.")

    logger.info(f"Deploying model {model_name} from {model_store}.")
    logger.info(f"Starting server for model {model_name}.")
    model_mar_file = (
        os.path.abspath(model_mar) if model_mar is not None else f"{model_name}.mar"
    )

    args = [
        "--start",
        "--model-store",
        model_store,
        "--models",
        f"{model_name}={model_mar_file}",
    ]
    if config_file is not None:
        args.extend(["--ts-config", os.path.abspath(config_file)])
    if not snapshots:
        args.extend(["--ncs"])
    if not background:
        args.extend(["--background"])

    subprocess.run([torchserve] + args)


@app.command()
def stop():
    subprocess.run([torchserve, "--stop"])
