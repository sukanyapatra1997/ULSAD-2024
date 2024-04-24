"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import logging
import warnings
import wandb
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters_custom
from anomalib.data import get_datamodule, DataFormat
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser

def train_all_categories(args: Namespace):

    config_filename = "config"
    config_file_extension = "yaml",

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    if args.model is None is args.config:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if args.model == "efficientad":
        warnings.warn("`efficientad` is deprecated as --model. Please use `efficient_ad` instead.", DeprecationWarning)
        args.model = "efficient_ad"

    config_path = args.config
    if config_path is None:
        config_path = Path(f"anomalib/models/{args.model}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)
    filename = Path(config_path).stem

    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    if config.dataset.categories is not None:
        for category in config.dataset.categories.split(','):

            torch.cuda.empty_cache()

            config_original: DictConfig = config.copy()
            config_original.dataset.category = category
            config_original = get_configurable_parameters_custom(config_original)

            train(config=config_original, filename = filename)
    else:
        config = get_configurable_parameters_custom(config)
        train(config=config, filename = filename)



def train(config, filename):
    """Train an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config, filename)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    args = get_parser().parse_args()
    train_all_categories(args)
