"""Load PyTorch Lightning Loggers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Iterable

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loggers import CSVLogger, Logger

from .comet import AnomalibCometLogger
from .tensorboard import AnomalibTensorBoardLogger
from .wandb import AnomalibWandbLogger

__all__ = [
    "AnomalibCometLogger",
    "AnomalibTensorBoardLogger",
    "AnomalibWandbLogger",
    "configure_logger",
    "get_experiment_logger",
]


AVAILABLE_LOGGERS = ["tensorboard", "wandb", "csv", "comet"]


logger = logging.getLogger(__name__)


class UnknownLogger(Exception):
    """This is raised when the logger option in `config.yaml` file is set incorrectly."""


def configure_logger(level: int | str = logging.INFO) -> None:
    """Get console logger by name.

    Args:
        level (int | str, optional): Logger Level. Defaults to logging.INFO.

    Returns:
        Logger: The expected logger.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)


def get_experiment_logger(
    config: DictConfig | ListConfig,
    filename: str,
) -> Logger | Iterable[Logger] | bool:
    """Return a logger based on the choice of logger in the config file.

    Args:
        config (DictConfig): config.yaml file for the corresponding anomalib model.

    Raises:
        ValueError: for any logger types apart from false and tensorboard

    Returns:
        Logger | Iterable[Logger] | bool]: Logger
    """
    logger.info("Loading the experiment logger(s)")

    # TODO remove when logger is deprecated from project
    if "logger" in config.project.keys():
        warnings.warn(
            "'logger' key will be deprecated from 'project' section of the config file."
            " Please use the logging section in config file.",
            DeprecationWarning,
        )
        if "logging" not in config:
            config.logging = {"logger": config.project.logger, "log_graph": False}
        else:
            config.logging.logger = config.project.logger

    if config.logging.logger in (None, False):
        return False

    logger_list: list[Logger] = []
    if isinstance(config.logging.logger, str):
        config.logging.logger = [config.logging.logger]

    for experiment_logger in config.logging.logger:
        if experiment_logger == "tensorboard":
            logger_list.append(
                AnomalibTensorBoardLogger(
                    name="Tensorboard Logs",
                    save_dir=os.path.join(config.project.path, "logs"),
                    log_graph=config.logging.log_graph,
                )
            )
        elif experiment_logger == "wandb":
            wandb_logdir = os.path.join(config.project.path, "logs")
            Path(wandb_logdir).mkdir(parents=True, exist_ok=True)
            if 'patchcore' in config.model.name:
                # if config.model.use_threshold and config.model.use_memory and config.model.use_gcn:
                #     name = f"{name}_threshold_memory_gcn"
                # elif config.model.use_threshold and config.model.use_memory:
                #     name = f"{name}_threshold_memory"
                # elif config.model.use_threshold and config.model.use_gcn:
                #     name = f"{name}_threshold_gcn"
                # if config.model.use_global and config.model.use_gcn and config.model.use_memory:
                #     name = f"{name}_full"
                # elif config.model.use_gcn and config.model.use_memory:
                #     name = f"{name}_gcn_memory"
                # elif config.model.use_global and config.model.use_gcn:
                #     name = f"{name}_global_gcn"
                # elif config.model.use_global and config.model.use_memory:
                #     name = f"{name}_global_memory"
                # elif config.model.use_memory:
                #     name = f"{name}_local_memory"
                # elif config.model.use_gcn:
                #     name = f"{name}_local_gcn"
                # elif config.model.use_global:
                #     name = f"{name}_global"
                name = f"{filename} {config.dataset.category}"
                logger_list.append(
                    AnomalibWandbLogger(
                        project='deep_prior',
                        name=name,
                        save_dir=wandb_logdir,
                        entity="sp_research",
                    )
                )
            else:
                name = (
                    config.model.name
                    if "category" not in config.dataset.keys()
                    else f"{config.dataset.category} {config.model.name}"
                )
                logger_list.append(
                    AnomalibWandbLogger(
                        project=config.dataset.name,
                        name=name,
                        save_dir=wandb_logdir,
                        entity="sp_research",
                    )
                )

        elif experiment_logger == "comet":
            comet_logdir = os.path.join(config.project.path, "logs")
            Path(comet_logdir).mkdir(parents=True, exist_ok=True)
            run_name = (
                config.model.name
                if "category" not in config.dataset.keys()
                else f"{config.dataset.category} {config.model.name}"
            )
            logger_list.append(
                AnomalibCometLogger(project_name=config.dataset.name, experiment_name=run_name, save_dir=comet_logdir)
            )
        elif experiment_logger == "csv":
            logger_list.append(CSVLogger(save_dir=os.path.join(config.project.path, "logs")))
        else:
            raise UnknownLogger(
                f"Unknown logger type: {config.logging.logger}. "
                f"Available loggers are: {AVAILABLE_LOGGERS}.\n"
                f"To enable the logger, set `project.logger` to `true` or use one of available loggers in config.yaml\n"
                f"To disable the logger, set `project.logger` to `false`."
            )

    return logger_list
