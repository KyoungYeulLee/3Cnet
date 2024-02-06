"""
python utility
"""
import os
import logging
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from datetime import datetime


def get_config(config_path="") -> Union[DictConfig, ListConfig]:
    if not config_path:
        config_path = Path(__file__).resolve().parent.parent / "omegaconf.yaml"

    return OmegaConf.load(config_path)


def get_logger(
    module_name: str = "",
    data_dir: str = "",
    file_name: str = "",
    ) -> logging.Logger:
    """ Get logger object for logging into one file

    Args:
        module_name (str): module name to be logged
        data_dir (str): data_dir to save the log files
            (default): the path of the source file
        file_name (str): the name of the log file.
            (default): the time when the code is run.

    Returns:
        logger (logging.Logger): formatted logger object

    Examples:
        >>> logger = get_logger(
            datetime.now().isoformat(timespec="seconds"), "SomeParser")
    """
    start_time = datetime.now().isoformat(timespec="seconds")
    if data_dir:
        logs_dir = os.path.join(data_dir, "logs")
    else:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(src_dir, "logs")

    if module_name:
        logs_dir = os.path.join(logs_dir, module_name)

    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    if file_name:
        log_file_path = os.path.join(logs_dir, f"{file_name}.log")
    else:
        log_file_path = os.path.join(logs_dir, f"{start_time}.log")
        

    logger_formatter = logging.Formatter(
        fmt="{asctime}\t{name}\t{filename}:{lineno}\t{levelname}\t{message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logger_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(logger_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(start_time)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def save_config(
    config: dict,
    save_dir: str = os.path.dirname(os.path.abspath(__file__)),
    file_name: str = datetime.now().isoformat(timespec="seconds"),
) -> None:
    """ Save the config file used to run the current file

    Args:
        config (dict): config file to be saved
        save_dir (str): the directory to save the config files
            (default): the path of the source file
        file_name (str): the name of the config file.
            (default): the time when the code is run.
    """
    save_dir = os.path.join(save_dir, "configs")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not file_name:
        file_name = datetime.now().isoformat(timespec="seconds")

    save_path = os.path.join(save_dir, f"{file_name}.yaml")

    OmegaConf.save(config=config, f=save_path)
