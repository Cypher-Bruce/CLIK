import os
import sys

import torch.distributed as dist
from loguru import logger

_LOGGER = None


def setup_logger(exp_dir):
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    rank = dist.get_rank()

    fmt = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{elapsed}] [rank{extra[rank]}] [{name}:{function}:{line}] {message}"
    level = "INFO" if rank == 0 else "ERROR"
    log_file = os.path.join(exp_dir, "logs", f"rank_{rank}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        log_file,
        format=fmt,
        level=level,
        rotation="500 MB",
        backtrace=True,
        diagnose=True,
    )

    _LOGGER = logger.bind(rank=rank)
    return _LOGGER


def get_logger():
    """Return the configured logger, or a plain loguru logger if setup_logger
    has not been called yet (e.g. standalone scripts)."""
    if _LOGGER is not None:
        return _LOGGER
    return logger
