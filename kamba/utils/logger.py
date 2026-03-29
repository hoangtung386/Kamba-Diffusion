"""Logging configuration utilities."""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str,
    save_dir: Optional[str] = None,
    distributed_rank: int = 0,
    filename: str = "log.txt",
    mode: str = "a",
) -> logging.Logger:
    """Configure a logger with console and optional file output.

    Only rank-0 processes receive handlers; other ranks get a logger with
    no handlers so that log calls are effectively silenced.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        save_dir: Directory for the log file.  If ``None``, no file handler
            is added.
        distributed_rank: The rank of the current process.  Non-zero ranks
            receive no handlers.
        filename: Name of the log file within ``save_dir``.
        mode: File open mode for the log file.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Non-main ranks get no handlers.
    if distributed_rank > 0:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(save_dir, filename), mode=mode
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
