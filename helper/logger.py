from __future__ import annotations

"""Central logging utilities for the pruning pipeline."""

import logging
from typing import Optional


class Logger:
    """Simple wrapper around :mod:`logging` providing a shared logger."""

    def __init__(self, name: str = "pruning", log_file: Optional[str] = None) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)


def get_logger(
    name: str = "pruning",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> Logger:
    """Return a :class:`Logger` configured with ``name`` and ``level``."""
    log = Logger(name, log_file=log_file)
    log.set_level(level)
    return log
