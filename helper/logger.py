from __future__ import annotations

"""Central logging utilities for the pruning pipeline."""

import logging
from typing import Optional


class Logger:
    """Simple wrapper around :mod:`logging` providing a shared logger."""

    def __init__(self, name: str = "pruning") -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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


def get_logger(name: str = "pruning", level: int = logging.INFO) -> Logger:
    """Return a :class:`Logger` configured with ``name`` and ``level``."""
    log = Logger(name)
    log.set_level(level)
    return log
