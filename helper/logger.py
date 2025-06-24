from __future__ import annotations

"""Central logging utilities for the pruning pipeline."""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Callable


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def format_header(text: str, width: int = 60, fill: str = "=") -> str:
    """Return ``text`` centered within ``width`` using ``fill`` characters.

    Parameters
    ----------
    text : str
        Text to display inside the header.
    width : int, optional
        Total width of the header, by default ``60``.
    fill : str, optional
        Single character used to pad the header, by default ``"="``.
    """
    if len(fill) != 1:
        raise ValueError("fill must be a single character")
    return text.center(width, fill)


def format_subheader(text: str, width: int = 60, fill: str = "-") -> str:
    """Return subheader centered using '-' characters."""
    return format_header(text, width=width, fill=fill)


def format_step(num: int, total: int, name: str, icon: str = "📌") -> str:
    """Return a formatted step description with emoji/icon.

    Example
    -------
    >>> format_step(1, 8, "LoadModel")
    '📌 Step 1/8: LoadModel'
    """
    return f"{icon} Step {num}/{total}: {name}"


class Logger:
    """Simple wrapper around :mod:`logging` providing a shared logger."""

    def __init__(self, name: str = "pruning", log_file: Optional[str] = None) -> None:
        self.logger = logging.getLogger(name)
        # Default to DEBUG level so detailed information is captured unless
        # overridden by ``set_level`` or ``get_logger`` parameters.
        self.logger.setLevel(logging.DEBUG)
        # Prevent messages from propagating to ancestor loggers
        # which can lead to duplicate entries if the root logger is configured.
        self.logger.propagate = False
        
        # Clear existing handlers to prevent duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Add console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Add file handler if specified
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log ``msg`` with exception information."""
        self.logger.exception(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)


def add_file_handler(logger: Logger, log_file: str) -> None:
    """Attach a :class:`logging.FileHandler` to ``logger`` replacing any existing ones."""
    for h in list(logger.logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.logger.removeHandler(h)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.logger.addHandler(handler)


def get_logger(
    name: str = "pruning",
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
) -> Logger:
    """Return a :class:`Logger` configured with ``name`` and ``level``."""
    log = Logger(name, log_file=log_file)
    log.set_level(level)
    return log


@contextmanager
def timed_step(logger: "Logger", title: str, recover: Optional[Callable] = None):
    """Context manager to log the duration of *title* step.

    Example
    -------
    >>> with timed_step(logger, "Load model"):
    ...     do_something()
    2025-06-23 10:00:00 - INFO - ===== Load model =====
    2025-06-23 10:00:01 - INFO - ✅ Load model selesai dalam 1.00s
    """
    logger.info(format_header(title))
    start = time.time()
    success = True
    try:
        yield
    except Exception as exc:
        success = False
        logger.exception("❌ %s gagal: %s", title, exc)
        # Attempt recovery callback if provided
        if recover is not None:
            try:
                logger.info("🔄 Mencoba pemulihan untuk %s", title)
                recover()
                logger.info("✅ Pemulihan untuk %s berhasil", title)
            except Exception as rec_exc:  # pragma: no cover
                logger.exception("❌ Pemulihan untuk %s gagal: %s", title, rec_exc)
        raise
    finally:
        duration = time.time() - start
        if success:
            logger.info("✅ %s selesai dalam %.2fs", title, duration)
        else:
            logger.info("⚠️ %s berakhir dengan error setelah %.2fs", title, duration)


def log_block(logger: Logger, text: str, width: int = 80):
    border = "=" * width
    logger.info(border)
    logger.info(f"== {text.center(width-6)} ==")
    logger.info(border)

def log_substep(logger: Logger, text: str, width: int = 80):
    logger.info(f"---- {text}")


__all__ = [
    "Logger",
    "get_logger",
    "add_file_handler",
    "format_header",
    "format_subheader",
    "format_step",
    "timed_step",
    "log_block",
    "log_substep",
]
