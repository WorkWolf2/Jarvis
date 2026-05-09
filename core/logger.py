"""
Centralized logging configuration for JARVIS.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/jarvis.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> None:
    """Configure root logger with console and file handlers."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if needed
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear existing handlers
    root.handlers.clear()

    # Format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)