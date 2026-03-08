"""Structured logger for the project.

Provides a :func:`get_logger` helper that returns a logger configured with a
console handler.  The logger preserves any emoji prefixes that may be present
in the message text and uses a consistent format across the codebase.
"""

import builtins
import logging
from typing import Any

# Default log format that keeps emoji prefixes intact
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def create_error_response(message: str, code: str = "ERROR") -> dict[str, Any]:
    """
    Return a standardized error response dictionary.

    This ensures consistent error formatting across all modules.
    Includes an "error" field for backward compatibility.
    """
    return {"status": "error", "code": code, "message": message, "error": message}


def _configure_root() -> None:
    """Configure the root logger once."""
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)


# Monkey-patch builtins.print to use the logger
builtins.print = lambda *args, **kwargs: logging.getLogger().info(
    " ".join(map(str, args))
)

# Ensure root logger is configured at import time
_configure_root()


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance with the given name.

    The logger is configured to output to the console with the format defined
    in :data:`_LOG_FORMAT`.  The function is idempotent; calling it multiple
    times with the same name will return the same logger instance.
    """
    return logging.getLogger(name)
