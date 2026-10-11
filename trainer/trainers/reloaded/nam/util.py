# File: util.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helpful utilities
"""

import importlib as _importlib
import logging as _logging
import warnings as _warnings
from datetime import datetime as _datetime
from typing import Any as _Any


def init(name: str, *args, **kwargs) -> _Any:
    """
    Extremely-powerful function to use nearly-arbitrary factories.
    """
    module_name = ".".join(name.split(".")[:-1])
    factory_name = name.split(".")[-1]
    m = _importlib.import_module(module_name)
    factory = getattr(m, factory_name)
    return factory(*args, **kwargs)


def timestamp() -> str:
    t = _datetime.now()
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"


class _FilterWarnings(object):
    """
    Context manager.

    Kinda hacky since it doesn't restore to what it was before, but to what the
    global default is.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        _warnings.filterwarnings(*self._args, **self._kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _warnings.resetwarnings()


def filter_warnings(*args, **kwargs):
    """
    Simple-but-kinda-hacky context manager that allows you to use
    `warnings.filterwarnings()` / `warnings.resetwarnings()` as if it were a
    context manager.
    """
    return _FilterWarnings(*args, **kwargs)


class _TemporaryLoggingLevels(object):
    def __init__(self, logger_names, level):
        self._logger_names = logger_names
        self._level = level
        self._loggers = []
        self._old_levels = []

    def __enter__(self):
        for name in self._logger_names:
            logger = _logging.getLogger(name)
            self._loggers.append(logger)
            self._old_levels.append(logger.level)
            logger.setLevel(self._level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger, old_level in zip(self._loggers, self._old_levels):
            logger.setLevel(old_level)


def temporary_logging_levels(logger_names, level):
    return _TemporaryLoggingLevels(logger_names, level)
