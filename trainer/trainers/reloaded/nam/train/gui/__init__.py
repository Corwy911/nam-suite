# File: gui.py
# Created Date: Saturday February 25th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training

Usage:
>>> from nam.train.gui import run
>>> run()
"""

import abc as _abc
import json as _json
import re as _re
import requests as _requests
import threading as _threading
import time as _time
import tkinter as _tk
import sys as _sys
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from functools import partial as _partial

try:  # Not supported in Colab
    from idlelib.tooltip import Hovertip
except ModuleNotFoundError:
    # Hovertips won't work
    class Hovertip(object):
        """
        Shell class
        """

        def __init__(self, *args, **kwargs):
            pass

from pathlib import Path as _Path
from tkinter import filedialog as _filedialog
from tkinter import messagebox as _messagebox
from tkinter import ttk as _ttk

from .nam_trainer.style import (
    FT_BODY as _FT_BODY,
    FT_BTN as _FT_BTN,
    FT_BTN_BIG as _FT_BTN_BIG,
    FT_DISPLAY as _FT_DISPLAY,
    FT_EYEBROW as _FT_EYEBROW,
    FT_LABEL as _FT_LABEL,
    FT_METRIC as _FT_METRIC,
    FT_MONO as _FT_MONO,
    FT_MONO_S as _FT_MONO_S,
    apply_style as _design_apply_style,
)
from .nam_trainer.themes import DEFAULT as _DEFAULT_THEME
from .nam_trainer.themes import THEMES as _THEMES
from .nam_trainer.widgets import divider as _design_divider
from .nam_trainer.widgets import eyebrow as _design_eyebrow
from .nam_trainer.widgets import make_metrics_widget as _make_metrics_widget
from .nam_trainer.widgets import render_metrics as _render_metrics
from .nam_trainer.widgets import status_dot as _design_status_dot
from .nam_trainer.widgets import vrule as _design_vrule
from typing import (
    Any as _Any,
    Callable as _Callable,
    List as _List,
    Dict as _Dict,
    NamedTuple as _NamedTuple,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

try:  # 3rd-party and 1st-party imports
    import torch as _torch

    from nam import __version__
    from nam.data import Split as _Split
    from nam.train import core as _core
    from nam.train.gui._resources import settings as _settings
    from nam.models.metadata import (
        GearType as _GearType,
        UserMetadata as _UserMetadata,
        ToneType as _ToneType,
    )

    # Ok private access here--this is technically allowed access
    from nam.train import metadata as _metadata
    from nam.train._names import LATEST_VERSION as _LATEST_VERSION
    from nam.train._version import (
        Version as _Version,
        get_current_version as _get_current_version,
    )
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator as _EventAccumulator,
    )

    _install_is_valid = True
    _HAVE_ACCELERATOR = _torch.cuda.is_available() or _torch.backends.mps.is_available()
except ImportError:
    _install_is_valid = False
    _HAVE_ACCELERATOR = False

if _HAVE_ACCELERATOR:
    _DEFAULT_NUM_EPOCHS = 1000 # SlaMo  # NAM 0.12.2
    _DEFAULT_BATCH_SIZE = 32   # 16     # 16
    _DEFAULT_LR = 0.002        # 0.0048 # 0.004
    _DEFAULT_LR_DECAY = 0.004  # 0.0016 # 0.007
    # Andrei (creator of REVxSTD and REVyHI) suggests these values:
    # For standard input.wav file:
    #     learning rate=0.0064
    #     learning rate decay=0.0023
    # For 50k input file:
    #     learning rate=0.0032
    #     learning rate decay=0.0023
else:
    _DEFAULT_NUM_EPOCHS = 100
    _DEFAULT_BATCH_SIZE = 32
    _DEFAULT_LR = 0.002
    _DEFAULT_LR_DECAY = 0.004

_TEXT_WIDTH = 100

_DEFAULT_DELAY = None
_DEFAULT_NY = _core._NY_DEFAULT
_DEFAULT_IGNORE_CHECKS = False
_DEFAULT_THRESHOLD_ESR = None
_DEFAULT_AUTO_FILL_RECOMMENDATIONS = False

_ADVANCED_OPTIONS_LEFT_WIDTH = 18
_ADVANCED_OPTIONS_RIGHT_WIDTH = 14
_ADVANCED_OPTIONS_LABEL_MINSIZE = 128
_METADATA_LEFT_WIDTH = 19
_METADATA_RIGHT_WIDTH = 104
_METADATA_LABEL_MINSIZE = 220
_ROW_PADY = 6
_SECTION_PADX = 28
_BUTTON_STACK_PADY = 5
_CURRENT_THEME = _DEFAULT_THEME
_CURRENT_PALETTE = dict(_THEMES[_CURRENT_THEME])
_THEME_CONFIG_PATH = _Path.home() / ".nam_trainer_ui.json"
_GUI_BG = _CURRENT_PALETTE["BG"]
_GUI_SURFACE = _CURRENT_PALETTE["SURFACE"]
_GUI_SURFACE_ALT = _CURRENT_PALETTE["SURFACE_ALT"]
_GUI_BORDER = _CURRENT_PALETTE["BORDER"]
_GUI_DIVIDER = _CURRENT_PALETTE["DIVIDER"]
_GUI_TEXT = _CURRENT_PALETTE["TEXT"]
_GUI_MUTED = _CURRENT_PALETTE["MUTED"]
_GUI_FAINT = _CURRENT_PALETTE["FAINT"]
_GUI_ACCENT = _CURRENT_PALETTE["ACCENT"]
_GUI_ACCENT_ACTIVE = _CURRENT_PALETTE["ACCENT_HOVER"]
_GUI_ACCENT_TEXT = _CURRENT_PALETTE["ACC_TEXT"]
_GUI_INPUT_BG = _GUI_SURFACE
_GUI_SELECT_BG = _CURRENT_PALETTE["BORDER"]
_GUI_BUTTON_BORDER = _GUI_TEXT
_GUI_DISABLED_BG = _CURRENT_PALETTE["SURFACE_ALT"]
_GUI_METRIC_CURRENT = _CURRENT_PALETTE["M_CURRENT"]
_GUI_METRIC_BEST = _CURRENT_PALETTE["M_BEST"]
_GUI_METRIC_LOSS = _CURRENT_PALETTE["M_LOSS"]
_GUI_METRIC_ESR = _CURRENT_PALETTE["M_ESR"]
_GUI_METRIC_MSE = _CURRENT_PALETTE["M_MSE"]
_GUI_PROGRESS_TRACK = _CURRENT_PALETTE["PROG_TRACK"]
_GUI_PROGRESS_FILL = _CURRENT_PALETTE["PROG_FILL"]
_DISPLAY_FONT = _FT_DISPLAY
_HEADER_TITLE_FONT = ("Segoe UI Light", 18)
_EYEBROW_FONT = _FT_EYEBROW
_LABEL_FONT = _FT_LABEL
_BODY_FONT = _FT_BODY
_MONO_FONT = _FT_MONO
_MONO_SMALL_FONT = _FT_MONO_S


def _scheduler_display_name(scheduler: _core.LearningRateScheduler) -> str:
    return {
        _core.LearningRateScheduler.REDUCE_ON_PLATEAU: "ReduceLROnPlateau",
        _core.LearningRateScheduler.EXPONENTIAL: "ExponentialLR",
        _core.LearningRateScheduler.COSINE_ANNEALING: "CosineAnnealingLR",
        _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS: "CosineAnnealingWarmRestarts",
        _core.LearningRateScheduler.WARMUP_COSINE_DECAY: "Warmup + Cosine Decay",
        _core.LearningRateScheduler.ONE_CYCLE: "OneCycleLR",
        _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU: "Linear Warmup + ReduceLROnPlateau",
    }.get(scheduler, scheduler.value)


def _scheduler_display_labels() -> _Dict[_core.LearningRateScheduler, str]:
    return {
        scheduler: _scheduler_display_name(scheduler)
        for scheduler in _core.LearningRateScheduler
    }


def _normalized_choice_text(value) -> str:
    return _re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _coerce_scheduler(value, fallback=None):
    if value in (None, ""):
        return fallback
    if isinstance(value, _core.LearningRateScheduler):
        return value
    try:
        return _core.LearningRateScheduler(value)
    except Exception:
        pass
    normalized = _normalized_choice_text(value)
    for scheduler in _core.LearningRateScheduler:
        if normalized in {
            _normalized_choice_text(scheduler.value),
            _normalized_choice_text(_scheduler_display_name(scheduler)),
        }:
            return scheduler
    return fallback


_ARCHITECTURE_ORDER = (
    _core.Architecture.A2_FULL_LITE,
    _core.Architecture.A2_COMPLEX_LITE,
    _core.Architecture.A2_COMPLEX_REVYLO,
    _core.Architecture.A2_COMPLEX_NANO64X4,
    _core.Architecture.A2_COMPLEX_NANO125X3,
    _core.Architecture.A2_DOUBLE_LITE,
    _core.Architecture.A2_XDOUBLE_LITE,
    _core.Architecture.A2_DOUBLEPLUS_LITE,
    _core.Architecture.COMPLEX,
    _core.Architecture.STANDARD,
    _core.Architecture.LITE,
    _core.Architecture.FEATHER,
    _core.Architecture.NANO,
    _core.Architecture.XSTD,
    _core.Architecture.XSTD3,
    _core.Architecture.XHI3,
    _core.Architecture.XHV_12,
    _core.Architecture.XHV_16,
    _core.Architecture.XHV_24,
    _core.Architecture.REVYHI,
    _core.Architecture.REVYSTD,
    _core.Architecture.REVYLO,
    _core.Architecture.REVXSTD,
    _core.Architecture.COMPLEXRF300,
    _core.Architecture.COMPLEXRF300LITE,
    _core.Architecture.COMPLEXRF600,
    _core.Architecture.COMPLEXRF600LITE,
    _core.Architecture.XCOMPLEX,
    _core.Architecture.XCOMPLEX_LITE,
    _core.Architecture.LSTM_COMPRESSOR_HQ_48X3,
    _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    _core.Architecture.LSTM_UHQ,
    _core.Architecture.LSTM_TONEX_LIKE_16,
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048,
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_HQ,
    _core.Architecture.ULTRA,
    _core.Architecture.XHQ,
    _core.Architecture.UHQ,
    _core.Architecture.NANO64X4,
    _core.Architecture.NANO125X3,
    _core.Architecture.DOUBLE,
    _core.Architecture.XDOUBLE,
    _core.Architecture.YDOUBLE,
    _core.Architecture.ACOMPLEX,
    _core.Architecture.DOUBLEPLUS,
)

_ARCHITECTURE_DISPLAY_LABELS = {
    _core.Architecture.A2_COMPLEX_REVYLO: "A2 Complex+RevYLo",
    _core.Architecture.A2_DOUBLEPLUS_LITE: "A2 DoublePlus-Lite",
    _core.Architecture.DOUBLEPLUS: "DoublePlus",
    _core.Architecture.REVYLO: "revylo",
    _core.Architecture.XHV_24: "xHV_24",
    _core.Architecture.XCOMPLEX_LITE: "xComplexLite",
    _core.Architecture.LSTM_COMPRESSOR_HQ_48X3: "LSTM Compressor HQ",
    _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3: "LSTM Compressor Lite",
    _core.Architecture.LSTM_TONEX_LIKE_16: "LSTM TX 16",
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048: "LSTM TX CC",
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_HQ: "TX HQ",
}

_ARCHITECTURE_ALIASES = {
    "a2complexrevylo": _core.Architecture.A2_COMPLEX_REVYLO,
    "a2complexnano64x4": _core.Architecture.A2_COMPLEX_NANO64X4,
    "a2complexnano125x3": _core.Architecture.A2_COMPLEX_NANO125X3,
    "a2doublelite": _core.Architecture.A2_DOUBLE_LITE,
    "a2xdoublelite": _core.Architecture.A2_XDOUBLE_LITE,
    "a2doublepluslite": _core.Architecture.A2_DOUBLEPLUS_LITE,
    "doubleplus": _core.Architecture.DOUBLEPLUS,
    "xdouble": _core.Architecture.XDOUBLE,
    "ydouble": _core.Architecture.YDOUBLE,
    "acomplex": _core.Architecture.ACOMPLEX,
    "a2complexisolated": _core.Architecture.ACOMPLEX,
    "revylo": _core.Architecture.REVYLO,
    "xhv24": _core.Architecture.XHV_24,
    "xhv_24": _core.Architecture.XHV_24,
    "xcomplexlite": _core.Architecture.XCOMPLEX_LITE,
    "lstmcompressorhq": _core.Architecture.LSTM_COMPRESSOR_HQ_48X3,
    "lstmcompressorhq48x3": _core.Architecture.LSTM_COMPRESSOR_HQ_48X3,
    "lstmcompressorlight": _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    "lstmcompressorlite": _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    "lstmcompressorlight30x3": _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    "lstmcompressorlite30x3": _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    "lstmtx16": _core.Architecture.LSTM_TONEX_LIKE_16,
    "lstmtonexlike16": _core.Architecture.LSTM_TONEX_LIKE_16,
    "lstmtxcc": _core.Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048,
    "tonexlikecausalconvlstm128162048": _core.Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048,
    "txhq": _core.Architecture.CAUSAL_CONV_LSTM_TONEX_HQ,
    "tonexhq": _core.Architecture.CAUSAL_CONV_LSTM_TONEX_HQ,
}


def _architecture_choices() -> _Tuple[_core.Architecture, ...]:
    return _ARCHITECTURE_ORDER


def _architecture_display_label(architecture: _core.Architecture) -> str:
    architecture = _core.Architecture(architecture)
    return _ARCHITECTURE_DISPLAY_LABELS.get(architecture, architecture.value)


def _architecture_display_labels() -> _Dict[_core.Architecture, str]:
    return {
        architecture: _architecture_display_label(architecture)
        for architecture in _core.Architecture
    }


def _coerce_architecture(value, fallback=None):
    if value in (None, ""):
        return fallback
    if isinstance(value, _core.Architecture):
        return value
    try:
        return _core.Architecture(value)
    except Exception:
        pass
    normalized = _normalized_choice_text(value)
    if normalized in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[normalized]
    for architecture in _core.Architecture:
        if normalized in {
            _normalized_choice_text(architecture.value),
            _normalized_choice_text(_architecture_display_label(architecture)),
        }:
            return architecture
    return fallback


def _parse_optional_float(value) -> _Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _architecture_lr_scale(architecture: _Optional[_core.Architecture]) -> float:
    if architecture is None:
        return 1.0
    try:
        if _core._is_causal_conv_lstm_architecture(architecture):
            return 0.70
        if _core._is_lstm_only_architecture(architecture):
            return 0.78
    except Exception:
        return 1.0
    if architecture in (
        _core.Architecture.ULTRA,
        _core.Architecture.COMPLEX,
        _core.Architecture.XCOMPLEX,
        _core.Architecture.UHQ,
        _core.Architecture.XHQ,
        _core.Architecture.DOUBLE,
        _core.Architecture.XDOUBLE,
        _core.Architecture.YDOUBLE,
        _core.Architecture.DOUBLEPLUS,
        _core.Architecture.ACOMPLEX,
    ):
        return 0.88
    return 1.0


def _scheduler_lr_tooltip_text(
    scheduler: _core.LearningRateScheduler, stage_name: str
) -> str:
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        return (
            f"{stage_name} Learning Rate with ExponentialLR\n\n"
            "Lower epoch counts (~50-200): 0.003 to 0.006\n"
            "Higher epoch counts (~500-1400): 0.002 to 0.004\n\n"
            "This follows the original NAM style. Use a smaller LR for refinement."
        )
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        return (
            f"{stage_name} Learning Rate with CosineAnnealingLR\n\n"
            "Lower epoch counts (~50-200): 0.0025 to 0.005\n"
            "Higher epoch counts (~500-1400): 0.0018 to 0.0035\n\n"
            "This gives a smooth one-way decay across the run, which can work well "
            "when your captures improve steadily before flattening out."
        )
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
        return (
            f"{stage_name} Learning Rate with CosineAnnealingWarmRestarts\n\n"
            "Lower epoch counts (~50-200): 0.0025 to 0.005\n"
            "Higher epoch counts (~500-1400): 0.0015 to 0.0035\n\n"
            "Because this scheduler revisits higher rates, avoid overly large "
            "starting values."
        )
    if scheduler == _core.LearningRateScheduler.WARMUP_COSINE_DECAY:
        return (
            f"{stage_name} Learning Rate with Warmup + Cosine Decay\n\n"
            "Lower epoch counts (~50-200): 0.0024 to 0.0048\n"
            "Higher epoch counts (~500-1400): 0.0016 to 0.0032\n\n"
            "Warmup protects the first updates, then cosine decay gives a smooth "
            "long glide toward refinement."
        )
    if scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        return (
            f"{stage_name} Learning Rate with OneCycleLR\n\n"
            "Lower epoch counts (~50-200): 0.002 to 0.004\n"
            "Higher epoch counts (~500-1400): 0.0012 to 0.0025\n\n"
            "This is the peak LR. OneCycle starts lower, rises early, then "
            "anneals aggressively."
        )
    if scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        return (
            f"{stage_name} Learning Rate with Linear Warmup + ReduceLROnPlateau\n\n"
            "Lower epoch counts (~50-200): 0.002 to 0.004\n"
            "Higher epoch counts (~500-1400): 0.0015 to 0.003\n\n"
            "This starts gently, then lets validation plateaus decide when LR "
            "should fall."
        )
    return (
        f"{stage_name} Learning Rate with ReduceLROnPlateau\n\n"
        "Lower epoch counts (~50-200): 0.002 to 0.004\n"
        "Higher epoch counts (~500-1400): 0.0015 to 0.003\n\n"
        "This scheduler lowers LR only after validation plateaus, so start a "
        "bit more conservatively."
    )


def _stage2_lr_ratio_range(stage1_epochs: int) -> _Tuple[float, float]:
    if stage1_epochs <= 150:
        return 0.50, 0.80
    if stage1_epochs <= 400:
        return 0.35, 0.60
    if stage1_epochs <= 800:
        return 0.20, 0.40
    return 0.10, 0.25


def _fmt_float(value: float) -> str:
    value = float(value)
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0"
    if magnitude >= 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if magnitude >= 0.001:
        return f"{value:.8f}".rstrip("0").rstrip(".")
    if magnitude >= 1.0e-8:
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return f"{value:.8e}"


def _interpolate_recommendation(
    short_value: float, long_value: float, epochs: int
) -> float:
    if epochs <= 200:
        return short_value
    if epochs >= 500:
        return long_value
    alpha = (epochs - 200) / 300.0
    return short_value + alpha * (long_value - short_value)


def _scheduler_lr_range(
    scheduler: _core.LearningRateScheduler,
) -> _Tuple[_Tuple[float, float], _Tuple[float, float]]:
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        return (0.003, 0.006), (0.002, 0.004)
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        return (0.0025, 0.005), (0.0018, 0.0035)
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
        return (0.0025, 0.005), (0.0015, 0.0035)
    if scheduler == _core.LearningRateScheduler.WARMUP_COSINE_DECAY:
        return (0.0024, 0.0048), (0.0016, 0.0032)
    if scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        return (0.002, 0.004), (0.0012, 0.0025)
    if scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        return (0.002, 0.004), (0.0015, 0.003)
    return (0.002, 0.004), (0.0015, 0.003)


def _scheduler_decay_range(
    scheduler: _core.LearningRateScheduler,
) -> _Tuple[_Tuple[float, float], _Tuple[float, float]]:
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        return (0.002, 0.006), (0.0008, 0.003)
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        return (0.05, 0.20), (0.01, 0.10)
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
        return (0.1, 0.3), (0.02, 0.15)
    if scheduler == _core.LearningRateScheduler.WARMUP_COSINE_DECAY:
        return (0.03, 0.15), (0.005, 0.05)
    if scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        return (0.003, 0.02), (0.001, 0.008)
    if scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        return (0.45, 0.75), (0.60, 0.88)
    return (0.4, 0.75), (0.6, 0.9)


def _recommended_lr_decay_for_scheduler(
    scheduler: _core.LearningRateScheduler,
    epochs: int,
    *,
    stage_two: bool = False,
    architecture: _Optional[_core.Architecture] = None,
) -> _Tuple[float, float]:
    if (
        not stage_two
        and architecture == _core.Architecture.A2_FULL_LITE
        and scheduler == _core.LearningRateScheduler.EXPONENTIAL
    ):
        return 0.004, 0.006
    if (
        not stage_two
        and architecture in (
            _core.Architecture.A2_COMPLEX_LITE,
            _core.Architecture.A2_COMPLEX_REVYLO,
            _core.Architecture.A2_COMPLEX_NANO64X4,
            _core.Architecture.A2_COMPLEX_NANO125X3,
            _core.Architecture.A2_DOUBLE_LITE,
            _core.Architecture.A2_XDOUBLE_LITE,
            _core.Architecture.A2_DOUBLEPLUS_LITE,
            _core.Architecture.ACOMPLEX,
        )
        and scheduler == _core.LearningRateScheduler.EXPONENTIAL
    ):
        return 0.003, 0.005

    (short_lr_low, short_lr_high), (long_lr_low, long_lr_high) = _scheduler_lr_range(
        scheduler
    )
    (short_decay_low, short_decay_high), (
        long_decay_low,
        long_decay_high,
    ) = _scheduler_decay_range(scheduler)

    if stage_two:
        lr_short = short_lr_low + 0.25 * (short_lr_high - short_lr_low)
        lr_long = long_lr_low + 0.25 * (long_lr_high - long_lr_low)
        if scheduler == _core.LearningRateScheduler.REDUCE_ON_PLATEAU:
            decay_short = short_decay_low + 0.75 * (short_decay_high - short_decay_low)
            decay_long = long_decay_low + 0.75 * (long_decay_high - long_decay_low)
        else:
            decay_short = short_decay_low + 0.25 * (short_decay_high - short_decay_low)
            decay_long = long_decay_low + 0.25 * (long_decay_high - long_decay_low)
    else:
        lr_short = (short_lr_low + short_lr_high) / 2.0
        lr_long = (long_lr_low + long_lr_high) / 2.0
        decay_short = (short_decay_low + short_decay_high) / 2.0
        decay_long = (long_decay_low + long_decay_high) / 2.0

    lr_scale = _architecture_lr_scale(architecture)
    lr_short *= lr_scale
    lr_long *= lr_scale

    return (
        _interpolate_recommendation(lr_short, lr_long, epochs),
        _interpolate_recommendation(decay_short, decay_long, epochs),
    )


def _stage2_decay_range(
    scheduler: _core.LearningRateScheduler, stage1_epochs: int
) -> _Tuple[float, float]:
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        if stage1_epochs <= 150:
            return 0.0015, 0.0040
        if stage1_epochs <= 400:
            return 0.0010, 0.0030
        if stage1_epochs <= 800:
            return 0.0008, 0.0020
        return 0.0005, 0.0015
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        if stage1_epochs <= 150:
            return 0.05, 0.20
        if stage1_epochs <= 400:
            return 0.03, 0.12
        if stage1_epochs <= 800:
            return 0.02, 0.08
        return 0.01, 0.05
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
        if stage1_epochs <= 150:
            return 0.05, 0.20
        if stage1_epochs <= 400:
            return 0.03, 0.15
        if stage1_epochs <= 800:
            return 0.02, 0.10
        return 0.01, 0.08
    if scheduler == _core.LearningRateScheduler.WARMUP_COSINE_DECAY:
        if stage1_epochs <= 150:
            return 0.03, 0.12
        if stage1_epochs <= 400:
            return 0.02, 0.08
        if stage1_epochs <= 800:
            return 0.01, 0.05
        return 0.005, 0.03
    if scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        if stage1_epochs <= 150:
            return 0.002, 0.012
        if stage1_epochs <= 400:
            return 0.0015, 0.008
        if stage1_epochs <= 800:
            return 0.001, 0.006
        return 0.0008, 0.004
    if scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        if stage1_epochs <= 150:
            return 0.55, 0.78
        if stage1_epochs <= 400:
            return 0.62, 0.84
        if stage1_epochs <= 800:
            return 0.70, 0.88
        return 0.78, 0.92
    if stage1_epochs <= 150:
        return 0.50, 0.75
    if stage1_epochs <= 400:
        return 0.60, 0.80
    if stage1_epochs <= 800:
        return 0.70, 0.90
    return 0.80, 0.92


def _stage2_lr_tooltip_text(
    scheduler: _core.LearningRateScheduler,
    stage1_epochs: int,
    stage1_lr: float,
) -> str:
    low_ratio, high_ratio = _stage2_lr_ratio_range(stage1_epochs)
    recommended_low = stage1_lr * low_ratio
    recommended_high = stage1_lr * high_ratio
    if stage1_epochs <= 150:
        context = "Stage 1 was short, so refinement can stay relatively close to Stage 1."
    elif stage1_epochs <= 400:
        context = "Stage 1 was moderate, so refinement should start noticeably lower."
    elif stage1_epochs <= 800:
        context = "Stage 1 was already long, so refinement should begin much lower."
    else:
        context = "Stage 1 was very extensive, so refinement should start far lower."
    short_low_ratio, short_high_ratio = _stage2_lr_ratio_range(100)
    long_low_ratio, long_high_ratio = _stage2_lr_ratio_range(1000)
    return (
        f"Refinement Learning Rate with {_scheduler_display_name(scheduler)}\n\n"
        f"Based on Stage 1 = {stage1_epochs} epochs at LR {_fmt_float(stage1_lr)}:\n"
        f"Recommended refinement LR now: {_fmt_float(recommended_low)} to {_fmt_float(recommended_high)}\n\n"
        "Reference ranges for the selected scheduler:\n"
        f"Lower Stage 1 epoch counts (~50-200): "
        f"{_fmt_float(stage1_lr * short_low_ratio)} to {_fmt_float(stage1_lr * short_high_ratio)}\n"
        f"Higher Stage 1 epoch counts (~500-1400): "
        f"{_fmt_float(stage1_lr * long_low_ratio)} to {_fmt_float(stage1_lr * long_high_ratio)}\n\n"
        f"{context}\n"
        "Use the lower end if refinement is mainly a fine-tune pass."
    )


def _stage2_decay_tooltip_text(
    scheduler: _core.LearningRateScheduler, stage1_epochs: int
) -> str:
    rec_low, rec_high = _stage2_decay_range(scheduler, stage1_epochs)
    short_low, short_high = _stage2_decay_range(scheduler, 100)
    long_low, long_high = _stage2_decay_range(scheduler, 1000)
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        meaning = (
            "This is the original NAM-style decay amount.\n"
            "Gamma is computed as 1.0 - decay."
        )
    elif scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        meaning = (
            "In this trainer, this field is the minimum-LR ratio.\n"
            "eta_min = lr * decay."
        )
    elif scheduler in (
        _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS,
        _core.LearningRateScheduler.WARMUP_COSINE_DECAY,
    ):
        meaning = (
            "In this trainer, this field is the minimum-LR ratio.\n"
            "eta_min = lr * decay."
        )
    elif scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        meaning = (
            "In this trainer, this field is the final-LR ratio.\n"
            "final_lr = peak_lr * decay."
        )
    elif scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        meaning = (
            "This field is the reduction factor used after a plateau.\n"
            "Warmup is based on the epoch count and architecture."
        )
    else:
        meaning = (
            "This field is the reduction factor used after a plateau.\n"
            "Higher values make refinement decay more gently."
        )
    return (
        f"Refinement Learning Rate Decay with {_scheduler_display_name(scheduler)}\n\n"
        f"{meaning}\n\n"
        f"Based on Stage 1 = {stage1_epochs} epochs:\n"
        f"Recommended refinement decay now: {_fmt_float(rec_low)} to {_fmt_float(rec_high)}\n\n"
        "Reference ranges for the selected scheduler:\n"
        f"Lower Stage 1 epoch counts (~50-200): {_fmt_float(short_low)} to {_fmt_float(short_high)}\n"
        f"Higher Stage 1 epoch counts (~500-1400): {_fmt_float(long_low)} to {_fmt_float(long_high)}\n\n"
        "Longer Stage 1 runs generally want a gentler refinement decay."
    )


def _scheduler_decay_tooltip_text(
    scheduler: _core.LearningRateScheduler, stage_name: str
) -> str:
    if scheduler == _core.LearningRateScheduler.EXPONENTIAL:
        return (
            f"{stage_name} Learning Rate Decay with ExponentialLR\n\n"
            "This field is the original NAM-style decay amount.\n"
            "Gamma is computed as 1.0 - decay.\n"
            "Example: 0.0023 -> gamma 0.9977\n\n"
            "Lower epoch counts (~50-200): 0.002 to 0.006\n"
            "Higher epoch counts (~500-1400): 0.0008 to 0.003\n\n"
            "Smaller values decay more slowly."
        )
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING:
        return (
            f"{stage_name} Learning Rate Decay with CosineAnnealingLR\n\n"
            "In this trainer, this field is the minimum-LR ratio.\n"
            "eta_min = lr * decay\n"
            "Example: LR 0.002 and decay 0.08 -> eta_min 0.00016\n\n"
            "Lower epoch counts (~50-200): 0.05 to 0.20\n"
            "Higher epoch counts (~500-1400): 0.01 to 0.10\n\n"
            "Lower values allow the LR to decay closer to zero by the end."
        )
    if scheduler == _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
        return (
            f"{stage_name} Learning Rate Decay with CosineAnnealingWarmRestarts\n\n"
            "In this trainer, this field is the minimum-LR ratio.\n"
            "eta_min = lr * decay\n"
            "Example: LR 0.002 and decay 0.1 -> eta_min 0.0002\n\n"
            "Lower epoch counts (~50-200): 0.1 to 0.3\n"
            "Higher epoch counts (~500-1400): 0.02 to 0.15\n\n"
            "Lower values allow deeper LR swings."
        )
    if scheduler == _core.LearningRateScheduler.WARMUP_COSINE_DECAY:
        return (
            f"{stage_name} Learning Rate Decay with Warmup + Cosine Decay\n\n"
            "In this trainer, this field is the minimum-LR ratio after warmup.\n"
            "eta_min = lr * decay\n"
            "Example: LR 0.002 and decay 0.03 -> eta_min 0.00006\n\n"
            "Lower epoch counts (~50-200): 0.03 to 0.15\n"
            "Higher epoch counts (~500-1400): 0.005 to 0.05\n\n"
            "Lower values finish with a finer, slower learning-rate glide."
        )
    if scheduler == _core.LearningRateScheduler.ONE_CYCLE:
        return (
            f"{stage_name} Learning Rate Decay with OneCycleLR\n\n"
            "In this trainer, this field is the final-LR ratio.\n"
            "final_lr = peak_lr * decay\n"
            "Example: peak LR 0.002 and decay 0.005 -> final LR 0.00001\n\n"
            "Lower epoch counts (~50-200): 0.003 to 0.02\n"
            "Higher epoch counts (~500-1400): 0.001 to 0.008\n\n"
            "Lower values make the final anneal more aggressive."
        )
    if scheduler == _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
        return (
            f"{stage_name} Learning Rate Decay with Linear Warmup + ReduceLROnPlateau\n\n"
            "This field is the reduction factor used after validation plateaus.\n"
            "Warmup length is chosen from the epoch count and architecture.\n"
            "Example: 0.7 keeps 70% of the LR after a reduction step.\n\n"
            "Lower epoch counts (~50-200): 0.45 to 0.75\n"
            "Higher epoch counts (~500-1400): 0.60 to 0.88\n\n"
            "Higher values decay more gently after the warmup period."
        )
    return (
        f"{stage_name} Learning Rate Decay with ReduceLROnPlateau\n\n"
        "This field is the reduction factor used after a plateau.\n"
        "Example: 0.666 keeps 66.6% of the LR after a reduction step.\n\n"
        "Lower epoch counts (~50-200): 0.4 to 0.75\n"
        "Higher epoch counts (~500-1400): 0.6 to 0.9\n\n"
        "Lower values drop the LR more aggressively."
    )


def _is_mac() -> bool:
    return _sys.platform == "darwin"


_SYSTEM_TEXT_COLOR = "systemTextColor" if _is_mac() else "black"


def _sync_gui_palette(P: _Dict[str, str]):
    global _CURRENT_PALETTE
    global _GUI_BG, _GUI_SURFACE, _GUI_SURFACE_ALT, _GUI_BORDER, _GUI_DIVIDER
    global _GUI_TEXT, _GUI_MUTED, _GUI_FAINT, _GUI_ACCENT, _GUI_ACCENT_ACTIVE
    global _GUI_ACCENT_TEXT, _GUI_INPUT_BG, _GUI_SELECT_BG, _GUI_BUTTON_BORDER
    global _GUI_DISABLED_BG, _GUI_METRIC_CURRENT, _GUI_METRIC_BEST
    global _GUI_METRIC_LOSS, _GUI_METRIC_ESR, _GUI_METRIC_MSE
    global _GUI_PROGRESS_TRACK, _GUI_PROGRESS_FILL

    _CURRENT_PALETTE = P
    _GUI_BG = P["BG"]
    _GUI_SURFACE = P["SURFACE"]
    _GUI_SURFACE_ALT = P["SURFACE_ALT"]
    _GUI_BORDER = P["BORDER"]
    _GUI_DIVIDER = P["DIVIDER"]
    _GUI_TEXT = P["TEXT"]
    _GUI_MUTED = P["MUTED"]
    _GUI_FAINT = P["FAINT"]
    _GUI_ACCENT = P["ACCENT"]
    _GUI_ACCENT_ACTIVE = P["ACCENT_HOVER"]
    _GUI_ACCENT_TEXT = P["ACC_TEXT"]
    _GUI_INPUT_BG = P["SURFACE"]
    _GUI_SELECT_BG = P["BORDER"]
    _GUI_BUTTON_BORDER = P["TEXT"]
    _GUI_DISABLED_BG = P["SURFACE_ALT"]
    _GUI_METRIC_CURRENT = P["M_CURRENT"]
    _GUI_METRIC_BEST = P["M_BEST"]
    _GUI_METRIC_LOSS = P["M_LOSS"]
    _GUI_METRIC_ESR = P["M_ESR"]
    _GUI_METRIC_MSE = P["M_MSE"]
    _GUI_PROGRESS_TRACK = P["PROG_TRACK"]
    _GUI_PROGRESS_FILL = P["PROG_FILL"]


def _load_theme_choice() -> str:
    try:
        saved = _json.loads(_THEME_CONFIG_PATH.read_text())
    except Exception:
        return _DEFAULT_THEME
    if not saved.get("theme_user_selected", False):
        return _DEFAULT_THEME
    theme = saved.get("theme", _DEFAULT_THEME)
    return theme if theme in _THEMES else _DEFAULT_THEME


def _save_theme_choice(name: str):
    try:
        _THEME_CONFIG_PATH.write_text(
            _json.dumps({"theme": name, "theme_user_selected": True})
        )
    except Exception:
        pass


def _apply_gui_theme(root, P: _Optional[_Dict[str, str]] = None):
    P = _CURRENT_PALETTE if P is None else P
    _sync_gui_palette(P)
    try:
        root.configure(bg=P["BG"])
    except _tk.TclError:
        pass
    _design_apply_style(root, P)
    root.option_add("*Label.Background", P["BG"])
    root.option_add("*Label.Foreground", P["TEXT"])
    root.option_add("*Frame.Background", P["BG"])
    root.option_add("*Text.Background", P["BG"])
    root.option_add("*Text.Foreground", P["TEXT"])
    root.option_add("*Text.InsertBackground", P["ACCENT"])
    root.option_add("*Menu.Background", P["SURFACE"])
    root.option_add("*Menu.Foreground", P["TEXT"])
    root.option_add("*Menu.ActiveBackground", P["ACCENT"])
    root.option_add("*Menu.ActiveForeground", P["ACC_TEXT"])
    root.option_add("*Listbox.Background", P["SURFACE"])
    root.option_add("*Listbox.Foreground", P["TEXT"])
    root.option_add("*Listbox.SelectBackground", P["ACCENT"])
    root.option_add("*Listbox.SelectForeground", P["ACC_TEXT"])

    style = _ttk.Style(root)
    style.configure(
        "Treeview",
        background=P["SURFACE"],
        fieldbackground=P["SURFACE"],
        foreground=P["TEXT"],
        rowheight=28,
        bordercolor=P["BORDER"],
        lightcolor=P["BORDER"],
        darkcolor=P["BORDER"],
    )
    style.configure(
        "Treeview.Heading",
        background=P["SURFACE_ALT"],
        foreground=P["TEXT"],
        relief="flat",
        font=_FT_BTN,
    )
    style.map(
        "Treeview.Heading",
        background=[("active", P["ACCENT"])],
        foreground=[("active", P["ACC_TEXT"])],
    )

def _eyebrow(parent, text: str, prefix: str = "◆"):
    return _design_eyebrow(parent, text, _CURRENT_PALETTE, prefix=prefix)


def _divider(parent) -> _tk.Frame:
    return _design_divider(parent, _CURRENT_PALETTE)


def _vrule(parent, height: int = 18) -> _tk.Frame:
    return _design_vrule(parent, _CURRENT_PALETTE, height=height)


def _status_dot(parent, color_key: str = "M_BEST"):
    return _design_status_dot(parent, _CURRENT_PALETTE, color_key)


def _style_toplevel(
    window: _tk.Toplevel,
    width: int = 500,
    height: int = 760,
    min_width: _Optional[int] = None,
    min_height: _Optional[int] = None,
):
    window.configure(bg=_GUI_BG)
    window.geometry(f"{width}x{height}")
    resolved_min_width = max(360, width - 80) if min_width is None else min_width
    resolved_min_height = max(420, height - 140) if min_height is None else min_height
    window.minsize(resolved_min_width, resolved_min_height)


def _position_window_center_screen(window: _tk.Misc, width: int, height: int):
    try:
        window.update_idletasks()
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
    except _tk.TclError:
        pass


def _position_toplevel_right_of_parent(
    window: _tk.Toplevel,
    parent: _tk.Misc,
    width: int,
    height: int,
    gap: int = 12,
):
    try:
        parent.update_idletasks()
        window.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = parent_x + parent_w + gap
        if x + width > screen_w:
            x = max(0, screen_w - width - gap)
        y = max(0, min(parent_y, screen_h - height - gap))
        window.geometry(f"{width}x{height}+{x}+{y}")
    except _tk.TclError:
        pass


def _style_combobox_popdown(combobox: _ttk.Combobox):
    try:
        popdown = combobox.tk.eval(f"ttk::combobox::PopdownWindow {combobox}")
        combobox.tk.call(
            popdown,
            "configure",
            "-background",
            _GUI_DIVIDER,
            "-borderwidth",
            1,
            "-relief",
            "solid",
        )
        listbox = f"{popdown}.f.l"
        combobox.tk.call(
            listbox,
            "configure",
            "-background",
            _GUI_SURFACE,
            "-foreground",
            _GUI_TEXT,
            "-selectbackground",
            _GUI_ACCENT,
            "-selectforeground",
            _GUI_ACCENT_TEXT,
            "-highlightthickness",
            0,
            "-borderwidth",
            0,
            "-relief",
            "flat",
            "-activestyle",
            "none",
            "-font",
            _FT_BODY,
        )
    except _tk.TclError:
        pass


class _FlatVerticalScrollbar(_tk.Canvas):
    def __init__(self, parent, command=None, width: int = 8):
        super().__init__(
            parent,
            width=width,
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg=_GUI_SURFACE,
            cursor="arrow",
        )
        self._command = command
        self._first = 0.0
        self._last = 1.0
        self._thumb = None
        self._drag_offset = 0
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Button-1>", self._jump_to)
        self.bind("<B1-Motion>", self._drag_to)

    def set(self, first, last):
        self._first = max(0.0, min(1.0, float(first)))
        self._last = max(self._first, min(1.0, float(last)))
        self._redraw()

    def recolor(self):
        self.configure(bg=_GUI_SURFACE)
        self._redraw()

    def _thumb_bounds(self):
        height = max(1, self.winfo_height())
        visible = max(0.02, self._last - self._first)
        thumb_h = min(height, max(28, int(height * visible)))
        travel = max(1, height - thumb_h)
        scrollable = max(0.000001, 1.0 - visible)
        top = int(travel * max(0.0, min(1.0, self._first / scrollable)))
        bottom = min(height, top + thumb_h)
        return top, bottom

    def _redraw(self):
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.create_rectangle(
            width // 2,
            0,
            width // 2 + 1,
            height,
            fill=_GUI_DIVIDER,
            outline="",
        )
        if self._first <= 0.0 and self._last >= 1.0:
            return
        top, bottom = self._thumb_bounds()
        x0 = max(1, width // 2 - 2)
        x1 = min(width, x0 + 4)
        self._thumb = self.create_rectangle(
            x0,
            top,
            x1,
            bottom,
            fill=_GUI_BORDER,
            outline="",
        )

    def _moveto(self, y):
        if self._command is None:
            return
        height = max(1, self.winfo_height())
        top, bottom = self._thumb_bounds()
        thumb_h = max(1, bottom - top)
        travel = max(1, height - thumb_h)
        visible = max(0.02, self._last - self._first)
        scrollable = max(0.0, 1.0 - visible)
        fraction = max(0.0, min(scrollable, ((y - (thumb_h / 2)) / travel) * scrollable))
        self._command("moveto", fraction)

    def _jump_to(self, event):
        top, bottom = self._thumb_bounds()
        if top <= event.y <= bottom:
            self._drag_offset = event.y - top
            return
        self._moveto(event.y)

    def _drag_to(self, event):
        if self._command is None:
            return
        height = max(1, self.winfo_height())
        top, bottom = self._thumb_bounds()
        thumb_h = max(1, bottom - top)
        travel = max(1, height - thumb_h)
        visible = max(0.02, self._last - self._first)
        scrollable = max(0.0, 1.0 - visible)
        fraction = max(0.0, min(scrollable, ((event.y - self._drag_offset) / travel) * scrollable))
        self._command("moveto", fraction)


class _FlatCheckbutton(_tk.Frame):
    def __init__(self, parent, text: str, variable: _tk.BooleanVar):
        super().__init__(parent, bg=_GUI_BG, bd=0, highlightthickness=0)
        self._variable = variable
        self._state = _tk.NORMAL
        self._box = _tk.Canvas(
            self,
            width=12,
            height=12,
            bd=0,
            highlightthickness=0,
            bg=_GUI_BG,
            cursor="hand2",
        )
        self._box.pack(side=_tk.LEFT, padx=(0, 5))
        self._label = _tk.Label(
            self,
            text=text,
            bg=_GUI_BG,
            fg=_GUI_TEXT,
            font=_FT_BODY,
            cursor="hand2",
        )
        self._label.pack(side=_tk.LEFT)
        self._trace_name = self._variable.trace_add("write", lambda *_args: self._redraw())
        for widget in (self, self._box, self._label):
            widget.bind("<Button-1>", self._toggle)
        self._redraw()

    def __setitem__(self, key, value):
        if key != "state":
            super().__setitem__(key, value)
            return
        self._state = value
        cursor = "" if value == _tk.DISABLED else "hand2"
        self._box.configure(cursor=cursor)
        self._label.configure(cursor=cursor)
        self._redraw()

    def __getitem__(self, key):
        if key == "state":
            return self._state
        return super().__getitem__(key)

    def _toggle(self, _event=None):
        if self._state == _tk.DISABLED:
            return
        self._variable.set(not self._variable.get())

    def recolor(self):
        self.configure(bg=_GUI_BG)
        self._box.configure(bg=_GUI_BG)
        self._label.configure(bg=_GUI_BG)
        self._redraw()

    def _redraw(self):
        selected = bool(self._variable.get())
        disabled = self._state == _tk.DISABLED
        border = _GUI_MUTED if disabled else _GUI_MUTED
        fill = _GUI_ACCENT if selected else _GUI_SURFACE
        label_fg = _GUI_MUTED if disabled else _GUI_TEXT
        check = _GUI_ACCENT_TEXT if selected else fill
        self._label.configure(fg=label_fg)
        self._box.delete("all")
        self._box.create_rectangle(
            0, 0, 11, 11, fill=fill, outline=border, width=1, tags=("box",)
        )
        if selected:
            self._box.create_line(
                3,
                6,
                5,
                8,
                9,
                3,
                fill=check,
                width=2,
                capstyle=_tk.ROUND,
                joinstyle=_tk.ROUND,
            )


def _get_latest_version_from_github() -> _Optional[_Version]:
    """
    Fetch releases from GitHub and return the newest version, or None on error.
    """
    url = "https://api.github.com/repos/sdatkinson/neural-amp-modeler/releases"
    try:
        response = _requests.get(url)
    except _requests.exceptions.ConnectionError:
        print("WARNING: Failed to reach the server to check for updates")
        return None
    if response.status_code != 200:
        print(f"Failed to fetch releases. Status code: {response.status_code}")
        return None
    releases = response.json()
    latest_version = None
    if releases:
        for release in releases:
            tag = release["tag_name"]
            if not tag.startswith("v"):
                print(f"Found invalid version {tag}")
            else:
                this_version = _Version.from_string(tag[1:])
                if latest_version is None or this_version > latest_version:
                    latest_version = this_version
    else:
        print("No releases found for this repository.")
    return latest_version


@_dataclass
class AdvancedOptions(object):
    """
    :param architecture: Which architecture to use.
    :param num_epochs: How many epochs to train for.
    :param latency: Latency between the input and output audio, in samples.
        None means we don't know and it has to be calibrated.
    :param ignore_checks: Keep going even if a check says that something is wrong.
    :param threshold_esr: Stop training if the ESR gets better than this. If None, don't
        stop.
    """

    architecture: _core.Architecture
    num_epochs: int
    lr: float
    lr_decay: float
    lr_scheduler: _core.LearningRateScheduler
    batch_size: int
    ny: int
    latency: _Optional[int]
    ignore_checks: bool
    threshold_esr: _Optional[float]
    stage_mode: _core.TrainingStageMode
    stage2_epochs: int
    stage2_lr: float
    stage2_lr_decay: float
    stage2_lr_scheduler: _core.LearningRateScheduler
    stage2_focus: _core.StageTwoFocus
    checkpoint_save_mode: _core.CheckpointSaveMode
    auto_fill_recommendations: bool = _DEFAULT_AUTO_FILL_RECOMMENDATIONS


_ADVANCED_OPTIONS_WIDGET_FIELDS = (
    "architecture",
    "num_epochs",
    "lr",
    "lr_decay",
    "lr_scheduler",
    "batch_size",
    "ny",
    "latency",
    "threshold_esr",
    "stage_mode",
    "stage2_epochs",
    "stage2_lr",
    "stage2_lr_decay",
    "stage2_lr_scheduler",
    "stage2_focus",
    "checkpoint_save_mode",
)


_METADATA_FIELD_NAMES = (
    "name",
    "modeled_by",
    "gear_make",
    "gear_model",
    "gear_type",
    "tone_type",
    "input_level_dbu",
    "output_level_dbu",
)

_DEFAULT_CHECKBOX_VALUES = {
    "silent_training": False,
    "save_plot": True,
    "advanced_init_info": False,
}


def _default_advanced_options() -> AdvancedOptions:
    default_stage1_scheduler = _core.LearningRateScheduler.EXPONENTIAL
    default_stage2_scheduler = _core.LearningRateScheduler.REDUCE_ON_PLATEAU
    default_architecture = _core.Architecture.COMPLEX
    default_stage1_lr, default_stage1_decay = _recommended_lr_decay_for_scheduler(
        default_stage1_scheduler,
        _DEFAULT_NUM_EPOCHS,
        architecture=default_architecture,
    )
    default_stage2_epochs = 100
    default_stage2_lr, default_stage2_decay = _recommended_lr_decay_for_scheduler(
        default_stage2_scheduler,
        default_stage2_epochs,
        stage_two=True,
        architecture=default_architecture,
    )
    return AdvancedOptions(
        default_architecture,
        _DEFAULT_NUM_EPOCHS,
        default_stage1_lr,
        default_stage1_decay,
        default_stage1_scheduler,
        _DEFAULT_BATCH_SIZE,
        _DEFAULT_NY,
        _DEFAULT_DELAY,
        _DEFAULT_IGNORE_CHECKS,
        _DEFAULT_THRESHOLD_ESR,
        _core.TrainingStageMode.SINGLE_STAGE,
        default_stage2_epochs,
        default_stage2_lr,
        default_stage2_decay,
        default_stage2_scheduler,
        _core.StageTwoFocus.LOW,
        _core.CheckpointSaveMode.MINIMAL,
        _DEFAULT_AUTO_FILL_RECOMMENDATIONS,
    )


def _serialize_advanced_options(options: AdvancedOptions) -> _Dict[str, _Any]:
    return {
        "architecture": options.architecture.value,
        "num_epochs": options.num_epochs,
        "lr": options.lr,
        "lr_decay": options.lr_decay,
        "lr_scheduler": options.lr_scheduler.value,
        "batch_size": options.batch_size,
        "ny": options.ny,
        "latency": options.latency,
        "ignore_checks": options.ignore_checks,
        "threshold_esr": options.threshold_esr,
        "stage_mode": options.stage_mode.value,
        "stage2_epochs": options.stage2_epochs,
        "stage2_lr": options.stage2_lr,
        "stage2_lr_decay": options.stage2_lr_decay,
        "stage2_lr_scheduler": options.stage2_lr_scheduler.value,
        "stage2_focus": options.stage2_focus.value,
        "checkpoint_save_mode": options.checkpoint_save_mode.value,
        "auto_fill_recommendations": bool(options.auto_fill_recommendations),
    }


def _advanced_options_from_dict(
    saved: _Optional[_Dict[str, _Any]], fallback: _Optional[AdvancedOptions] = None
) -> AdvancedOptions:
    default = _default_advanced_options()
    if fallback is not None:
        default = fallback
    if not saved:
        return default

    def _parse_enum(enum_cls, value, fallback, aliases: _Optional[_Dict[str, str]] = None):
        if value is None:
            return fallback
        alias_map = {} if aliases is None else aliases
        normalized = alias_map.get(value, value)
        try:
            return enum_cls(normalized)
        except Exception:
            return fallback

    def _parse_int(value, fallback):
        try:
            return int(value)
        except Exception:
            return fallback

    def _parse_positive_int(value, fallback):
        parsed = _parse_int(value, fallback)
        if parsed < 1:
            return fallback
        return parsed

    def _parse_float(value, fallback):
        try:
            return float(value)
        except Exception:
            return fallback

    def _parse_optional_float(value, fallback):
        if value in (None, ""):
            return fallback
        try:
            return float(value)
        except Exception:
            return fallback

    def _parse_optional_int(value, fallback):
        if value in (None, ""):
            return fallback
        try:
            return int(value)
        except Exception:
            return fallback

    def _parse_bool(value, fallback):
        if value is None:
            return fallback
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            return fallback
        return bool(value)

    stage_two_focus_aliases = {
        "low": _core.StageTwoFocus.LOW.value,
        "mid": _core.StageTwoFocus.MID.value,
        "high": _core.StageTwoFocus.HIGH.value,
    }
    checkpoint_mode_aliases = {
        "Maximum": _core.CheckpointSaveMode.MAXIMUM.value,
        "Minimum": _core.CheckpointSaveMode.MINIMAL.value,
        "Four Best + Current": _core.CheckpointSaveMode.FOUR_BEST_PLUS_CURRENT.value,
        "4 Best + Current Run": _core.CheckpointSaveMode.FOUR_BEST_PLUS_CURRENT.value,
    }

    return AdvancedOptions(
        _parse_enum(
            _core.Architecture,
            saved.get("architecture"),
            default.architecture,
        ),
        _parse_int(saved.get("num_epochs"), default.num_epochs),
        _parse_float(saved.get("lr"), default.lr),
        _parse_float(saved.get("lr_decay"), default.lr_decay),
        _parse_enum(
            _core.LearningRateScheduler,
            saved.get("lr_scheduler"),
            default.lr_scheduler,
        ),
        _parse_positive_int(saved.get("batch_size"), default.batch_size),
        _parse_positive_int(saved.get("ny"), default.ny),
        _parse_optional_int(saved.get("latency"), default.latency),
        bool(saved.get("ignore_checks", default.ignore_checks)),
        _parse_optional_float(saved.get("threshold_esr"), default.threshold_esr),
        _parse_enum(
            _core.TrainingStageMode,
            saved.get("stage_mode"),
            default.stage_mode,
        ),
        _parse_int(saved.get("stage2_epochs"), default.stage2_epochs),
        _parse_float(saved.get("stage2_lr"), default.stage2_lr),
        _parse_float(saved.get("stage2_lr_decay"), default.stage2_lr_decay),
        _parse_enum(
            _core.LearningRateScheduler,
            saved.get("stage2_lr_scheduler"),
            default.stage2_lr_scheduler,
        ),
        _parse_enum(
            _core.StageTwoFocus,
            saved.get("stage2_focus"),
            default.stage2_focus,
            aliases=stage_two_focus_aliases,
        ),
        _parse_enum(
            _core.CheckpointSaveMode,
            saved.get("checkpoint_save_mode"),
            default.checkpoint_save_mode,
            aliases=checkpoint_mode_aliases,
        ),
        _parse_bool(
            saved.get("auto_fill_recommendations"),
            default.auto_fill_recommendations,
        ),
    )


def _load_advanced_options_from_settings() -> AdvancedOptions:
    return _advanced_options_from_dict(_settings.get_advanced_options_settings())


def _serialize_user_metadata(
    metadata: _UserMetadata, enabled: bool
) -> _Dict[str, _Any]:
    values = {}
    for name in _METADATA_FIELD_NAMES:
        value = getattr(metadata, name)
        values[name] = value.value if isinstance(value, _Enum) else value
    return {"enabled": bool(enabled), "values": values}


def _load_user_metadata_from_settings() -> _Tuple[_UserMetadata, bool]:
    saved = _settings.get_metadata_settings()
    values = saved.get("values") or {}
    kwargs = {}
    try:
        for name in _METADATA_FIELD_NAMES:
            if name not in values:
                continue
            value = values.get(name)
            if name == "gear_type" and value is not None:
                kwargs[name] = _GearType(value)
            elif name == "tone_type" and value is not None:
                kwargs[name] = _ToneType(value)
            else:
                kwargs[name] = value
        metadata = _UserMetadata(**kwargs)
    except Exception:
        metadata = _UserMetadata()
    enabled = bool(
        saved.get(
            "enabled",
            any(getattr(metadata, name) is not None for name in _METADATA_FIELD_NAMES),
        )
    )
    return metadata, enabled


def _copy_user_metadata(metadata: _UserMetadata) -> _UserMetadata:
    try:
        return metadata.model_copy(deep=True)
    except AttributeError:
        kwargs = {name: getattr(metadata, name) for name in _METADATA_FIELD_NAMES}
        return _UserMetadata(**kwargs)


def _load_checkbox_values_from_settings() -> _Dict[str, bool]:
    saved = _settings.get_checkbox_settings()
    values = dict(_DEFAULT_CHECKBOX_VALUES)
    for key, default_value in _DEFAULT_CHECKBOX_VALUES.items():
        values[key] = bool(saved.get(key, default_value))
    return values


class _PathType(_Enum):
    FILE = "file"
    DIRECTORY = "directory"
    MULTIFILE = "multifile"


_RESUME_CHECKPOINT_BUTTON_TEXT = "Resume Training From CKPT"
_RESUME_CHECKPOINT_BUTTON_CHARS = 26
_RESUME_CHECKPOINT_SCROLL_MS = 220
_RESUME_CHECKPOINT_FILETYPES = (
    ("NAM checkpoints", "*.ckpt"),
    ("All files", "*.*"),
)
_A2_CHECKPOINT_SUBMODEL_COUNT = 2
_ResumeCheckpointSelection = _Union[str, _Tuple[str, ...]]

def _load_path_selection(
    path_key: _settings.PathKey, path_type: _PathType
) -> _Optional[_Union[str, _Tuple[str, ...]]]:
    saved = _settings.get_path_selection(path_key)
    if saved is None:
        saved_path = _settings.get_last_path(path_key)
        if saved_path is None:
            return None
        saved = str(saved_path)

    if path_type == _PathType.MULTIFILE:
        values = saved if isinstance(saved, tuple) else (saved,)
        paths = tuple(
            str(path) for path in values if path and _Path(str(path)).is_file()
        )
        return paths or None

    if isinstance(saved, tuple):
        if len(saved) == 0:
            return None
        saved = saved[0]

    path = _Path(str(saved))
    if path_type == _PathType.FILE and not path.is_file():
        return None
    if path_type == _PathType.DIRECTORY and not path.is_dir():
        return None
    return str(path)


class _PathButton(object):
    """
    Button and the path
    """

    def __init__(
        self,
        frame: _tk.Frame,
        button_text: str,
        info_str: str,
        path_type: _PathType,
        path_key: _settings.PathKey,
        hooks: _Optional[_Sequence[_Callable[[], None]]] = None,
        color_when_not_set: _Optional[str] = None,
        color_when_set: _Optional[str] = None,
        default: _Optional[_Path] = None,
        grid_row: _Optional[int] = None,
        path_is_hint: bool = False,
    ):
        """
        :param hooks: Callables run at the end of setting the value.
        """
        self._button_text = button_text
        self._info_str = info_str
        self._path: _Optional[_Path] = default
        self._path_type = path_type
        self._path_key = path_key
        self._frame = frame
        self._grid_row = grid_row
        self._path_is_hint = path_is_hint
        self._widgets = {}
        self._widgets["button"] = _ttk.Button(
            self._frame,
            text=button_text,
            command=self._set_val,
            style="Ink.TButton",
        )
        self._widgets["path_frame"] = _tk.Frame(self._frame, bg=_GUI_BG)
        self._widgets["prefix"] = _ttk.Label(
            self._widgets["path_frame"],
            text="›",
            background=_GUI_BG,
            foreground=_GUI_FAINT,
            font=_FT_LABEL,
        )
        self._widgets["prefix"].pack(side=_tk.LEFT, padx=(0, 8))
        self._widgets["label"] = _ttk.Label(
            self._widgets["path_frame"],
            style="Mono.TLabel",
        )
        self._widgets["label"].pack(side=_tk.LEFT)
        if grid_row is None:
            self._widgets["button"].pack(side=_tk.LEFT, padx=(0, 18))
            self._widgets["path_frame"].pack(side=_tk.LEFT)
        else:
            self._widgets["button"].grid(
                row=grid_row, column=0, sticky="ew", padx=(0, 18), pady=4
            )
            self._widgets["path_frame"].grid(row=grid_row, column=1, sticky="w")
        self._hooks = hooks
        self._color_when_not_set = (
            _GUI_FAINT if color_when_not_set is None else color_when_not_set
        )
        self._color_when_set = _GUI_MUTED if color_when_set is None else color_when_set
        self._set_text()

    def __setitem__(self, key, val):
        """
        Implement tk-style setter for state
        """
        if key == "state":
            for widget in self._widgets.values():
                try:
                    widget["state"] = val
                except _tk.TclError:
                    pass
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} instance does not support item assignment for non-state key {key}!"
            )

    @property
    def val(self) -> _Optional[_Path]:
        return self._path

    def _set_text(self):
        if self._path is None:
            self._widgets["label"].configure(
                foreground=self._color_when_not_set,
                text=self._info_str,
            )
        else:
            val = self.val
            val = val[0] if isinstance(val, tuple) and len(val) == 1 else val
            self._widgets["label"].configure(
                foreground=_GUI_FAINT if self._path_is_hint else self._color_when_set,
                text=str(val),
            )

    def recolor(self):
        self._color_when_not_set = _GUI_FAINT
        self._color_when_set = _GUI_MUTED
        for key in ("path_frame",):
            self._widgets[key].configure(bg=_GUI_BG)
        self._widgets["prefix"].configure(background=_GUI_BG, foreground=_GUI_FAINT)
        self._widgets["label"].configure(background=_GUI_BG)
        self._set_text()

    def _set_val(self):
        last_path = _settings.get_last_path(self._path_key)
        if last_path is None:
            initial_dir = None
        elif not last_path.is_dir():
            initial_dir = last_path.parent
        else:
            initial_dir = last_path
        result = {
            _PathType.FILE: _filedialog.askopenfilename,
            _PathType.DIRECTORY: _filedialog.askdirectory,
            _PathType.MULTIFILE: _filedialog.askopenfilenames,
        }[self._path_type](initialdir=str(initial_dir))
        if result:
            self._path = result
            _settings.set_last_path(
                self._path_key,
                _Path(result[0] if self._path_type == _PathType.MULTIFILE else result),
            )
            _settings.set_path_selection(self._path_key, result)
        self._set_text()

        if self._hooks is not None:
            for h in self._hooks:
                h()



class _CheckboxKeys(_Enum):
    """
    Keys for checkboxes
    """

    SILENT_TRAINING = "silent_training"
    SAVE_PLOT = "save_plot"
    ADVANCED_INIT_INFO = "advanced_init_info"


class _TopLevelWithOk(_tk.Toplevel):
    """
    Toplevel with an Ok button (provide yourself!)
    """

    def __init__(
        self, on_ok: _Callable[[None], None], resume_main: _Callable[[None], None]
    ):
        """
        :param on_ok: What to do when "Ok" button is pressed
        """
        super().__init__()
        self.configure(bg=_GUI_BG)
        self._on_ok = on_ok
        self._resume_main = resume_main

    def destroy(self, pressed_ok: bool = False):
        if pressed_ok:
            self._on_ok()
        self._resume_main()
        super().destroy()


class _TopLevelWithYesNo(_tk.Toplevel):
    """
    Toplevel holding functions for yes/no buttons to close
    """

    def __init__(
        self,
        on_yes: _Callable[[None], None],
        on_no: _Callable[[None], None],
        on_close: _Optional[_Callable[[None], None]],
        resume_main: _Callable[[None], None],
    ):
        """
        :param on_yes: What to do when "Yes" button is pressed.
        :param on_no: What to do when "No" button is pressed.
        :param on_close: Do this regardless when closing (via yes/no/x) before
            resuming.
        """
        super().__init__()
        self.configure(bg=_GUI_BG)
        self._on_yes = on_yes
        self._on_no = on_no
        self._on_close = on_close
        self._resume_main = resume_main

    def destroy(self, pressed_yes: bool = False, pressed_no: bool = False):
        if pressed_yes:
            self._on_yes()
        if pressed_no:
            self._on_no()
        if self._on_close is not None:
            self._on_close()
        self._resume_main()
        super().destroy()


class _OkModal(object):
    """
    Message and OK button
    """

    def __init__(self, resume_main, msg: str, label_kwargs: _Optional[dict] = None):
        label_kwargs = {} if label_kwargs is None else label_kwargs

        self._root = _TopLevelWithOk((lambda: None), resume_main)
        self._root.configure(bg=_GUI_BG)
        self._text = _tk.Label(self._root, text=msg, **label_kwargs)
        self._text.configure(bg=_GUI_BG, fg=_GUI_TEXT, font=_BODY_FONT)
        self._text.pack(padx=20, pady=(18, 10))
        self._ok = _ttk.Button(
            self._root,
            text="OK",
            command=lambda: self._root.destroy(pressed_ok=True),
            style="Primary.TButton",
        )
        self._ok.pack(pady=(0, 18))
        self._root.protocol(
            "WM_DELETE_WINDOW", lambda: self._root.destroy(pressed_ok=False)
        )


class _YesNoModal(object):
    """
    Modal w/ yes/no buttons
    """

    def __init__(
        self,
        on_yes: _Callable[[None], None],
        on_no: _Callable[[None], None],
        resume_main,
        msg: str,
        on_close: _Optional[_Callable[[None], None]] = None,
        label_kwargs: _Optional[dict] = None,
    ):
        label_kwargs = {} if label_kwargs is None else label_kwargs
        self._root = _TopLevelWithYesNo(on_yes, on_no, on_close, resume_main)
        self._root.configure(bg=_GUI_BG)
        self._text = _tk.Label(self._root, text=msg, **label_kwargs)
        self._text.configure(bg=_GUI_BG, fg=_GUI_TEXT, font=_BODY_FONT)
        self._text.pack(padx=20, pady=(18, 10))
        self._buttons_frame = _tk.Frame(self._root, bg=_GUI_BG)
        self._buttons_frame.pack(pady=(0, 18))
        self._yes = _ttk.Button(
            self._buttons_frame,
            text="Yes",
            command=lambda: self._root.destroy(pressed_yes=True),
            style="Primary.TButton",
        )
        self._yes.pack(side=_tk.LEFT, padx=(0, 8))
        self._no = _ttk.Button(
            self._buttons_frame,
            text="No",
            command=lambda: self._root.destroy(pressed_no=True),
            style="Ghost.TButton",
        )
        self._no.pack(side=_tk.RIGHT)
        self._root.protocol("WM_DELETE_WINDOW", lambda: self._root.destroy())


class _FaultyClipsModal(object):
    def __init__(
        self,
        resume_main,
        parent: _tk.Misc,
        msg: str,
        *,
        failed_count: int,
        good_count: int,
        can_ignore: bool,
        on_exclude: _Optional[_Callable[[], None]],
        on_ignore: _Optional[_Callable[[], None]],
    ):
        self._resume_main = resume_main
        self._on_exclude = on_exclude
        self._on_ignore = on_ignore
        self._closed = False

        self._root = _tk.Toplevel(parent)
        self._root.title("Faulty Clips Detected")
        _apply_gui_theme(self._root)
        _style_toplevel(
            self._root,
            width=680,
            height=460,
            min_width=520,
            min_height=360,
        )
        self._root.transient(parent)

        body = _tk.Frame(self._root, bg=_GUI_BG)
        body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(3, weight=1)

        _ttk.Label(
            body,
            text="Faulty Clips Detected",
            style="Display.TLabel",
        ).grid(row=0, column=0, sticky="w")
        _eyebrow(
            body,
            f"{failed_count} flagged  |  {good_count} ready",
            prefix="",
        ).grid(row=1, column=0, sticky="w", pady=(2, 14))

        summary = (
            "Some output clips failed validation. You can keep only the clips "
            "that passed, disregard the warning and use every clip, or cancel."
        )
        if not can_ignore:
            summary = (
                "Some output clips failed critical validation. Critical failures "
                "cannot be forced into training, but clips that passed can still "
                "be used."
            )
        _tk.Label(
            body,
            text=summary,
            bg=_GUI_BG,
            fg=_GUI_TEXT,
            font=_FT_BODY,
            justify="left",
            wraplength=620,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 12))

        details_border = _tk.Frame(body, bg=_GUI_BORDER)
        details_border.grid(row=3, column=0, sticky="nsew", pady=(0, 14))
        details_frame = _tk.Frame(details_border, bg=_GUI_SURFACE)
        details_frame.pack(fill=_tk.BOTH, expand=True, padx=1, pady=1)
        details = _tk.Text(
            details_frame,
            bg=_GUI_SURFACE,
            fg=_GUI_TEXT,
            insertbackground=_GUI_ACCENT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=_FT_BODY,
            wrap="word",
            padx=12,
            pady=10,
            height=8,
        )
        details.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True)
        scrollbar = _FlatVerticalScrollbar(details_frame, command=details.yview)
        scrollbar.pack(side=_tk.RIGHT, fill=_tk.Y, pady=8, padx=(0, 8))
        details.configure(yscrollcommand=scrollbar.set)
        details.insert("1.0", msg)
        details.configure(state="disabled", cursor="arrow")

        action_frame = _tk.Frame(body, bg=_GUI_BG)
        action_frame.grid(row=4, column=0, sticky="ew")

        _ttk.Button(
            action_frame,
            text="Cancel",
            command=self._cancel,
            style="Ghost.TButton",
            width=16,
        ).pack(side=_tk.LEFT, padx=(0, 8))

        ignore_button = _ttk.Button(
            action_frame,
            text="Disregard This Info",
            command=self._ignore,
            style="Ghost.TButton",
            width=22,
        )
        ignore_button.pack(side=_tk.LEFT)
        if not can_ignore:
            ignore_button.configure(state=_tk.DISABLED)

        exclude_button = _ttk.Button(
            action_frame,
            text="Exclude Faulty Clips",
            command=self._exclude,
            style="Primary.TButton",
            width=22,
        )
        exclude_button.pack(side=_tk.RIGHT)
        if good_count == 0 or on_exclude is None:
            exclude_button.configure(state=_tk.DISABLED)

        self._root.protocol("WM_DELETE_WINDOW", self._cancel)
        self._root.grab_set()
        exclude_button.focus_set()

    def _cancel(self):
        self._close(action=None)

    def _exclude(self):
        self._close(action="exclude")

    def _ignore(self):
        self._close(action="ignore")

    def _close(self, action):
        if self._closed:
            return
        self._closed = True
        callback = None
        if action == "exclude":
            callback = self._on_exclude
        elif action == "ignore":
            callback = self._on_ignore
        try:
            self._root.grab_release()
        except _tk.TclError:
            pass
        try:
            if callback is not None:
                callback()
        finally:
            self._resume_main()
            self._root.destroy()


@_dataclass
class _LightningAnalysisRow(object):
    input_file: str
    output_file: str
    file_name: str
    stage: str
    architecture: str
    epochs: str
    best_epoch: str
    training_time: str
    tpe: str
    esr: str
    mrstft: str
    mse: str
    capture_efficiency: str
    quality_weighted_efficiency: str
    learning_rate: str
    decay: str
    scheduler: str
    run_path: str


class _LightningAnalysisModal(object):
    _COLUMNS = (
        ("input_file", "Input"),
        ("output_file", "Output"),
        ("stage", "Stage"),
        ("architecture", "Arch"),
        ("epochs", "Epochs"),
        ("best_epoch", "Best Epoch"),
        ("training_time", "Train Time"),
        ("tpe", "TPE"),
        ("esr", "ESR"),
        ("mrstft", "MRSTFT"),
        ("mse", "MSE"),
        ("capture_efficiency", "Capture Eff."),
        ("quality_weighted_efficiency", "Quality Eff."),
        ("learning_rate", "LR"),
        ("decay", "Decay"),
        ("scheduler", "Scheduler"),
    )
    _SORTABLE_COLUMNS = {
        "training_time",
        "tpe",
        "esr",
        "mrstft",
        "mse",
        "capture_efficiency",
        "quality_weighted_efficiency",
    }

    def __init__(self, resume_main, rows: _Sequence[_LightningAnalysisRow]):
        self._resume_main = resume_main
        self._rows = list(rows)
        self._sort_states = {key: 0 for key, _ in self._COLUMNS}
        self._root = _tk.Toplevel()
        self._root.title("Lightning Folder Analysis")
        self._root.geometry("2160x600")
        self._root.configure(bg=_GUI_BG)

        frame = _tk.Frame(self._root, bg=_GUI_BG)
        frame.pack(fill=_tk.BOTH, expand=True, padx=12, pady=(12, 0))

        self._tree = _ttk.Treeview(
            frame,
            columns=[key for key, _ in self._COLUMNS],
            show="headings",
        )
        for key, title in self._COLUMNS:
            width = 170 if key in ("file_name", "run_path") else 95
            heading_command = (
                (lambda column_key=key: self._on_heading_click(column_key))
                if key in self._SORTABLE_COLUMNS
                else None
            )
            if heading_command is None:
                self._tree.heading(key, text=title)
            else:
                self._tree.heading(key, text=title, command=heading_command)
            self._tree.column(key, width=width, anchor="center")
        self._tree.column("input_file", width=260, anchor="w")
        self._tree.column("output_file", width=450, anchor="w")
        self._tree.column("stage", width=110, anchor="w")
        self._tree.column("tpe", width=90, anchor="e")
        self._tree.column("capture_efficiency", width=115, anchor="e")
        self._tree.column("quality_weighted_efficiency", width=115, anchor="e")
        self._tree.column("scheduler", width=150, anchor="w")

        yscroll = _ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        xscroll = _ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self._populate_rows(self._rows)

        button_frame = _tk.Frame(self._root, bg=_GUI_BG)
        button_frame.pack(fill=_tk.X, pady=10)
        close_button = _ttk.Button(
            button_frame,
            text="Close",
            command=self._close,
            style="Primary.TButton",
        )
        close_button.pack()
        self._root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        self._resume_main()
        self._root.destroy()

    def _populate_rows(self, rows: _Sequence[_LightningAnalysisRow]):
        self._tree.delete(*self._tree.get_children())
        for row in rows:
            self._tree.insert(
                "",
                _tk.END,
                values=[getattr(row, key) for key, _ in self._COLUMNS],
            )

        if len(rows) == 0:
            self._tree.insert(
                "",
                _tk.END,
                values=(
                    "No compatible Lightning runs found.",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ),
            )

    def _on_heading_click(self, column_key: str):
        current_state = self._sort_states.get(column_key, 0)
        next_state = (current_state + 1) % 3
        self._sort_states = {key: 0 for key, _ in self._COLUMNS}
        self._sort_states[column_key] = next_state
        self._refresh_headings()

        if next_state == 0:
            self._populate_rows(self._rows)
            return

        sorted_rows = sorted(
            self._rows,
            key=lambda row: self._sort_value(
                column_key, getattr(row, column_key), descending=(next_state == 2)
            ),
        )
        self._populate_rows(sorted_rows)

    def _refresh_headings(self):
        arrow_map = {0: "", 1: " ↑", 2: " ↓"}
        for key, title in self._COLUMNS:
            suffix = arrow_map.get(self._sort_states.get(key, 0), "")
            self._tree.heading(key, text=f"{title}{suffix}")

    @classmethod
    def _sort_value(cls, column_key: str, value: str, descending: bool = False):
        if column_key == "training_time":
            seconds = cls._duration_to_seconds(value)
            if seconds is None:
                return (1, float("inf"))
            return (0, -seconds if descending else seconds)
        numeric_value = cls._to_float(value)
        if numeric_value is None:
            return (1, float("inf"))
        return (0, -numeric_value if descending else numeric_value)

    @staticmethod
    def _duration_to_seconds(value: str) -> _Optional[int]:
        if not value:
            return None
        parts = value.split(":")
        if len(parts) != 3:
            return None
        try:
            hours, minutes, seconds = [int(part) for part in parts]
        except ValueError:
            return None
        return (hours * 3600) + (minutes * 60) + seconds

    @staticmethod
    def _to_float(value: str) -> _Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class _UpdateAvailableModal(object):
    """
    Modal shown when a new version is available. Message, "Do not show this again"
    checkbox, and Close button. Does not offer an in-GUI upgrade; instructs user to
    run pip install --upgrade.
    """

    def __init__(self, resume_main: _Callable[[], None], version: str):
        self._root = _tk.Toplevel()
        self._root.title("Update available")
        self._root.configure(bg=_GUI_BG)
        msg = (
            f"neural-amp-modeler v{version} is now available. To upgrade, run "
            "pip install --upgrade neural-amp-modeler in your terminal."
        )
        self._label = _tk.Label(
            self._root,
            text=msg,
            justify=_tk.LEFT,
            wraplength=400,
            bg=_GUI_BG,
            fg=_GUI_TEXT,
            font=_BODY_FONT,
        )
        self._label.pack(padx=20, pady=(18, 10))
        self._never_show_var = _tk.BooleanVar(value=False)
        self._checkbox = _FlatCheckbutton(
            self._root,
            text="Do not show this again",
            variable=self._never_show_var,
        )
        self._checkbox.pack(anchor="w", padx=20, pady=5)
        self._close_btn = _ttk.Button(
            self._root,
            text="Close",
            command=self._on_close,
            style="Primary.TButton",
        )
        self._close_btn.pack(pady=(8, 18))
        self._resume_main = resume_main

    def _on_close(self):
        if self._never_show_var.get():
            _settings.set_update_settings(never_show_again=True)
        self._resume_main()
        self._root.destroy()


class _GUIWidgets(_Enum):
    INPUT_PATH = "input_path"
    OUTPUT_PATH = "output_path"
    TRAINING_DESTINATION = "training_destination"
    RESUME_CHECKPOINT = "resume_checkpoint"
    METADATA = "metadata"
    ADVANCED_OPTIONS = "advanced_options"
    TRAIN = "train"
    ADD_TO_SCHEDULE = "add_to_schedule"
    START_SCHEDULE = "start_schedule"
    REMOVE_SCHEDULED = "remove_scheduled"
    CLEAR_SCHEDULED = "clear_scheduled"


_SCHEDULE_STATUS_QUEUED = "queued"
_SCHEDULE_STATUS_RUNNING = "running"
_SCHEDULE_STATUS_FINISHED = "finished"
_SCHEDULE_STATUS_FAILED = "failed"
_SCHEDULE_STATUS_ABORTED = "aborted"


@_dataclass
class Checkbox(object):
    variable: _tk.BooleanVar
    check_button: _tk.Checkbutton


@_dataclass
class _TrainingJob(object):
    input_path: str
    output_paths: _Tuple[str, ...]
    training_destination: str
    advanced_options: AdvancedOptions
    checkbox_values: _Dict[_CheckboxKeys, bool]
    user_metadata: _UserMetadata
    user_metadata_flag: bool
    resume_checkpoint_path: _Optional[_ResumeCheckpointSelection] = None
    ignore_checks: bool = False
    export_suffixes: _Optional[_Dict[str, str]] = None
    schedule_status: str = _SCHEDULE_STATUS_QUEUED
    schedule_result: str = ""


_ARCHITECTURE_EXPORT_SUFFIX_LABELS = {
    _core.Architecture.LSTM_COMPRESSOR_HQ_48X3: "LSTM-HQ-48x3",
    _core.Architecture.LSTM_COMPRESSOR_LIGHT_30X3: "LSTM-Light-30x3",
    _core.Architecture.LSTM_UHQ: "LSTM-UHQ",
    _core.Architecture.LSTM_TONEX_LIKE_16: "LSTM-TONEX16",
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048: "CCLSTM-128-16-2048",
    _core.Architecture.CAUSAL_CONV_LSTM_TONEX_HQ: "Tonex-HQ",
    _core.Architecture.A2_FULL_LITE: "A2-Full-Lite",
    _core.Architecture.A2_COMPLEX_LITE: "A2-Complex-Lite",
    _core.Architecture.A2_COMPLEX_REVYLO: "A2-Complex-RevYLo",
    _core.Architecture.A2_COMPLEX_NANO64X4: "A2-Complex-Nano64x4",
    _core.Architecture.A2_COMPLEX_NANO125X3: "A2-Complex-Nano125x3",
    _core.Architecture.A2_DOUBLE_LITE: "A2-Double-Lite",
    _core.Architecture.A2_XDOUBLE_LITE: "A2-xDouble-Lite",
    _core.Architecture.A2_DOUBLEPLUS_LITE: "A2-DoublePlus-Lite",
    _core.Architecture.DOUBLEPLUS: "DoublePlus",
    _core.Architecture.REVYLO: "revylo",
}


def _filename_suffix_token(value: str) -> str:
    value = _re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-")
    return value or "setting"


def _number_suffix_token(value: float) -> str:
    return _fmt_float(float(value)).replace("-", "m").replace(".", "p")


def _architecture_suffix_label(architecture: _core.Architecture) -> str:
    architecture = _core.Architecture(architecture)
    return _ARCHITECTURE_EXPORT_SUFFIX_LABELS.get(
        architecture, _filename_suffix_token(architecture.value)
    )


def _advanced_options_export_suffix(options: AdvancedOptions) -> str:
    parts = [
        _architecture_suffix_label(options.architecture),
        f"{int(options.num_epochs)}e",
        f"lr{_number_suffix_token(options.lr)}",
        f"decay{_number_suffix_token(options.lr_decay)}",
    ]
    if options.stage_mode == _core.TrainingStageMode.TWO_STAGE:
        parts.extend(
            [
                "s2",
                f"{int(options.stage2_epochs)}e",
                f"lr{_number_suffix_token(options.stage2_lr)}",
                f"decay{_number_suffix_token(options.stage2_lr_decay)}",
            ]
        )
    return "__" + "_".join(parts)


def _export_basename_for_output_path(output_path: str) -> str:
    filename = _re.split(r"[\\/]", str(output_path))[-1]
    return _re.sub(r"\.wav$", "", filename, flags=_re.IGNORECASE)


def _normalized_training_destination(path: str) -> str:
    try:
        return str(_Path(path).expanduser().resolve(strict=False)).casefold()
    except (OSError, RuntimeError, ValueError):
        return str(path).replace("\\", "/").rstrip("/").casefold()


def _job_export_basename(job: _TrainingJob, output_path: str) -> str:
    basename = _export_basename_for_output_path(output_path)
    suffixes = job.export_suffixes or {}
    return basename + suffixes.get(str(output_path), "")


def _recompute_export_suffixes(
    jobs: _Sequence[_TrainingJob],
    locked_jobs: _Sequence[_TrainingJob] = (),
):
    mutable_jobs = list(jobs)
    mutable_job_ids = {id(job) for job in mutable_jobs}
    for job in mutable_jobs:
        job.export_suffixes = {}

    export_groups = {}
    for job in list(locked_jobs) + mutable_jobs:
        destination = _normalized_training_destination(job.training_destination)
        for output_path in job.output_paths:
            basename = _export_basename_for_output_path(output_path)
            key = (destination, basename.casefold())
            export_groups.setdefault(key, []).append((job, str(output_path)))

    for group in export_groups.values():
        if len(group) <= 1:
            continue
        used_suffixes = set()
        for job, output_path in group:
            suffix = _advanced_options_export_suffix(job.advanced_options)
            if id(job) not in mutable_job_ids:
                locked_suffix = (job.export_suffixes or {}).get(output_path, "")
                if locked_suffix:
                    used_suffixes.add(locked_suffix)
                continue
            if suffix in used_suffixes:
                run_number = 2
                while f"{suffix}_run{run_number}" in used_suffixes:
                    run_number += 1
                suffix = f"{suffix}_run{run_number}"
            used_suffixes.add(suffix)
            if id(job) in mutable_job_ids:
                if job.export_suffixes is None:
                    job.export_suffixes = {}
                job.export_suffixes[output_path] = suffix


class GUI(object):
    def __init__(self):
        self._root = _tk.Tk()
        self._theme_name = _load_theme_choice()
        self._P = dict(_THEMES[self._theme_name])
        _apply_gui_theme(self._root, self._P)
        self._root.title("NAM Trainer - V1.0.1")
        _position_window_center_screen(self._root, 1040, 900)
        self._root.minsize(940, 860)
        self._widgets = {}
        self.user_metadata, self.user_metadata_flag = _load_user_metadata_from_settings()
        self._saved_checkbox_values = _load_checkbox_values_from_settings()
        self.advanced_options = _load_advanced_options_from_settings()
        self._schedule_lock = _threading.Lock()
        self._scheduled_jobs: _List[_TrainingJob] = []
        self._training_active = False
        self._current_training_label = ""
        self._current_training_job: _Optional[_TrainingJob] = None
        self._current_training_output_path = ""
        self._current_training_started_at = 0.0
        self._current_training_stage_label = "core"
        self._eta_seconds_per_epoch: _Optional[float] = None
        self._active_remaining_epoch_units = 0
        self._scheduler_metrics_state = None
        self._progress_poll_after_id = None
        self._training_thread: _Optional[_threading.Thread] = None

        self._frame_main = _ttk.Frame(
            self._root, style="TFrame", padding=(28, 16, 28, 22)
        )
        self._frame_main.pack(fill=_tk.BOTH, expand=True)

        self._frame_header = _tk.Frame(self._frame_main, bg=_GUI_BG)
        self._frame_header.pack(fill=_tk.X, pady=(0, 18))
        self._header_left = _tk.Frame(self._frame_header, bg=_GUI_BG)
        self._header_left.pack(side=_tk.LEFT)
        self._title_label = _tk.Label(
            self._header_left,
            text="NAM Trainer",
            bg=_GUI_BG,
            fg=_GUI_TEXT,
            font=_HEADER_TITLE_FONT,
        )
        self._title_label.pack(side=_tk.LEFT)
        self._header_title_rule = _vrule(self._header_left, height=28)
        self._header_title_rule.pack(side=_tk.LEFT, padx=12)
        self._header_subtitle_label = _tk.Label(
            self._header_left,
            text="  ".join(list("NEURAL AMP MODELER")),
            bg=_GUI_BG,
            fg=_GUI_FAINT,
            font=_EYEBROW_FONT,
            bd=0,
            padx=0,
            pady=0,
        )
        self._header_subtitle_label.pack(side=_tk.LEFT)
        self._header_right = _tk.Frame(self._frame_header, bg=_GUI_BG)
        self._header_right.pack(side=_tk.RIGHT)
        self._header_status_dot = _status_dot(self._header_right, "FAINT")
        self._header_status_dot.pack(side=_tk.LEFT, padx=(0, 6))
        self._header_status_label = _eyebrow(self._header_right, "Idle", prefix="")
        self._header_status_label.pack(
            side=_tk.LEFT, padx=(0, 14)
        )
        _vrule(self._header_right, height=18).pack(side=_tk.LEFT, padx=(0, 14))
        _eyebrow(self._header_right, "Theme", prefix="").pack(
            side=_tk.LEFT, padx=(0, 10)
        )
        self._theme_chip_wrap = _tk.Frame(
            self._header_right, bg=self._P["BORDER"], bd=0, highlightthickness=0
        )
        self._theme_chip_wrap.pack(side=_tk.LEFT, padx=(0, 8))
        self._theme_chip = _tk.Canvas(
            self._theme_chip_wrap,
            bg=self._P["ACCENT"],
            width=12,
            height=12,
            bd=0,
            highlightthickness=0,
        )
        self._theme_chip.pack(padx=1, pady=1)
        self._theme_var = _tk.StringVar(value=self._theme_name)
        self._theme_dropdown = _ttk.Combobox(
            self._header_right,
            textvariable=self._theme_var,
            values=list(_THEMES.keys()),
            state="readonly",
            width=12,
            font=_FT_BTN,
            style="TCombobox",
        )
        self._theme_dropdown.pack(side=_tk.LEFT)
        self._theme_dropdown.bind("<<ComboboxSelected>>", self._on_theme_change)

        _eyebrow(self._frame_main, "Files").pack(anchor="w", pady=(0, 6))
        self._frame_files = _tk.Frame(self._frame_main, bg=_GUI_BG, highlightthickness=0)
        self._frame_files.pack(fill=_tk.X, pady=(0, 4))
        self._frame_files.columnconfigure(0, minsize=218)
        self._frame_files.columnconfigure(1, weight=1)
        self._frame_files.columnconfigure(2, minsize=218)

        self._widgets[_GUIWidgets.INPUT_PATH] = _PathButton(
            self._frame_files,
            "Input Audio",
            f"Select input (DI) file (e.g. {_LATEST_VERSION.name})",
            _PathType.FILE,
            _settings.PathKey.INPUT_FILE,
            hooks=[self._check_button_states],
            default=_load_path_selection(_settings.PathKey.INPUT_FILE, _PathType.FILE),
            grid_row=0,
        )
        self._widgets[_GUIWidgets.OUTPUT_PATH] = _PathButton(
            self._frame_files,
            "Output Audio",
            "Select output (reamped) file - (Choose MULTIPLE FILES to enable BATCH TRAINING)",
            _PathType.MULTIFILE,
            _settings.PathKey.OUTPUT_FILE,
            hooks=[self._check_button_states],
            default=_load_path_selection(_settings.PathKey.OUTPUT_FILE, _PathType.MULTIFILE),
            grid_row=1,
        )
        self._widgets[_GUIWidgets.TRAINING_DESTINATION] = _PathButton(
            self._frame_files,
            "Train Destination",
            "Select training output directory",
            _PathType.DIRECTORY,
            _settings.PathKey.TRAINING_DESTINATION,
            hooks=[self._check_button_states],
            default=_load_path_selection(_settings.PathKey.TRAINING_DESTINATION, _PathType.DIRECTORY),
            grid_row=2,
        )

        self._widgets["metadata"] = _ttk.Button(
            self._frame_files,
            text="Metadata",
            command=self._open_metadata,
            style="Ink.TButton",
        )
        self._widgets["metadata"].grid(
            row=3, column=0, sticky="ew", padx=(0, 18), pady=4
        )
        self._metadata_path_frame = _tk.Frame(self._frame_files, bg=_GUI_BG)
        self._metadata_path_frame.grid(row=3, column=1, sticky="w")
        self._metadata_prefix = _ttk.Label(
            self._metadata_path_frame,
            text="›",
            background=_GUI_BG,
            foreground=_GUI_FAINT,
            font=_FT_LABEL,
        )
        self._metadata_prefix.pack(side=_tk.LEFT, padx=(0, 8))
        self._metadata_label = _ttk.Label(
            self._metadata_path_frame,
            text="Optional model and gear metadata",
            background=_GUI_BG,
            foreground=_GUI_FAINT,
            font=_FT_MONO_S,
        )
        self._metadata_label.pack(side=_tk.LEFT)

        self._resume_checkpoint_path: _Optional[_ResumeCheckpointSelection] = None
        self._resume_checkpoint_marquee_after_id = None
        self._resume_checkpoint_marquee_offset = 0
        self._resume_checkpoint_marquee_direction = 1
        self._widgets[_GUIWidgets.RESUME_CHECKPOINT] = _ttk.Button(
            self._frame_files,
            text=_RESUME_CHECKPOINT_BUTTON_TEXT,
            command=self._choose_resume_checkpoint,
            style="Ghost.TButton",
            width=_RESUME_CHECKPOINT_BUTTON_CHARS,
        )
        self._widgets[_GUIWidgets.RESUME_CHECKPOINT].grid(
            row=0, column=2, sticky="ew", padx=(18, 0), pady=4
        )
        self._widgets["analyze_lightning"] = _ttk.Button(
            self._frame_files,
            text="Analyze Lightning Folders",
            command=self._analyze_lightning_folders,
            style="Ghost.TButton",
        )
        self._widgets["analyze_lightning"].grid(
            row=1, column=2, sticky="ew", padx=(18, 0), pady=4
        )

        self._get_additional_options_frame()

        self._widgets[_GUIWidgets.ADVANCED_OPTIONS] = _ttk.Button(
            self._frame_files,
            text="Advanced Options",
            command=self._open_advanced_options,
            style="Ghost.TButton",
        )
        self._widgets[_GUIWidgets.ADVANCED_OPTIONS].grid(
            row=2, column=2, sticky="ew", padx=(18, 0), pady=4
        )
        self._widgets[_GUIWidgets.TRAIN] = _ttk.Button(
            self._frame_files,
            text="Train",
            command=self._train,
            style="Primary.TButton",
        )
        self._widgets[_GUIWidgets.TRAIN].grid(
            row=3, column=2, sticky="ew", padx=(18, 0), pady=4
        )

        self._build_scheduler_section()

        self._show_update_modal_if_update_available()

        self._check_button_states()
        self._refresh_header_status()
        self._persist_gui_state()

    def _set_theme(self, name: str):
        if name not in _THEMES:
            return
        self._theme_name = name
        self._P.clear()
        self._P.update(_THEMES[name])
        _save_theme_choice(name)
        self._root.configure(bg=self._P["BG"])
        _apply_gui_theme(self._root, self._P)
        self._repaint_theme_widgets()
        self._configure_scheduler_progress_info_tags()
        self._root.focus_set()
        self._root.update_idletasks()

    def _on_theme_change(self, event):
        event.widget.selection_clear()
        self._root.focus_set()
        self._set_theme(self._theme_var.get())

    @staticmethod
    def _header_status_eyebrow_text(text: str) -> str:
        return "  ".join(list(text.upper()))

    def _refresh_header_status(self):
        if not hasattr(self, "_header_status_label"):
            return
        with self._schedule_lock:
            training_active = self._training_active
            stage_label = self._current_training_stage_label
        status_text = f"Running \u00b7 {stage_label}" if training_active else "Idle"
        self._header_status_label.configure(
            text=self._header_status_eyebrow_text(status_text),
            background=self._P["BG"],
            foreground=self._P["FAINT"],
        )
        if hasattr(self, "_header_status_dot"):
            dot_color = self._P["M_BEST"] if training_active else self._P["FAINT"]
            try:
                self._header_status_dot.configure(bg=self._P["BG"])
                self._header_status_dot.itemconfigure(
                    "dot", fill=dot_color, outline=dot_color
                )
            except _tk.TclError:
                pass

    def _repaint_theme_widgets(self):
        P = self._P
        if hasattr(self, "_theme_chip_wrap"):
            self._theme_chip_wrap.configure(bg=P["BORDER"])
        if hasattr(self, "_theme_chip"):
            self._theme_chip.configure(bg=P["ACCENT"])
        if hasattr(self, "_title_label"):
            self._title_label.configure(bg=P["BG"], fg=P["TEXT"])
        if hasattr(self, "_header_subtitle_label"):
            self._header_subtitle_label.configure(bg=P["BG"], fg=P["FAINT"])
        self._refresh_header_status()
        if hasattr(self, "_scheduler_list_border"):
            self._scheduler_list_border.configure(bg=P["BORDER"])
        if hasattr(self, "_scheduler_list_frame"):
            self._scheduler_list_frame.configure(bg=P["SURFACE"])
        if hasattr(self, "_schedule_tree"):
            self._schedule_tree.tag_configure("normal", foreground=P["TEXT"])
            self._schedule_tree.tag_configure("queued", foreground=P["TEXT"])
            self._schedule_tree.tag_configure("running", foreground=P["ACCENT"])
            self._schedule_tree.tag_configure("finished", foreground=P["M_BEST"])
            self._schedule_tree.tag_configure("failed", foreground=P["M_LOSS"])
            self._schedule_tree.tag_configure("aborted", foreground=P["FAINT"])
            self._schedule_tree.tag_configure("empty", foreground=P["FAINT"])
        if hasattr(self, "_schedule_scrollbar"):
            self._schedule_scrollbar.recolor()
        if hasattr(self, "_checkboxes"):
            for checkbox in self._checkboxes.values():
                if hasattr(checkbox.check_button, "recolor"):
                    checkbox.check_button.recolor()
        if hasattr(self, "_scheduler_status"):
            self._scheduler_status.configure(bg=P["BG"], fg=P["MUTED"])
        if hasattr(self, "_scheduler_queued"):
            self._scheduler_queued.configure(background=P["BG"], foreground=P["MUTED"])
        if hasattr(self, "_scheduler_progress_track"):
            self._scheduler_progress_track.configure(bg=P["PROG_TRACK"])
        if hasattr(self, "_scheduler_progress_fill"):
            self._scheduler_progress_fill.configure(bg=P["PROG_FILL"])
        if hasattr(self, "_scheduler_progress_info"):
            self._scheduler_progress_info.configure(bg=P["BG"], fg=P["MUTED"])
            if getattr(self, "_scheduler_metrics_state", None) is not None:
                _render_metrics(self._scheduler_progress_info, P, self._scheduler_metrics_state)
        for key in (
            _GUIWidgets.INPUT_PATH,
            _GUIWidgets.OUTPUT_PATH,
            _GUIWidgets.TRAINING_DESTINATION,
        ):
            widget = self._widgets.get(key)
            if hasattr(widget, "recolor"):
                widget.recolor()
        if hasattr(self, "_metadata_path_frame"):
            self._metadata_path_frame.configure(bg=P["BG"])
        if hasattr(self, "_metadata_prefix"):
            self._metadata_prefix.configure(background=P["BG"], foreground=P["FAINT"])
        if hasattr(self, "_metadata_label"):
            self._metadata_label.configure(background=P["BG"], foreground=P["FAINT"])

        def repaint(widget):
            try:
                cls = widget.winfo_class()
            except _tk.TclError:
                return
            if cls in ("Frame", "Toplevel", "Tk"):
                try:
                    widget.configure(bg=P["BG"])
                except _tk.TclError:
                    pass
            elif cls == "Label":
                try:
                    config = {"bg": P["BG"]}
                    if widget.cget("text") == "\u25c6":
                        config["fg"] = P["FAINT"]
                    widget.configure(**config)
                except _tk.TclError:
                    pass
            elif cls == "TLabel":
                try:
                    style = widget.cget("style")
                    text = widget.cget("text")
                    foreground = None
                    if style == "Display.TLabel":
                        foreground = P["TEXT"]
                    elif style == "Muted.TLabel":
                        foreground = P["MUTED"]
                    elif style == "Faint.TLabel":
                        foreground = P["FAINT"]
                    elif style == "Mono.TLabel":
                        foreground = P["MUTED"]
                    elif text.startswith("◆") or "  " in text:
                        foreground = P["FAINT"]
                    config = {"background": P["BG"]}
                    if foreground is not None:
                        config["foreground"] = foreground
                    widget.configure(**config)
                except _tk.TclError:
                    pass
            elif cls == "Text":
                try:
                    widget.configure(bg=P["BG"], fg=P["MUTED"], insertbackground=P["ACCENT"])
                except _tk.TclError:
                    pass
            for child in widget.winfo_children():
                repaint(child)

        repaint(self._root)
        if hasattr(self, "_header_title_rule"):
            self._header_title_rule.configure(bg=P["BORDER"])
        if hasattr(self, "_theme_chip_wrap"):
            self._theme_chip_wrap.configure(bg=P["BORDER"])
        if hasattr(self, "_theme_chip"):
            self._theme_chip.configure(bg=P["ACCENT"])
        if hasattr(self, "_scheduler_list_border"):
            self._scheduler_list_border.configure(bg=P["BORDER"])
        if hasattr(self, "_scheduler_list_frame"):
            self._scheduler_list_frame.configure(bg=P["SURFACE"])
        if hasattr(self, "_scheduler_progress_track"):
            self._scheduler_progress_track.configure(bg=P["PROG_TRACK"])
        if hasattr(self, "_scheduler_progress_fill"):
            self._scheduler_progress_fill.configure(bg=P["PROG_FILL"])
        if hasattr(self, "_schedule_scrollbar"):
            self._schedule_scrollbar.recolor()

    def _build_scheduler_section(self):
        self._frame_scheduler = _tk.Frame(self._frame_main, bg=_GUI_BG)
        self._frame_scheduler.pack(fill=_tk.BOTH, expand=True, pady=(0, _ROW_PADY))

        self._scheduler_header = _tk.Frame(self._frame_scheduler, bg=_GUI_BG)
        self._scheduler_header.pack(fill=_tk.X, pady=(0, 8))
        _eyebrow(self._scheduler_header, "Scheduler").pack(side=_tk.LEFT, padx=(0, 14))
        self._scheduler_status_var = _tk.StringVar(
            master=self._frame_scheduler, value="Idle    Queued: 0"
        )
        self._scheduler_status = _tk.Label(
            self._scheduler_header,
            textvariable=self._scheduler_status_var,
            anchor="w",
            justify="left",
            wraplength=760,
            bg=_GUI_BG,
            fg=_GUI_MUTED,
            font=_MONO_SMALL_FONT,
        )
        self._scheduler_status.pack(side=_tk.LEFT, fill=_tk.X, expand=True)
        self._scheduler_queued_var = _tk.StringVar(
            master=self._frame_scheduler, value="Queued  0"
        )
        self._scheduler_queued = _ttk.Label(
            self._scheduler_header,
            textvariable=self._scheduler_queued_var,
            background=_GUI_BG,
            foreground=_GUI_MUTED,
            font=_FT_MONO_S,
        )
        self._scheduler_queued.pack(side=_tk.RIGHT)

        self._scheduler_list_border = _tk.Frame(self._frame_scheduler, bg=_GUI_BORDER)
        self._scheduler_list_border.pack(fill=_tk.BOTH, expand=True, pady=(0, 10))
        self._scheduler_list_frame = _tk.Frame(self._scheduler_list_border, bg=_GUI_SURFACE)
        self._scheduler_list_frame.pack(fill=_tk.BOTH, expand=True, padx=1, pady=1)
        self._schedule_tree = _ttk.Treeview(
            self._scheduler_list_frame,
            height=5,
            columns=("run", "eta"),
            show="headings",
            selectmode="browse",
            style="Treeview",
        )
        self._schedule_tree.heading("run", text="Run")
        self._schedule_tree.heading("eta", text="ETA / Final ESR")
        self._schedule_tree.column("run", anchor="w", stretch=True, width=680)
        self._schedule_tree.column("eta", anchor="e", stretch=False, width=220)
        self._schedule_tree.tag_configure("normal", foreground=_GUI_TEXT)
        self._schedule_tree.tag_configure("queued", foreground=_GUI_TEXT)
        self._schedule_tree.tag_configure("running", foreground=_GUI_ACCENT)
        self._schedule_tree.tag_configure("finished", foreground=_GUI_METRIC_BEST)
        self._schedule_tree.tag_configure("failed", foreground=_GUI_METRIC_LOSS)
        self._schedule_tree.tag_configure("aborted", foreground=_GUI_FAINT)
        self._schedule_tree.tag_configure("empty", foreground=_GUI_FAINT)
        self._schedule_tree.pack(
            side=_tk.LEFT, fill=_tk.BOTH, expand=True, padx=(10, 0), pady=8
        )
        self._schedule_scrollbar = _FlatVerticalScrollbar(
            self._scheduler_list_frame,
            command=self._schedule_tree.yview,
        )
        self._schedule_scrollbar.pack(side=_tk.RIGHT, fill=_tk.Y, pady=8, padx=(0, 8))
        self._schedule_tree.configure(yscrollcommand=self._schedule_scrollbar.set)

        self._scheduler_buttons_frame = _tk.Frame(self._frame_scheduler, bg=_GUI_BG)
        self._scheduler_buttons_frame.pack(fill=_tk.X, pady=(10, 0))
        self._scheduler_buttons_frame.grid_columnconfigure(0, weight=1)
        self._scheduler_buttons_frame.grid_columnconfigure(1, weight=1)

        self._widgets[_GUIWidgets.ADD_TO_SCHEDULE] = _ttk.Button(
            self._scheduler_buttons_frame,
            text="Add to Schedule",
            command=self._add_current_to_schedule,
            style="Subtle.TButton",
        )
        self._widgets[_GUIWidgets.ADD_TO_SCHEDULE].grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 5),
            pady=(0, 5),
        )

        self._widgets[_GUIWidgets.START_SCHEDULE] = _ttk.Button(
            self._scheduler_buttons_frame,
            text="Start Schedule",
            command=self._start_scheduled_jobs,
            style="Primary.TButton",
        )
        self._widgets[_GUIWidgets.START_SCHEDULE].grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(5, 0),
            pady=(0, 5),
        )

        self._widgets[_GUIWidgets.REMOVE_SCHEDULED] = _ttk.Button(
            self._scheduler_buttons_frame,
            text="Remove Selected",
            command=self._remove_selected_scheduled_job,
            style="Subtle.TButton",
        )
        self._widgets[_GUIWidgets.REMOVE_SCHEDULED].grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 5),
            pady=(5, 0),
        )

        self._widgets[_GUIWidgets.CLEAR_SCHEDULED] = _ttk.Button(
            self._scheduler_buttons_frame,
            text="Clear Schedule",
            command=self._clear_scheduled_jobs,
            style="Ghost.TButton",
        )
        self._widgets[_GUIWidgets.CLEAR_SCHEDULED].grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(5, 0),
            pady=(5, 0),
        )

        self._scheduler_progress_var = _tk.DoubleVar(
            master=self._frame_scheduler, value=0.0
        )
        self._scheduler_progress_track = _tk.Frame(
            self._frame_scheduler,
            bg=_GUI_PROGRESS_TRACK,
            height=6,
            bd=0,
            highlightthickness=0,
        )
        self._scheduler_progress_track.pack(fill=_tk.X, pady=(14, 8))
        self._scheduler_progress_track.pack_propagate(False)
        self._scheduler_progress_fill = _tk.Frame(
            self._scheduler_progress_track,
            bg=_GUI_PROGRESS_FILL,
            bd=0,
            highlightthickness=0,
        )
        self._scheduler_progress_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)
        self._scheduler_progress = self._scheduler_progress_track

        def sync_progress(*_args):
            value = max(0.0, min(100.0, self._scheduler_progress_var.get()))
            self._scheduler_progress_fill.place_configure(relwidth=value / 100.0)

        self._scheduler_progress_var.trace_add("write", sync_progress)
        self._scheduler_progress_track.bind("<Configure>", sync_progress)
        sync_progress()
        self._scheduler_progress_info_var = _tk.StringVar(
            master=self._frame_scheduler, value="No active run"
        )
        self._scheduler_progress_info = _make_metrics_widget(
            self._frame_scheduler, self._P
        )
        self._scheduler_progress_info.configure(takefocus=0)
        self._scheduler_progress_info.pack(fill=_tk.X)
        self._configure_scheduler_progress_info_tags()
        self._set_scheduler_progress_info("No active run")
        self._refresh_scheduler_widgets()

    def core_train_kwargs(self) -> _Dict[str, _Any]:
        """
        Get any additional kwargs to provide to `core.train`
        """
        return {
            "seed": 0,
        }

    def _persist_gui_state(self):
        _settings.set_advanced_options_settings(
            _serialize_advanced_options(self.advanced_options)
        )
        _settings.set_metadata_settings(
            _serialize_user_metadata(self.user_metadata, self.user_metadata_flag)
        )
        if hasattr(self, "_checkboxes"):
            _settings.set_checkbox_settings(
                {
                    key.value: checkbox.variable.get()
                    for key, checkbox in self._checkboxes.items()
                }
            )

    def get_mrstft_fit(self) -> bool:
        """
        Use a pre-emphasized multi-resolution shot-time Fourier transform loss during
        training.

        This improves agreement in the high frequencies, usually with a minimal loss in
        ESR.
        """
        # Leave this as a public method to anticipate an extension to make it
        # changeable.
        return True

    def _check_button_states(self):
        """
        Determine if any buttons should be disabled
        """
        # Train button is disabled unless all paths are set
        paths_ready = not any(
            pb.val is None
            for pb in (
                self._widgets[_GUIWidgets.INPUT_PATH],
                self._widgets[_GUIWidgets.OUTPUT_PATH],
                self._widgets[_GUIWidgets.TRAINING_DESTINATION],
            )
        )
        self._widgets[_GUIWidgets.TRAIN]["state"] = (
            _tk.NORMAL if paths_ready and not self._training_active else _tk.DISABLED
        )
        if _GUIWidgets.ADD_TO_SCHEDULE in self._widgets:
            self._widgets[_GUIWidgets.ADD_TO_SCHEDULE]["state"] = (
                _tk.NORMAL if paths_ready else _tk.DISABLED
            )
        self._refresh_scheduler_widgets()

    def _normalize_output_paths(self, value) -> _Tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, (str, _Path)):
            return (str(value),)
        return tuple(str(path) for path in value)

    def _snapshot_advanced_options(self) -> AdvancedOptions:
        return _advanced_options_from_dict(_serialize_advanced_options(self.advanced_options))

    def _snapshot_checkbox_values(self) -> _Dict[_CheckboxKeys, bool]:
        return {
            key: checkbox.variable.get()
            for key, checkbox in self._checkboxes.items()
        }

    def _resume_checkpoint_button(self):
        return self._widgets.get(_GUIWidgets.RESUME_CHECKPOINT)

    @staticmethod
    def _resume_checkpoint_paths_tuple(selection) -> _Tuple[str, ...]:
        if selection in (None, ""):
            return tuple()
        if isinstance(selection, (str, _Path)):
            return (str(selection),)
        return tuple(str(path) for path in selection if path not in (None, ""))

    @staticmethod
    def _is_a2_architecture_choice(architecture) -> bool:
        try:
            return _core._is_a2_architecture(_core.Architecture(architecture))
        except Exception:
            return False

    def _current_resume_architecture_is_a2(self) -> bool:
        controller = getattr(self, "_advanced_options_controller", None)
        if controller is not None:
            try:
                return self._is_a2_architecture_choice(controller._current_architecture())
            except Exception:
                pass
        return self._is_a2_architecture_choice(self.advanced_options.architecture)

    @staticmethod
    def _a2_submodel_index_from_path(checkpoint_path: _Path) -> _Optional[int]:
        name = checkpoint_path.name.lower()
        match = _re.search(r"submodel[_ -]?([0-9]+)", name)
        if match:
            index = int(match.group(1))
            return index if 0 <= index < _A2_CHECKPOINT_SUBMODEL_COUNT else None
        match = _re.search(r"packed[_ -]?best[_ -]?([12])(?:\D|$)", name)
        if match:
            return int(match.group(1)) - 1
        return None

    def _is_a2_checkpoint_candidate(self, checkpoint_path: _Path) -> bool:
        return (
            self._a2_submodel_index_from_path(checkpoint_path) is not None
            or (checkpoint_path.parent / "packed_best.json").is_file()
        )

    def _resume_checkpoint_display_name(self, selection=None) -> str:
        selection = self._resume_checkpoint_path if selection is None else selection
        paths = self._resume_checkpoint_paths_tuple(selection)
        if len(paths) == 0:
            return _RESUME_CHECKPOINT_BUTTON_TEXT
        if len(paths) == 1:
            return _Path(paths[0]).name
        by_index = {}
        for path in paths:
            checkpoint_path = _Path(path)
            index = self._a2_submodel_index_from_path(checkpoint_path)
            if index is not None:
                by_index[index] = checkpoint_path.name
        if set(by_index) == set(range(_A2_CHECKPOINT_SUBMODEL_COUNT)):
            names = [by_index[i] for i in range(_A2_CHECKPOINT_SUBMODEL_COUNT)]
            return "A2: " + " + ".join(names)
        return " + ".join(_Path(path).name for path in paths)

    def _stop_resume_checkpoint_marquee(self):
        after_id = getattr(self, "_resume_checkpoint_marquee_after_id", None)
        self._resume_checkpoint_marquee_after_id = None
        if after_id is not None:
            try:
                self._root.after_cancel(after_id)
            except _tk.TclError:
                pass

    def _resume_checkpoint_visible_text(self, name: str) -> str:
        width = _RESUME_CHECKPOINT_BUTTON_CHARS
        if len(name) <= width:
            return name
        offset = max(0, min(self._resume_checkpoint_marquee_offset, len(name) - width))
        return name[offset : offset + width]

    def _schedule_resume_checkpoint_marquee(self):
        self._stop_resume_checkpoint_marquee()
        if not self._resume_checkpoint_path:
            return
        name = self._resume_checkpoint_display_name()
        if len(name) <= _RESUME_CHECKPOINT_BUTTON_CHARS:
            return

        def tick():
            if not self._resume_checkpoint_path:
                self._stop_resume_checkpoint_marquee()
                return
            button = self._resume_checkpoint_button()
            if button is None:
                return
            name_now = self._resume_checkpoint_display_name()
            max_offset = max(0, len(name_now) - _RESUME_CHECKPOINT_BUTTON_CHARS)
            offset = max(0, min(self._resume_checkpoint_marquee_offset, max_offset))
            try:
                button.configure(text=name_now[offset : offset + _RESUME_CHECKPOINT_BUTTON_CHARS])
            except _tk.TclError:
                self._stop_resume_checkpoint_marquee()
                return
            direction = self._resume_checkpoint_marquee_direction
            if offset >= max_offset:
                direction = -1
            elif offset <= 0:
                direction = 1
            self._resume_checkpoint_marquee_direction = direction
            self._resume_checkpoint_marquee_offset = max(0, min(max_offset, offset + direction))
            self._resume_checkpoint_marquee_after_id = self._root.after(
                _RESUME_CHECKPOINT_SCROLL_MS, tick
            )

        self._resume_checkpoint_marquee_after_id = self._root.after(
            _RESUME_CHECKPOINT_SCROLL_MS, tick
        )

    def _refresh_resume_checkpoint_button(self):
        button = self._resume_checkpoint_button()
        if button is None:
            return
        self._stop_resume_checkpoint_marquee()
        if self._resume_checkpoint_path:
            self._resume_checkpoint_marquee_offset = 0
            self._resume_checkpoint_marquee_direction = 1
            name = self._resume_checkpoint_display_name()
            button.configure(
                text=self._resume_checkpoint_visible_text(name),
                style="Ink.TButton",
                width=_RESUME_CHECKPOINT_BUTTON_CHARS,
            )
            self._schedule_resume_checkpoint_marquee()
        else:
            button.configure(
                text=_RESUME_CHECKPOINT_BUTTON_TEXT,
                style="Ghost.TButton",
                width=_RESUME_CHECKPOINT_BUTTON_CHARS,
            )

    def _clear_resume_checkpoint(self):
        self._resume_checkpoint_path = None
        self._refresh_resume_checkpoint_button()

    def _initial_resume_checkpoint_dir(self) -> _Optional[str]:
        last_path = _settings.get_last_path(_settings.PathKey.CHECKPOINT_FILE)
        if last_path is None:
            last_path = _settings.get_last_path(_settings.PathKey.TRAINING_DESTINATION)
        if last_path is None:
            return None
        return str(last_path if last_path.is_dir() else last_path.parent)

    @staticmethod
    def _read_packed_best_checkpoint_paths(checkpoints_dir: _Path) -> _Dict[int, _Path]:
        metadata_path = checkpoints_dir / "packed_best.json"
        if not metadata_path.is_file():
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as fp:
                data = _json.load(fp)
        except (OSError, ValueError, TypeError):
            return {}
        result = {}
        for submodel in data.get("submodels") or []:
            if not isinstance(submodel, dict):
                continue
            try:
                index = int(submodel.get("submodel_index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < _A2_CHECKPOINT_SUBMODEL_COUNT:
                continue
            checkpoint_value = submodel.get("checkpoint_path")
            if not checkpoint_value:
                continue
            checkpoint_path = _Path(str(checkpoint_value))
            candidates = []
            if checkpoint_path.is_absolute():
                candidates.append(checkpoint_path)
            else:
                candidates.append(checkpoints_dir / checkpoint_path)
            candidates.append(checkpoints_dir / checkpoint_path.name)
            result[index] = next((path for path in candidates if path.is_file()), candidates[0])
        return result

    @staticmethod
    def _a2_sibling_checkpoint_path(
        checkpoint_path: _Path, index: int, sibling_index: int
    ) -> _Optional[_Path]:
        replacements = (
            (
                _re.compile(r"(submodel[_ -]?)([0-9]+)", flags=_re.IGNORECASE),
                lambda match: f"{match.group(1)}{sibling_index}",
            ),
            (
                _re.compile(r"(packed[_ -]?best[_ -]?)([12])(?=\D|$)", flags=_re.IGNORECASE),
                lambda match: f"{match.group(1)}{sibling_index + 1}",
            ),
        )
        for pattern, replacement in replacements:
            sibling_name = pattern.sub(replacement, checkpoint_path.name, count=1)
            if sibling_name == checkpoint_path.name:
                continue
            sibling_path = checkpoint_path.with_name(sibling_name)
            if sibling_path.is_file():
                return sibling_path
        return None

    @staticmethod
    def _same_path(left: _Path, right: _Path) -> bool:
        try:
            return left.resolve(strict=False) == right.resolve(strict=False)
        except OSError:
            return str(left).lower() == str(right).lower()

    def _resolve_a2_resume_checkpoint_pair(
        self, selected_paths: _Sequence[_Union[str, _Path]]
    ) -> _Optional[_Tuple[str, ...]]:
        selected = tuple(_Path(path) for path in selected_paths if path not in (None, ""))
        if len(selected) == 0:
            return None
        by_index: _Dict[int, _Path] = {}

        def add_path(index: _Optional[int], path: _Path):
            if index is None:
                return
            if not 0 <= index < _A2_CHECKPOINT_SUBMODEL_COUNT:
                return
            existing = by_index.get(index)
            if existing is not None and not self._same_path(existing, path):
                raise ValueError(
                    f"Two different checkpoints were provided for A2 submodel {index}."
                )
            by_index[index] = path

        for path in selected:
            add_path(self._a2_submodel_index_from_path(path), path)
            for index, metadata_path in self._read_packed_best_checkpoint_paths(path.parent).items():
                add_path(index, metadata_path)

        for path in selected:
            index = self._a2_submodel_index_from_path(path)
            if index is None:
                continue
            for sibling_index in range(_A2_CHECKPOINT_SUBMODEL_COUNT):
                if sibling_index in by_index:
                    continue
                sibling_path = self._a2_sibling_checkpoint_path(path, index, sibling_index)
                if sibling_path is not None:
                    add_path(sibling_index, sibling_path)

        if all(index in by_index for index in range(_A2_CHECKPOINT_SUBMODEL_COUNT)):
            return tuple(str(by_index[index]) for index in range(_A2_CHECKPOINT_SUBMODEL_COUNT))
        return None

    def _validated_a2_resume_checkpoint_pair(
        self, selected_paths: _Sequence[_Union[str, _Path]]
    ) -> _Optional[_Tuple[str, ...]]:
        selected = tuple(_Path(path) for path in selected_paths if path not in (None, ""))
        if len(selected) == 0:
            return None
        for path in selected:
            if not self._validate_resume_checkpoint_file(path):
                return None
        try:
            checkpoint_paths = self._resolve_a2_resume_checkpoint_pair(selected)
        except ValueError as exc:
            _messagebox.showerror(
                "Resume Training From CKPT",
                str(exc),
                parent=self._root,
            )
            return None
        if checkpoint_paths is None:
            _messagebox.showerror(
                "Resume Training From CKPT",
                "A2 resume requires both packed-best checkpoints: submodel 0 and submodel 1.",
                parent=self._root,
            )
            return None
        for checkpoint_path in checkpoint_paths:
            if not self._validate_resume_checkpoint_file(_Path(checkpoint_path)):
                return None
        return checkpoint_paths

    def _choose_a2_resume_checkpoints(self, initial_dir: _Optional[str]):
        dialog_kwargs = {
            "title": "Choose both A2 packed-best checkpoints",
            "filetypes": _RESUME_CHECKPOINT_FILETYPES,
            "parent": self._root,
        }
        if initial_dir:
            dialog_kwargs["initialdir"] = initial_dir
        result = _filedialog.askopenfilenames(**dialog_kwargs)
        if not result:
            return None
        return self._validated_a2_resume_checkpoint_pair(result)

    def _choose_resume_checkpoint(self):
        initial_dir = self._initial_resume_checkpoint_dir()
        if self._current_resume_architecture_is_a2():
            checkpoint_paths = self._choose_a2_resume_checkpoints(initial_dir)
            if checkpoint_paths is None:
                return
            self._resume_checkpoint_path = checkpoint_paths
            _settings.set_last_path(
                _settings.PathKey.CHECKPOINT_FILE, _Path(checkpoint_paths[0])
            )
            self._apply_resume_checkpoint_defaults(checkpoint_paths)
            self._refresh_resume_checkpoint_button()
            return

        dialog_kwargs = {
            "title": "Choose checkpoint to continue training",
            "filetypes": _RESUME_CHECKPOINT_FILETYPES,
            "parent": self._root,
        }
        if initial_dir:
            dialog_kwargs["initialdir"] = initial_dir
        result = _filedialog.askopenfilename(**dialog_kwargs)
        if not result:
            return
        checkpoint_path = _Path(result)
        if not self._validate_resume_checkpoint_file(checkpoint_path):
            return
        if self._is_a2_checkpoint_candidate(checkpoint_path):
            checkpoint_paths = self._validated_a2_resume_checkpoint_pair((checkpoint_path,))
            if checkpoint_paths is None:
                return
            self._resume_checkpoint_path = checkpoint_paths
            _settings.set_last_path(
                _settings.PathKey.CHECKPOINT_FILE, _Path(checkpoint_paths[0])
            )
            self._apply_resume_checkpoint_defaults(checkpoint_paths)
            self._refresh_resume_checkpoint_button()
            return
        self._resume_checkpoint_path = str(checkpoint_path)
        _settings.set_last_path(_settings.PathKey.CHECKPOINT_FILE, checkpoint_path)
        self._apply_resume_checkpoint_defaults(checkpoint_path)
        self._refresh_resume_checkpoint_button()

    def _validate_resume_checkpoint_file(self, checkpoint_path: _Path) -> bool:
        if not checkpoint_path.is_file():
            _messagebox.showerror(
                "Resume Training From CKPT",
                f"Checkpoint file does not exist:\n{checkpoint_path}",
                parent=self._root,
            )
            return False
        if checkpoint_path.suffix.lower() != ".ckpt":
            _messagebox.showerror(
                "Resume Training From CKPT",
                "Choose a PyTorch Lightning checkpoint file ending in .ckpt.",
                parent=self._root,
            )
            return False
        return True

    def _validated_resume_checkpoint_path(self) -> _Optional[_ResumeCheckpointSelection]:
        paths = self._resume_checkpoint_paths_tuple(self._resume_checkpoint_path)
        if len(paths) == 0:
            return None
        if len(paths) == 1:
            path = _Path(paths[0])
            if not self._validate_resume_checkpoint_file(path):
                return None
            if self._current_resume_architecture_is_a2() or self._is_a2_checkpoint_candidate(path):
                return self._validated_a2_resume_checkpoint_pair((path,))
            return str(path)
        if len(paths) == _A2_CHECKPOINT_SUBMODEL_COUNT:
            return self._validated_a2_resume_checkpoint_pair(paths)
        _messagebox.showerror(
            "Resume Training From CKPT",
            "A2 resume requires exactly two checkpoint files.",
            parent=self._root,
        )
        return None

    @staticmethod
    def _checkpoint_run_dir(checkpoint_path: _Path) -> _Path:
        return (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name.lower() == "checkpoints"
            else checkpoint_path.parent
        )

    def _infer_architecture_from_run_path(self, run_dir: _Path) -> _Optional[_core.Architecture]:
        for part in reversed(run_dir.parts):
            architecture = _coerce_architecture(part)
            if architecture is not None:
                return architecture
        return None

    @staticmethod
    def _parse_lr_decay_from_run_path(run_dir: _Path) -> _Tuple[_Optional[float], _Optional[float]]:
        for part in reversed(run_dir.parts):
            match = _re.search(
                r"(?:^|_)lr(?P<lr>[^_]+)_lrdecay(?P<decay>[^_]+)",
                part,
                flags=_re.IGNORECASE,
            )
            if match:
                return (
                    _parse_optional_float(match.group("lr")),
                    _parse_optional_float(match.group("decay")),
                )
        return None, None

    @staticmethod
    def _checkpoint_optimizer_lrs(checkpoint: _Optional[_Dict[str, _Any]]) -> _Tuple[float, ...]:
        if not isinstance(checkpoint, dict):
            return tuple()
        values = []
        optimizers = checkpoint.get("optimizer_states") or []
        for optimizer in optimizers:
            if not isinstance(optimizer, dict):
                continue
            for group in optimizer.get("param_groups") or []:
                if isinstance(group, dict):
                    lr = _parse_optional_float(group.get("lr"))
                    if lr is not None:
                        values.append(lr)
        return tuple(values)

    @classmethod
    def _checkpoint_optimizer_lr(cls, checkpoint: _Optional[_Dict[str, _Any]]) -> _Optional[float]:
        lrs = cls._checkpoint_optimizer_lrs(checkpoint)
        return None if len(lrs) == 0 else lrs[0]

    @staticmethod
    def _consensus_float(values: _Sequence[float]) -> _Optional[float]:
        values = [float(value) for value in values if value is not None]
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        low = min(values)
        high = max(values)
        mean = sum(values) / len(values)
        tolerance = max(1.0e-12, abs(mean) * 0.01)
        return mean if high - low <= tolerance else None

    @staticmethod
    def _infer_a2_architecture_from_checkpoint(
        checkpoint: _Optional[_Dict[str, _Any]]
    ) -> _Optional[_core.Architecture]:
        if not isinstance(checkpoint, dict):
            return None
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return None
        if not any(str(key).endswith("_weight_mask") for key in state_dict):
            return None
        weight = state_dict.get("_net._net._layer_arrays.0._rechannel.weight")
        shape = getattr(weight, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        channels = int(shape[0])
        bottleneck_channels = None
        conv_weight = state_dict.get("_net._net._layer_arrays.0._layers.0._conv.weight")
        conv_shape = getattr(conv_weight, "shape", None)
        if conv_shape is not None and len(conv_shape) > 0:
            bottleneck_channels = int(conv_shape[0])
        if channels == 11:
            return _core.Architecture.A2_FULL_LITE
        if channels == 20:
            if bottleneck_channels == 19:
                return _core.Architecture.A2_COMPLEX_NANO64X4
            if bottleneck_channels == 18:
                return _core.Architecture.A2_COMPLEX_NANO125X3
            return _core.Architecture.A2_COMPLEX_REVYLO
        if channels == 19:
            layer_indices = []
            for key in state_dict:
                match = _re.search(r"_layers\.(\d+)\._conv\.weight$", str(key))
                if match is not None:
                    layer_indices.append(int(match.group(1)))
            if layer_indices and max(layer_indices) + 1 == 35:
                return _core.Architecture.A2_DOUBLEPLUS_LITE
            return _core.Architecture.A2_COMPLEX_LITE
        return None

    @staticmethod
    def _checkpoint_scheduler_info(
        checkpoint: _Optional[_Dict[str, _Any]]
    ) -> _Tuple[_Optional[_core.LearningRateScheduler], _Optional[float]]:
        if not isinstance(checkpoint, dict):
            return None, None
        schedulers = [s for s in (checkpoint.get("lr_schedulers") or []) if isinstance(s, dict)]
        if len(schedulers) == 0:
            return None, None
        if len(schedulers) > 1 and (
            "factor" in schedulers[1] or "patience" in schedulers[1]
        ):
            return (
                _core.LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU,
                _parse_optional_float(schedulers[1].get("factor")),
            )
        scheduler_state = schedulers[0]
        if "gamma" in scheduler_state:
            gamma = _parse_optional_float(scheduler_state.get("gamma"))
            return (
                _core.LearningRateScheduler.EXPONENTIAL,
                None if gamma is None else max(0.0, 1.0 - gamma),
            )
        if "factor" in scheduler_state or "patience" in scheduler_state:
            return (
                _core.LearningRateScheduler.REDUCE_ON_PLATEAU,
                _parse_optional_float(scheduler_state.get("factor")),
            )
        if "T_max" in scheduler_state:
            return _core.LearningRateScheduler.COSINE_ANNEALING, None
        if "_milestones" in scheduler_state or "_schedulers" in scheduler_state:
            return _core.LearningRateScheduler.WARMUP_COSINE_DECAY, None
        if "_schedule_phases" in scheduler_state:
            return _core.LearningRateScheduler.ONE_CYCLE, None
        if "T_0" in scheduler_state or "T_mult" in scheduler_state:
            return _core.LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS, None
        return None, None

    def _infer_resume_checkpoint_defaults(
        self, checkpoint_selection: _ResumeCheckpointSelection
    ) -> _Dict[str, _Any]:
        checkpoint_paths = tuple(_Path(path) for path in self._resume_checkpoint_paths_tuple(checkpoint_selection))
        if len(checkpoint_paths) == 0:
            return {}
        run_dir = self._checkpoint_run_dir(checkpoint_paths[0])
        saved_info = self._read_saved_analysis_info(run_dir)
        path_lr, path_decay = self._parse_lr_decay_from_run_path(run_dir)
        checkpoints = []
        for checkpoint_path in checkpoint_paths:
            try:
                checkpoints.append(self._load_checkpoint(checkpoint_path))
            except Exception as exc:
                print(f"Could not inspect checkpoint metadata for {checkpoint_path}: {exc}")
        checkpoint_scheduler = None
        checkpoint_decay = None
        for checkpoint in checkpoints:
            scheduler, decay = self._checkpoint_scheduler_info(checkpoint)
            if checkpoint_scheduler is None:
                checkpoint_scheduler = scheduler
            if checkpoint_decay is None:
                checkpoint_decay = decay
        is_a2_pair = len(checkpoint_paths) > 1

        architecture = _coerce_architecture(
            None if saved_info is None else saved_info.get("architecture")
        )
        if architecture is None:
            architecture = self._infer_architecture_from_run_path(run_dir)
        if architecture is None:
            for checkpoint in checkpoints:
                architecture = self._infer_a2_architecture_from_checkpoint(checkpoint)
                if architecture is not None:
                    break

        saved_scheduler = _coerce_scheduler(
            None if saved_info is None else saved_info.get("scheduler")
        )
        saved_lr = None if saved_info is None else _parse_optional_float(saved_info.get("learning_rate"))
        saved_decay = None if saved_info is None else _parse_optional_float(saved_info.get("lr_decay"))

        warnings = []
        if is_a2_pair:
            scheduler = saved_scheduler or checkpoint_scheduler
            learning_rate = saved_lr
            if learning_rate is None:
                learning_rate = path_lr
            if learning_rate is None:
                checkpoint_lrs = [
                    lr
                    for checkpoint in checkpoints
                    for lr in self._checkpoint_optimizer_lrs(checkpoint)
                ]
                learning_rate = self._consensus_float(checkpoint_lrs)
                if learning_rate is None and len(checkpoint_lrs) > 0:
                    lr_text = ", ".join(_fmt_float(value) for value in checkpoint_lrs)
                    warnings.append(
                        "A2 packed-best checkpoints have different optimizer LRs "
                        f"({lr_text}); kept the current LR because no run metadata was available."
                    )
            decay = saved_decay
            if decay is None:
                decay = path_decay
            if decay is None:
                decay = checkpoint_decay
        else:
            checkpoint = checkpoints[0] if len(checkpoints) > 0 else None
            scheduler = _coerce_scheduler(
                None if saved_info is None else saved_info.get("scheduler"),
                checkpoint_scheduler,
            )
            learning_rate = self._checkpoint_optimizer_lr(checkpoint)
            if learning_rate is None and saved_info is not None:
                learning_rate = saved_lr
            if learning_rate is None:
                learning_rate = path_lr
            decay = checkpoint_decay
            if decay is None and saved_info is not None:
                decay = saved_decay
            if decay is None:
                decay = path_decay

        return {
            "architecture": architecture,
            "lr_scheduler": scheduler,
            "lr": learning_rate,
            "lr_decay": decay,
            "run_dir": run_dir,
            "warnings": warnings,
        }

    def _apply_resume_checkpoint_defaults(self, checkpoint_selection: _ResumeCheckpointSelection):
        inferred = self._infer_resume_checkpoint_defaults(checkpoint_selection)
        updates = _serialize_advanced_options(self.advanced_options)
        architecture = inferred.get("architecture")
        scheduler = inferred.get("lr_scheduler")
        learning_rate = inferred.get("lr")
        decay = inferred.get("lr_decay")
        if architecture is not None:
            updates["architecture"] = architecture.value
        if scheduler is not None:
            updates["lr_scheduler"] = scheduler.value
            updates["stage2_lr_scheduler"] = scheduler.value
        if learning_rate is not None:
            updates["lr"] = learning_rate
            updates["stage2_lr"] = learning_rate
        if decay is not None:
            updates["lr_decay"] = decay
            updates["stage2_lr_decay"] = decay
        self.advanced_options = _advanced_options_from_dict(
            updates, fallback=self.advanced_options
        )
        controller = getattr(self, "_advanced_options_controller", None)
        if controller is not None:
            try:
                controller._set_widgets_from_advanced_options(self.advanced_options)
            except _tk.TclError:
                self._advanced_options_controller = None
        self._persist_gui_state()
        summary = []
        checkpoint_count = len(self._resume_checkpoint_paths_tuple(checkpoint_selection))
        if checkpoint_count > 1:
            summary.append(f"checkpoints={checkpoint_count}")
        if architecture is not None:
            summary.append(f"architecture={_architecture_display_label(architecture)}")
        if scheduler is not None:
            summary.append(f"scheduler={_scheduler_display_name(scheduler)}")
        if learning_rate is not None:
            summary.append(f"lr={_fmt_float(learning_rate)}")
        if decay is not None:
            summary.append(f"decay={_fmt_float(decay)}")
        if summary:
            print("Checkpoint defaults loaded: " + ", ".join(summary))
        for warning in inferred.get("warnings") or []:
            print("WARNING: " + warning)

    def _create_training_job(self, ignore_checks: bool = False) -> _Optional[_TrainingJob]:
        input_path = self._widgets[_GUIWidgets.INPUT_PATH].val
        output_paths = self._normalize_output_paths(
            self._widgets[_GUIWidgets.OUTPUT_PATH].val
        )
        training_destination = self._widgets[_GUIWidgets.TRAINING_DESTINATION].val
        if input_path is None or training_destination is None or len(output_paths) == 0:
            self._check_button_states()
            return None
        resume_checkpoint_path = self._validated_resume_checkpoint_path()
        if self._resume_checkpoint_path and resume_checkpoint_path is None:
            return None
        return _TrainingJob(
            input_path=str(input_path),
            output_paths=output_paths,
            training_destination=str(training_destination),
            advanced_options=self._snapshot_advanced_options(),
            checkbox_values=self._snapshot_checkbox_values(),
            user_metadata=_copy_user_metadata(self.user_metadata),
            user_metadata_flag=bool(self.user_metadata_flag),
            resume_checkpoint_path=resume_checkpoint_path,
            ignore_checks=ignore_checks,
        )

    def _copy_training_job(
        self,
        job: _TrainingJob,
        *,
        output_paths: _Optional[_Tuple[str, ...]] = None,
        ignore_checks: _Optional[bool] = None,
    ) -> _TrainingJob:
        return _TrainingJob(
            input_path=job.input_path,
            output_paths=job.output_paths if output_paths is None else output_paths,
            training_destination=job.training_destination,
            advanced_options=job.advanced_options,
            checkbox_values=dict(job.checkbox_values),
            user_metadata=_copy_user_metadata(job.user_metadata),
            user_metadata_flag=job.user_metadata_flag,
            resume_checkpoint_path=job.resume_checkpoint_path,
            ignore_checks=job.ignore_checks if ignore_checks is None else ignore_checks,
            export_suffixes=(
                None if job.export_suffixes is None else dict(job.export_suffixes)
            ),
            schedule_status=_SCHEDULE_STATUS_QUEUED,
            schedule_result="",
        )

    def _training_job_with_ignore_checks(self, job: _TrainingJob) -> _TrainingJob:
        return self._copy_training_job(job, ignore_checks=True)

    def _filter_training_jobs_by_output_paths(
        self,
        jobs: _Sequence[_TrainingJob],
        output_paths: _Sequence[str],
    ) -> _List[_TrainingJob]:
        allowed_paths = {str(path) for path in output_paths}
        filtered_jobs = []
        for job in jobs:
            kept_paths = tuple(
                output_path
                for output_path in job.output_paths
                if str(output_path) in allowed_paths
            )
            if kept_paths:
                filtered_jobs.append(
                    self._copy_training_job(job, output_paths=kept_paths)
                )
        return filtered_jobs

    def _split_training_job_by_output(self, job: _TrainingJob) -> _List[_TrainingJob]:
        return [
            self._copy_training_job(job, output_paths=(output_path,))
            for output_path in job.output_paths
        ]

    def _format_training_job_label(self, job: _TrainingJob) -> str:
        options = job.advanced_options
        output_names = [_job_export_basename(job, path) for path in job.output_paths]
        if len(output_names) == 1:
            output_label = output_names[0]
        else:
            output_label = f"{output_names[0]} + {len(output_names) - 1} more"
        resume_label = ""
        if job.resume_checkpoint_path:
            resume_label = f" | ckpt {self._resume_checkpoint_display_name(job.resume_checkpoint_path)}"
        return (
            f"{_architecture_display_label(options.architecture)} | {options.num_epochs}e"
            f" | b{options.batch_size} ny{options.ny}"
            f" | lr{_fmt_float(options.lr)}{resume_label} | {output_label}"
        )

    def _job_epoch_units(self, job: _TrainingJob) -> int:
        options = job.advanced_options
        total_epochs = max(0, int(options.num_epochs))
        if options.stage_mode == _core.TrainingStageMode.TWO_STAGE:
            total_epochs += max(0, int(options.stage2_epochs))
        elif options.stage_mode == _core.TrainingStageMode.REFINEMENT_ONLY:
            total_epochs = max(0, int(options.stage2_epochs))
        return max(1, total_epochs)

    @staticmethod
    def _schedule_status(job: _TrainingJob) -> str:
        return (
            getattr(job, "schedule_status", _SCHEDULE_STATUS_QUEUED)
            or _SCHEDULE_STATUS_QUEUED
        )

    @staticmethod
    def _set_schedule_state(
        job: _TrainingJob, status: str, result: str = ""
    ) -> None:
        job.schedule_status = status
        job.schedule_result = result

    def _job_is_queued(self, job: _TrainingJob) -> bool:
        return self._schedule_status(job) == _SCHEDULE_STATUS_QUEUED

    def _first_queued_scheduled_job_locked(self) -> _Optional[_TrainingJob]:
        for job in self._scheduled_jobs:
            if self._job_is_queued(job):
                return job
        return None

    def _format_eta_cell(self, seconds: _Optional[float]) -> str:
        if seconds is None:
            return "Estimating"
        seconds = max(0.0, float(seconds))
        done_at = _time.strftime("%H:%M", _time.localtime(_time.time() + seconds))
        return f"{self._format_duration(seconds)} @ {done_at}"

    def _schedule_eta_cells(
        self,
        scheduled_jobs: _Sequence[_TrainingJob],
        training_active: bool,
        seconds_per_epoch: _Optional[float],
        active_remaining_epoch_units: int,
    ) -> _Tuple[_List[str], str]:
        if seconds_per_epoch is None:
            summary = "Estimating" if (training_active or scheduled_jobs) else ""
            return ["Estimating" for _job in scheduled_jobs], summary

        cumulative_seconds = (
            max(0, active_remaining_epoch_units) * seconds_per_epoch
            if training_active
            else 0.0
        )
        eta_cells = []
        for job in scheduled_jobs:
            cumulative_seconds += self._job_epoch_units(job) * seconds_per_epoch
            eta_cells.append(self._format_eta_cell(cumulative_seconds))
        if training_active or scheduled_jobs:
            return eta_cells, self._format_eta_cell(cumulative_seconds)
        return eta_cells, ""

    @staticmethod
    def _coerce_metric_value(value: _Any) -> _Optional[float]:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _logged_metric_value(
        self, metrics: _Dict[str, _Any], *keys: str
    ) -> _Optional[float]:
        for key in keys:
            if key in metrics:
                value = self._coerce_metric_value(metrics.get(key))
                if value is not None:
                    return value
        return None

    def _format_schedule_metric(self, value: _Optional[float]) -> str:
        text = self._format_metric(value)
        return text if text else "pending"

    def _format_training_result_cell(
        self, train_output: _core.TrainOutput, architecture: _core.Architecture
    ) -> str:
        metadata = getattr(train_output, "metadata", None)
        logged_metrics = getattr(metadata, "logged_metrics", {}) or {}
        try:
            architecture = _core.Architecture(architecture)
        except ValueError:
            architecture = None

        if architecture is not None and _core._is_a2_architecture(architecture):
            lower_esr = self._logged_metric_value(
                logged_metrics,
                "final_plotted_validation_esr_packed_0",
                "ESR_packed_0",
                "ESRPREEMPH_packed_0",
            )
            higher_esr = self._logged_metric_value(
                logged_metrics,
                "final_plotted_validation_esr_packed_1",
                "ESR_packed_1",
                "ESRPREEMPH_packed_1",
            )
            if lower_esr is not None or higher_esr is not None:
                return (
                    f"lower {self._format_schedule_metric(lower_esr)} | "
                    f"higher {self._format_schedule_metric(higher_esr)}"
                )

        final_esr = self._coerce_metric_value(
            getattr(metadata, "validation_esr", None)
        )
        if final_esr is None:
            final_esr = self._logged_metric_value(
                logged_metrics,
                "final_plotted_validation_esr",
                "ESR",
                "ESRPREEMPH",
            )
        if final_esr is not None:
            return f"ESR {self._format_schedule_metric(final_esr)}"
        return "ESR pending"

    def _refresh_scheduler_widgets(self):
        if not hasattr(self, "_schedule_tree"):
            return
        self._refresh_header_status()
        with self._schedule_lock:
            scheduled_jobs = list(self._scheduled_jobs)
            queued_jobs = [job for job in scheduled_jobs if self._job_is_queued(job)]
            training_active = self._training_active
            current_label = self._current_training_label
            seconds_per_epoch = self._eta_seconds_per_epoch
            active_remaining_epoch_units = self._active_remaining_epoch_units

        eta_cells, schedule_eta = self._schedule_eta_cells(
            queued_jobs,
            training_active,
            seconds_per_epoch,
            active_remaining_epoch_units,
        )
        eta_by_job_id = {
            id(job): eta_cells[index] for index, job in enumerate(queued_jobs)
        }
        status_counts = {
            _SCHEDULE_STATUS_QUEUED: 0,
            _SCHEDULE_STATUS_RUNNING: 0,
            _SCHEDULE_STATUS_FINISHED: 0,
            _SCHEDULE_STATUS_FAILED: 0,
            _SCHEDULE_STATUS_ABORTED: 0,
        }
        self._schedule_tree.delete(*self._schedule_tree.get_children())
        for index, job in enumerate(scheduled_jobs, start=1):
            status = self._schedule_status(job)
            status_counts[status] = status_counts.get(status, 0) + 1
            label = self._format_training_job_label(job)
            if status == _SCHEDULE_STATUS_RUNNING:
                label = f"[running] {label}"
                result_cell = "Running"
            elif status == _SCHEDULE_STATUS_FINISHED:
                label = f"[done] {label}"
                result_cell = job.schedule_result or "ESR pending"
            elif status == _SCHEDULE_STATUS_FAILED:
                label = f"[failed] {label}"
                result_cell = job.schedule_result or "failed"
            elif status == _SCHEDULE_STATUS_ABORTED:
                label = f"[aborted] {label}"
                result_cell = job.schedule_result or "aborted"
            else:
                status = _SCHEDULE_STATUS_QUEUED
                result_cell = eta_by_job_id.get(id(job), "Estimating")
            self._schedule_tree.insert(
                "",
                _tk.END,
                iid=f"job-{index - 1}",
                values=(f"{index}. {label}", result_cell),
                tags=(status,),
            )
        if len(scheduled_jobs) == 0:
            self._schedule_tree.insert(
                "",
                _tk.END,
                values=("-- queue empty --", ""),
                tags=("empty",),
            )

        if training_active:
            status_text = f"Running: {current_label}"
        else:
            status_text = "Idle"
        details = []
        if schedule_eta:
            details.append(f"Schedule ETA {schedule_eta}")
        if status_counts.get(_SCHEDULE_STATUS_FINISHED, 0) > 0:
            details.append(f"Done {status_counts[_SCHEDULE_STATUS_FINISHED]}")
        if status_counts.get(_SCHEDULE_STATUS_FAILED, 0) > 0:
            details.append(f"Failed {status_counts[_SCHEDULE_STATUS_FAILED]}")
        if status_counts.get(_SCHEDULE_STATUS_ABORTED, 0) > 0:
            details.append(f"Aborted {status_counts[_SCHEDULE_STATUS_ABORTED]}")
        if details:
            status_text = f"{status_text} | {' | '.join(details)}"
        self._scheduler_status_var.set(status_text)
        if hasattr(self, "_scheduler_queued_var"):
            self._scheduler_queued_var.set(
                f"Queued  {status_counts.get(_SCHEDULE_STATUS_QUEUED, 0)}"
            )

        has_queued_jobs = status_counts.get(_SCHEDULE_STATUS_QUEUED, 0) > 0
        has_rows = len(scheduled_jobs) > 0
        has_clearable_rows = any(
            self._schedule_status(job) != _SCHEDULE_STATUS_RUNNING
            for job in scheduled_jobs
        )
        if _GUIWidgets.START_SCHEDULE in self._widgets:
            self._widgets[_GUIWidgets.START_SCHEDULE]["state"] = (
                _tk.NORMAL if has_queued_jobs and not training_active else _tk.DISABLED
            )
        if _GUIWidgets.REMOVE_SCHEDULED in self._widgets:
            self._widgets[_GUIWidgets.REMOVE_SCHEDULED]["state"] = (
                _tk.NORMAL if has_rows else _tk.DISABLED
            )
        if _GUIWidgets.CLEAR_SCHEDULED in self._widgets:
            self._widgets[_GUIWidgets.CLEAR_SCHEDULED]["state"] = (
                _tk.NORMAL if has_clearable_rows else _tk.DISABLED
            )

    def _reset_scheduler_progress(self, message: str = "No active run"):
        self._scheduler_metrics_state = None
        if hasattr(self, "_scheduler_progress_var"):
            self._scheduler_progress_var.set(0.0)
        self._set_scheduler_progress_info(message)

    def _configure_scheduler_progress_info_tags(self):
        if not hasattr(self, "_scheduler_progress_info"):
            return
        tag_colors = {
            "stage": _GUI_TEXT,
            "muted": _GUI_MUTED,
            "current_label": _GUI_METRIC_CURRENT,
            "best_label": _GUI_METRIC_BEST,
            "eta": _GUI_METRIC_LOSS,
            "loss": _GUI_METRIC_LOSS,
            "esr": _GUI_METRIC_ESR,
            "esrpre": _GUI_METRIC_ESR,
            "mrstft": _GUI_METRIC_CURRENT,
            "mse": _GUI_METRIC_MSE,
            "band": _GUI_MUTED,
        }
        for tag, color in tag_colors.items():
            self._scheduler_progress_info.tag_configure(tag, foreground=color)
        for tag in ("current_label", "best_label", "eta"):
            self._scheduler_progress_info.tag_configure(tag, font=_FT_METRIC)

    def _set_scheduler_progress_info(self, message: str):
        self._scheduler_metrics_state = None
        if hasattr(self, "_scheduler_progress_info_var"):
            self._scheduler_progress_info_var.set(message)
        widget = getattr(self, "_scheduler_progress_info", None)
        if widget is None:
            return
        if not isinstance(widget, _tk.Text):
            return
        widget.configure(state=_tk.NORMAL)
        widget.delete("1.0", _tk.END)
        self._insert_colored_progress_message(widget, message)
        widget.configure(state=_tk.DISABLED)

    def _insert_colored_progress_message(self, widget: _tk.Text, message: str):
        parts = message.split(" | ")
        for index, part in enumerate(parts):
            if index > 0:
                widget.insert(_tk.END, " | ", "muted")
            lower_part = part.lower()
            if lower_part.startswith("current: "):
                widget.insert(_tk.END, "current: ", "current_label")
                self._insert_metric_chunks(widget, part[len("current: "):], "current_label")
            elif lower_part.startswith("best: "):
                widget.insert(_tk.END, "best: ", "best_label")
                best_text = part[len("best: "):]
                if " at epoch " in best_text:
                    metrics_text, suffix = best_text.split(" at epoch ", 1)
                    self._insert_metric_chunks(widget, metrics_text, "best_label")
                    widget.insert(_tk.END, f" at epoch {suffix}", "best_label")
                else:
                    self._insert_metric_chunks(widget, best_text, "best_label")
            elif lower_part.startswith("eta"):
                widget.insert(_tk.END, part, "eta")
            else:
                widget.insert(_tk.END, part, "stage")

    def _insert_metric_chunks(self, widget: _tk.Text, text: str, fallback_tag: str):
        metric_tags = {
            "loss": "loss",
            "ESR": "esr",
            "ESRpre": "esrpre",
            "MRSTFT": "mrstft",
            "MSE": "mse",
            "low": "band",
            "mid": "band",
            "high": "band",
        }
        chunks = text.split(", ")
        for index, chunk in enumerate(chunks):
            if index > 0:
                widget.insert(_tk.END, ", ", "muted")
            label = chunk.split(" ", 1)[0]
            widget.insert(_tk.END, chunk, metric_tags.get(label, fallback_tag))

    def _start_progress_polling(self):
        self._stop_progress_polling()
        self._poll_training_progress()

    def _stop_progress_polling(self):
        if self._progress_poll_after_id is None:
            return
        try:
            self._root.after_cancel(self._progress_poll_after_id)
        except _tk.TclError:
            pass
        self._progress_poll_after_id = None

    def _poll_training_progress(self):
        with self._schedule_lock:
            training_active = self._training_active
        if not training_active:
            self._progress_poll_after_id = None
            return
        progress, message = self._collect_current_progress_info()
        if hasattr(self, "_scheduler_progress_var"):
            self._scheduler_progress_var.set(max(0.0, min(100.0, progress)))
        if getattr(self, "_scheduler_metrics_state", None) is not None:
            if hasattr(self, "_scheduler_progress_info_var"):
                self._scheduler_progress_info_var.set(message)
            _render_metrics(
                self._scheduler_progress_info, self._P, self._scheduler_metrics_state
            )
        else:
            self._set_scheduler_progress_info(message)
        self._refresh_scheduler_widgets()
        try:
            self._progress_poll_after_id = self._root.after(
                4000, self._poll_training_progress
            )
        except RuntimeError:
            self._progress_poll_after_id = None

    def _set_current_training_file(self, job: _TrainingJob, output_path: str):
        with self._schedule_lock:
            if self._schedule_status(job) not in (
                _SCHEDULE_STATUS_FINISHED,
                _SCHEDULE_STATUS_FAILED,
                _SCHEDULE_STATUS_ABORTED,
            ):
                self._set_schedule_state(job, _SCHEDULE_STATUS_RUNNING)
            self._current_training_job = job
            self._current_training_output_path = output_path
            self._current_training_label = self._format_training_job_label(job)
            self._current_training_started_at = _time.time()
            self._active_remaining_epoch_units = self._job_epoch_units(job)
        self._reset_scheduler_progress(
            f"Starting {_Path(output_path).stem}: waiting for run files..."
        )
        self._refresh_scheduler_widgets()
        self._start_progress_polling()

    def _find_current_event_file(
        self, job: _TrainingJob, output_path: str
    ) -> _Optional[_Path]:
        model_root = _Path(job.training_destination) / _job_export_basename(
            job, output_path
        )
        if not model_root.is_dir():
            return None
        try:
            event_files = [
                path
                for path in model_root.glob("**/events.out.tfevents.*")
                if path.is_file()
            ]
        except OSError:
            return None
        if len(event_files) == 0:
            return None

        def mtime(path: _Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        recent_cutoff = self._current_training_started_at - 30.0
        recent_event_files = [
            path for path in event_files if mtime(path) >= recent_cutoff
        ]
        candidates = recent_event_files or event_files
        return max(candidates, key=mtime)

    def _stage_progress_context(
        self, event_file: _Path, job: _TrainingJob
    ) -> _Tuple[str, int, int, int]:
        stage_part = next(
            (
                part
                for part in event_file.parts
                if part.lower().startswith("stage_2")
            ),
            None,
        )
        if stage_part is None:
            stage_part = next(
                (
                    part
                    for part in event_file.parts
                    if part.lower() == "stage_1"
                ),
                "stage",
            )
        if stage_part.lower().startswith("stage_2"):
            total_epochs = job.advanced_options.stage2_epochs
            completed_before_stage = (
                job.advanced_options.num_epochs
                if job.advanced_options.stage_mode == _core.TrainingStageMode.TWO_STAGE
                else 0
            )
        else:
            total_epochs = job.advanced_options.num_epochs
            completed_before_stage = 0
        job_total_epochs = self._job_epoch_units(job)
        stage_label = stage_part.replace("_", " ")
        if stage_part.lower().startswith("stage_2"):
            stage_label = stage_label.replace("stage 2", "refinement", 1)
        elif stage_part.lower() == "stage_1":
            stage_label = "core"
        return (
            stage_label,
            max(1, int(total_epochs)),
            max(0, int(completed_before_stage)),
            job_total_epochs,
        )

    def _series_latest_numeric_value(self, series) -> _Optional[float]:
        if len(series) == 0:
            return None
        return float(series[-1].value)

    def _metric_summary_for_index(self, scalar_series, index: _Optional[int]) -> str:
        metric_tags = (
            ("loss", ("val_loss",)),
            ("ESR", ("ESR",)),
            ("ESRpre", ("ESRPREEMPH",)),
            ("MRSTFT", ("MRSTFTPREEMPH", "Pre-emphasized MRSTFT", "MRSTFT")),
            ("MSE", ("MSE",)),
            ("low", ("LOW_BAND_MSE",)),
            ("mid", ("MID_BAND_MSE",)),
            ("high", ("HIGH_BAND_MSE",)),
        )
        parts = []
        for label, tags in metric_tags:
            value = None
            for tag in tags:
                series = scalar_series.get(tag, [])
                value = (
                    self._series_latest_numeric_value(series)
                    if index is None
                    else self._series_numeric_value(series, index)
                )
                if value is not None:
                    break
            if value is not None:
                parts.append(f"{label} {self._format_metric(value)}")
        return ", ".join(parts)

    def _metric_text_for_tags(
        self, scalar_series, index: _Optional[int], tags: _Sequence[str]
    ) -> str:
        for tag in tags:
            series = scalar_series.get(tag, [])
            value = (
                self._series_latest_numeric_value(series)
                if index is None
                else self._series_numeric_value(series, index)
            )
            if value is not None:
                return self._format_metric(value)
        return "pending"

    @staticmethod
    def _eta_parts_for_metrics(eta_summary: str) -> _Tuple[str, str]:
        eta_match = _re.search(r"ETA\s+([0-9:]+|calculating\.\.\.)", eta_summary)
        around_match = _re.search(r"around\s+([^)]+)", eta_summary)
        eta = eta_match.group(1) if eta_match is not None else "calculating..."
        around = around_match.group(1) if around_match is not None else ""
        return eta, around

    def _eta_summary_for_progress(
        self,
        scalar_series,
        reference_series,
        completed_epoch_units: int,
        total_epoch_units: int,
    ) -> str:
        remaining_epochs = max(0, total_epoch_units - completed_epoch_units)
        if remaining_epochs == 0:
            with self._schedule_lock:
                self._active_remaining_epoch_units = 0
            return "ETA 00:00:00"

        seconds_per_epoch = None
        tpe_series = scalar_series.get("TPE", [])
        recent_tpe_values = [
            float(item.value) for item in tpe_series[-5:] if float(item.value) > 0.0
        ]
        if len(recent_tpe_values) > 0:
            seconds_per_epoch = sum(recent_tpe_values) / len(recent_tpe_values)
        elif len(reference_series) > 1:
            elapsed = reference_series[-1].wall_time - reference_series[0].wall_time
            intervals = max(1, len(reference_series) - 1)
            if elapsed > 0.0:
                seconds_per_epoch = elapsed / intervals
        if seconds_per_epoch is None:
            elapsed = max(0.0, _time.time() - self._current_training_started_at)
            if elapsed > 0.0:
                seconds_per_epoch = elapsed / max(1, completed_epoch_units)
        if seconds_per_epoch is None:
            return "ETA calculating..."

        with self._schedule_lock:
            self._eta_seconds_per_epoch = seconds_per_epoch
            self._active_remaining_epoch_units = remaining_epochs

        eta_seconds = seconds_per_epoch * remaining_epochs
        done_at = _time.strftime(
            "%H:%M:%S", _time.localtime(_time.time() + eta_seconds)
        )
        return f"ETA {self._format_duration(eta_seconds)} (around {done_at})"

    def _collect_current_progress_info(self) -> _Tuple[float, str]:
        self._scheduler_metrics_state = None
        with self._schedule_lock:
            job = self._current_training_job
            output_path = self._current_training_output_path
        if job is None:
            return 0.0, "No active run"
        if output_path == "" and len(job.output_paths) > 0:
            output_path = job.output_paths[0]
        if output_path == "":
            return 0.0, "Starting run: waiting for output file..."

        event_file = self._find_current_event_file(job, output_path)
        if event_file is None:
            return (
                0.0,
                f"Starting {_Path(output_path).stem}: waiting for TensorBoard metrics...",
            )
        if "_EventAccumulator" not in globals():
            return 0.0, "TensorBoard event reader is unavailable."

        (
            stage_label,
            stage_total_epochs,
            completed_before_stage,
            job_total_epochs,
        ) = self._stage_progress_context(event_file, job)
        header_stage_label = (
            "refinement" if stage_label.lower().startswith("refinement") else "core"
        )
        with self._schedule_lock:
            if self._current_training_stage_label != header_stage_label:
                self._current_training_stage_label = header_stage_label
                self._root.after(0, self._refresh_header_status)
        try:
            accumulator = _EventAccumulator(str(event_file))
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags().get("scalars", []))
        except Exception as exc:
            return 0.0, f"{stage_label}: waiting for readable metrics ({exc})"

        def scalars(tag: str):
            return accumulator.Scalars(tag) if tag in scalar_tags else []

        scalar_series = {
            "val_loss": scalars("val_loss"),
            "epoch": scalars("epoch"),
            "TPE": scalars("TPE"),
            "ESR": scalars("ESR"),
            "ESRPREEMPH": scalars("ESRPREEMPH"),
            "MRSTFT": scalars("MRSTFT"),
            "MRSTFTPREEMPH": scalars("MRSTFTPREEMPH"),
            "Pre-emphasized MRSTFT": scalars("Pre-emphasized MRSTFT"),
            "MSE": scalars("MSE"),
            "LOW_BAND_MSE": scalars("LOW_BAND_MSE"),
            "MID_BAND_MSE": scalars("MID_BAND_MSE"),
            "HIGH_BAND_MSE": scalars("HIGH_BAND_MSE"),
        }
        reference_series = (
            scalar_series["val_loss"]
            or scalar_series["ESRPREEMPH"]
            or scalar_series["ESR"]
            or scalar_series["MRSTFTPREEMPH"]
            or scalar_series["Pre-emphasized MRSTFT"]
            or scalar_series["MRSTFT"]
            or scalar_series["MSE"]
            or scalar_series["LOW_BAND_MSE"]
            or scalar_series["MID_BAND_MSE"]
            or scalar_series["HIGH_BAND_MSE"]
        )
        if len(reference_series) == 0:
            return 0.0, f"{stage_label}: waiting for first validation metrics..."

        epoch_series = scalar_series["epoch"]
        if len(epoch_series) > 0:
            current_epoch_zero_based = int(round(float(epoch_series[-1].value)))
        else:
            current_epoch_zero_based = len(reference_series) - 1
        current_epoch = min(stage_total_epochs, max(1, current_epoch_zero_based + 1))
        completed_epoch_units = min(
            job_total_epochs, completed_before_stage + current_epoch
        )
        progress = 100.0 * (completed_epoch_units / job_total_epochs)

        best_index = min(
            range(len(reference_series)), key=lambda index: reference_series[index].value
        )
        if len(epoch_series) > 0:
            best_epoch_zero_based = int(
                round(float(self._series_value(epoch_series, best_index).value))
            )
        else:
            best_epoch_zero_based = best_index
        best_epoch = min(stage_total_epochs, max(1, best_epoch_zero_based + 1))
        epochs_back = max(0, current_epoch - best_epoch)

        current_metrics = self._metric_summary_for_index(scalar_series, None)
        best_metrics = self._metric_summary_for_index(scalar_series, best_index)
        if current_metrics == "":
            current_metrics = "metrics pending"
        if best_metrics == "":
            best_metrics = "metrics pending"
        eta_summary = self._eta_summary_for_progress(
            scalar_series, reference_series, completed_epoch_units, job_total_epochs
        )
        eta, around = self._eta_parts_for_metrics(eta_summary)
        self._scheduler_metrics_state = {
            "phase": stage_label,
            "ep": current_epoch,
            "total": stage_total_epochs,
            "pct": 100.0 * (current_epoch / stage_total_epochs),
            "cur_loss": self._metric_text_for_tags(scalar_series, None, ("val_loss",)),
            "cur_esr": self._metric_text_for_tags(
                scalar_series, None, ("ESR", "ESRPREEMPH")
            ),
            "cur_mse": self._metric_text_for_tags(scalar_series, None, ("MSE",)),
            "bst_loss": self._metric_text_for_tags(
                scalar_series, best_index, ("val_loss",)
            ),
            "bst_esr": self._metric_text_for_tags(
                scalar_series, best_index, ("ESR", "ESRPREEMPH")
            ),
            "bst_ep": best_epoch,
            "bst_back": epochs_back,
            "eta": eta,
            "around": around,
        }
        return (
            progress,
            (
                f"{stage_label}: epoch {current_epoch}/{stage_total_epochs} "
                f"({progress:.1f}%) | current: {current_metrics} | "
                f"best: {best_metrics} at epoch {best_epoch} "
                f"({epochs_back} epochs back) | {eta_summary}"
            ),
        )

    def _post_to_gui(self, callback: _Callable[[], None]):
        try:
            self._root.after(0, callback)
        except RuntimeError:
            pass

    def _reconcile_scheduled_export_suffixes_locked(self):
        mutable_jobs = []
        locked_jobs = []
        locked_job_ids = set()
        for job in self._scheduled_jobs:
            if self._job_is_queued(job):
                mutable_jobs.append(job)
            else:
                locked_jobs.append(job)
                locked_job_ids.add(id(job))
        if (
            self._training_active
            and self._current_training_job is not None
            and id(self._current_training_job) not in locked_job_ids
        ):
            locked_jobs.append(self._current_training_job)
        _recompute_export_suffixes(mutable_jobs, locked_jobs=locked_jobs)

    def _submit_training_job(self, job: _TrainingJob, start_immediately: bool):
        self._submit_training_jobs([job], start_immediately=start_immediately)

    def _submit_training_jobs(
        self, jobs: _Sequence[_TrainingJob], start_immediately: bool
    ):
        jobs = list(jobs)
        if len(jobs) == 0:
            return
        for job in jobs:
            self._set_schedule_state(job, _SCHEDULE_STATUS_QUEUED)
        should_start = False
        job_to_start = None
        with self._schedule_lock:
            if start_immediately and not self._training_active:
                job_to_start = jobs[0]
                self._set_schedule_state(job_to_start, _SCHEDULE_STATUS_RUNNING)
                self._training_active = True
                self._current_training_job = job_to_start
                self._current_training_output_path = (
                    job_to_start.output_paths[0]
                    if len(job_to_start.output_paths) > 0
                    else ""
                )
                self._current_training_label = self._format_training_job_label(
                    job_to_start
                )
                self._current_training_stage_label = "core"
                self._active_remaining_epoch_units = self._job_epoch_units(job_to_start)
                should_start = True
                jobs = jobs[1:]
            if len(jobs) > 0:
                self._scheduled_jobs.extend(jobs)
                self._reconcile_scheduled_export_suffixes_locked()
        self._refresh_scheduler_widgets()
        self._check_button_states()
        if any(job.resume_checkpoint_path for job in jobs) or (
            job_to_start is not None and job_to_start.resume_checkpoint_path
        ):
            self._clear_resume_checkpoint()
        if should_start and job_to_start is not None:
            self._start_training_thread(job_to_start)

    def _validate_and_submit_training_job(
        self, job: _TrainingJob, start_immediately: bool
    ):
        self._validate_and_submit_training_jobs(
            job,
            [job],
            start_immediately=start_immediately,
        )

    def _validate_and_submit_training_jobs(
        self,
        validation_job: _TrainingJob,
        jobs: _Sequence[_TrainingJob],
        start_immediately: bool,
    ):
        jobs = list(jobs)

        def submit_ignored_job():
            self._submit_training_jobs(
                [self._training_job_with_ignore_checks(job) for job in jobs],
                start_immediately=start_immediately,
            )

        def submit_good_jobs(output_paths):
            good_jobs = self._filter_training_jobs_by_output_paths(jobs, output_paths)
            if good_jobs:
                self._submit_training_jobs(
                    good_jobs,
                    start_immediately=start_immediately,
                )

        if self._validate_all_data(
            validation_job.input_path,
            validation_job.output_paths,
            advanced_options=validation_job.advanced_options,
            on_ignore=submit_ignored_job,
            on_exclude=submit_good_jobs,
        ):
            self._submit_training_jobs(jobs, start_immediately=start_immediately)

    def _add_current_to_schedule(self):
        job = self._create_training_job()
        if job is not None:
            self._validate_and_submit_training_jobs(
                job,
                self._split_training_job_by_output(job),
                start_immediately=False,
            )

    def _start_scheduled_jobs(self):
        with self._schedule_lock:
            if self._training_active:
                return
            job = self._first_queued_scheduled_job_locked()
            if job is None:
                return
            self._set_schedule_state(job, _SCHEDULE_STATUS_RUNNING)
            self._training_active = True
            self._current_training_job = job
            self._current_training_output_path = (
                job.output_paths[0] if len(job.output_paths) > 0 else ""
            )
            self._current_training_label = self._format_training_job_label(job)
            self._current_training_stage_label = "core"
            self._active_remaining_epoch_units = self._job_epoch_units(job)
        self._refresh_scheduler_widgets()
        self._check_button_states()
        self._start_training_thread(job)

    def _remove_selected_scheduled_job(self):
        selection = self._schedule_tree.selection()
        if not selection:
            return
        index = self._schedule_tree.index(selection[0])
        with self._schedule_lock:
            if 0 <= index < len(self._scheduled_jobs):
                job = self._scheduled_jobs[index]
                if self._schedule_status(job) == _SCHEDULE_STATUS_RUNNING:
                    return
                del self._scheduled_jobs[index]
                self._reconcile_scheduled_export_suffixes_locked()
        self._refresh_scheduler_widgets()

    def _clear_scheduled_jobs(self):
        with self._schedule_lock:
            if self._training_active:
                self._scheduled_jobs = [
                    job
                    for job in self._scheduled_jobs
                    if self._schedule_status(job) == _SCHEDULE_STATUS_RUNNING
                ]
            else:
                self._scheduled_jobs.clear()
            self._reconcile_scheduled_export_suffixes_locked()
        self._refresh_scheduler_widgets()

    def _start_training_thread(self, first_job: _TrainingJob):
        self._training_thread = _threading.Thread(
            target=self._training_worker_loop,
            args=(first_job,),
            daemon=True,
        )
        self._training_thread.start()

    def _training_worker_loop(self, first_job: _TrainingJob):
        job: _Optional[_TrainingJob] = first_job
        while job is not None:
            self._post_to_gui(
                lambda job=job: self._set_current_training_job(job)
            )
            stop_schedule, result_status, result_text = self._run_training_job(job)
            with self._schedule_lock:
                self._set_schedule_state(job, result_status, result_text)
                if stop_schedule:
                    job = None
                else:
                    job = self._first_queued_scheduled_job_locked()
                    if job is not None:
                        self._set_schedule_state(job, _SCHEDULE_STATUS_RUNNING)
                        self._current_training_job = job
                        self._current_training_output_path = (
                            job.output_paths[0] if len(job.output_paths) > 0 else ""
                        )
                        self._current_training_label = self._format_training_job_label(job)
                        self._current_training_stage_label = "core"
                        self._active_remaining_epoch_units = self._job_epoch_units(job)
            self._post_to_gui(self._refresh_scheduler_widgets)
        self._post_to_gui(self._finish_training_worker)

    def _set_current_training_job(self, job: _TrainingJob):
        with self._schedule_lock:
            if self._schedule_status(job) not in (
                _SCHEDULE_STATUS_FINISHED,
                _SCHEDULE_STATUS_FAILED,
                _SCHEDULE_STATUS_ABORTED,
            ):
                self._set_schedule_state(job, _SCHEDULE_STATUS_RUNNING)
            self._current_training_label = self._format_training_job_label(job)
            self._current_training_job = job
            self._current_training_output_path = (
                job.output_paths[0] if len(job.output_paths) > 0 else ""
            )
            self._current_training_started_at = _time.time()
            self._current_training_stage_label = "core"
            self._active_remaining_epoch_units = self._job_epoch_units(job)
        self._reset_scheduler_progress("Starting run: waiting for run files...")
        self._refresh_scheduler_widgets()
        self._check_button_states()
        self._start_progress_polling()

    def _finish_training_worker(self):
        with self._schedule_lock:
            self._training_active = False
            self._current_training_label = ""
            self._current_training_job = None
            self._current_training_output_path = ""
            self._current_training_started_at = 0.0
            self._current_training_stage_label = "core"
            self._active_remaining_epoch_units = 0
            self._training_thread = None
        self._stop_progress_polling()
        self._reset_scheduler_progress()
        self._refresh_scheduler_widgets()
        self._check_button_states()

    def _analyze_lightning_folders(self):
        self._disable()
        try:
            selected_dirs = self._prompt_for_lightning_folders()
            if len(selected_dirs) == 0:
                self._resume()
                return
            rows = self._collect_lightning_analysis_rows(selected_dirs)
        except Exception as exc:
            self._resume()
            _messagebox.showerror(
                "Lightning Analysis Failed",
                f"Failed to analyze Lightning folders:\n{exc}",
                parent=self._root,
            )
            return
        _LightningAnalysisModal(self._resume, rows)

    def _prompt_for_lightning_folders(self) -> _List[_Path]:
        selected_dirs: _List[_Path] = []
        last_path = _settings.get_last_path(_settings.PathKey.LIGHTNING_FOLDER)
        if last_path is None:
            last_path = _settings.get_last_path(_settings.PathKey.TRAINING_DESTINATION)
        if last_path is None:
            initial_dir = None
        elif last_path.is_dir():
            initial_dir = str(last_path)
        else:
            initial_dir = str(last_path.parent)

        while True:
            result = _filedialog.askdirectory(
                parent=self._root,
                title="Select Lightning Folder",
                initialdir=initial_dir,
            )
            if result:
                path = _Path(result)
                selected_dirs.append(path)
                _settings.set_last_path(_settings.PathKey.LIGHTNING_FOLDER, path)
                initial_dir = str(path)

            add_another = _messagebox.askyesno(
                "Add Another Lightning Folder?",
                "Do you want to add another Lightning folder to the analysis?",
                parent=self._root,
            )
            if not add_another:
                if selected_dirs:
                    _settings.set_path_selection(
                        _settings.PathKey.LIGHTNING_FOLDER, selected_dirs
                    )
                return selected_dirs

    def _collect_lightning_analysis_rows(
        self, selected_dirs: _Sequence[_Path]
    ) -> _List[_LightningAnalysisRow]:
        rows_by_path: _Dict[str, _LightningAnalysisRow] = {}
        for selected_dir in selected_dirs:
            for run_dir, event_files in self._find_lightning_runs(selected_dir):
                row = self._analyze_lightning_run(run_dir, event_files)
                rows_by_path[row.run_path] = row
        rows = list(rows_by_path.values())
        rows.sort(
            key=lambda row: (
                row.file_name.lower(),
                row.stage.lower(),
                row.architecture.lower(),
                self._sort_key_for_number(row.epochs),
                self._sort_key_for_number(row.learning_rate),
            )
        )
        return rows

    def _find_lightning_runs(
        self, selected_dir: _Path
    ) -> _List[_Sequence[_Any]]:
        run_event_files: _Dict[_Path, _List[_Path]] = {}
        for event_file in selected_dir.rglob("events.out.tfevents.*"):
            run_event_files.setdefault(event_file.parent, []).append(event_file)
        return list(run_event_files.items())

    def _analyze_lightning_run(
        self, run_dir: _Path, event_files: _Sequence[_Path]
    ) -> _LightningAnalysisRow:
        path_info = self._parse_lightning_run_path(run_dir)
        saved_info = self._read_saved_analysis_info(run_dir)
        metric_info = self._extract_lightning_metrics(event_files)
        scheduler = (
            self._string_or_blank(saved_info.get("scheduler"))
            if saved_info is not None
            else self._infer_scheduler(run_dir)
        )
        training_time = (
            self._format_duration(saved_info.get("training_time_seconds"))
            if saved_info is not None
            else metric_info["training_time"]
        )
        tpe = (
            self._format_metric(saved_info.get("training_time_per_epoch_seconds"))
            if saved_info is not None and saved_info.get("training_time_per_epoch_seconds") is not None
            else metric_info["tpe"]
        )
        capture_efficiency, quality_weighted_efficiency = self._compute_efficiency_scores(
            metric_info["esr"], training_time
        )
        return _LightningAnalysisRow(
            input_file=self._saved_or_path(saved_info, "input_file", ""),
            output_file=self._saved_or_path(
                saved_info, "output_file", path_info["file_name"]
            ),
            file_name=self._saved_or_path(saved_info, "model_name", path_info["file_name"]),
            stage=self._saved_or_path(saved_info, "stage", path_info["stage"]),
            architecture=self._saved_or_path(
                saved_info, "architecture", path_info["architecture"]
            ),
            epochs=self._saved_or_path(saved_info, "epochs", path_info["epochs"] or metric_info["epochs"]),
            best_epoch=metric_info["best_epoch"],
            training_time=training_time,
            tpe=tpe,
            esr=metric_info["esr"],
            mrstft=metric_info["mrstft"],
            mse=metric_info["mse"],
            capture_efficiency=capture_efficiency,
            quality_weighted_efficiency=quality_weighted_efficiency,
            learning_rate=self._saved_or_path(
                saved_info, "learning_rate", path_info["learning_rate"]
            ),
            decay=self._saved_or_path(saved_info, "lr_decay", path_info["decay"]),
            scheduler=scheduler,
            run_path=str(run_dir),
        )

    def _parse_lightning_run_path(self, run_dir: _Path) -> _Dict[str, str]:
        parts = [part.lower() for part in run_dir.parts]
        architecture_values = {architecture.value for architecture in _core.Architecture}
        architecture_index = next(
            (i for i, part in enumerate(parts) if part in architecture_values), None
        )
        file_name = run_dir.parent.name
        stage = "single_stage"
        architecture = ""
        epochs = ""
        if architecture_index is not None:
            architecture = run_dir.parts[architecture_index]
            if architecture_index > 0:
                previous_part = run_dir.parts[architecture_index - 1]
                if previous_part.startswith("stage_"):
                    stage = previous_part
                    if architecture_index > 1:
                        file_name = run_dir.parts[architecture_index - 2]
                else:
                    file_name = previous_part
            if architecture_index + 1 < len(run_dir.parts):
                next_part = run_dir.parts[architecture_index + 1]
                if next_part.isdigit():
                    epochs = next_part
        lr_match = _re.search(
            r"(?:^|_)lr(?P<lr>[^_]+)_lrdecay(?P<decay>[^_]+)",
            run_dir.name,
            flags=_re.IGNORECASE,
        )
        return {
            "file_name": file_name,
            "stage": stage,
            "architecture": architecture,
            "epochs": epochs,
            "learning_rate": lr_match.group("lr") if lr_match else "",
            "decay": lr_match.group("decay") if lr_match else "",
        }

    def _extract_lightning_metrics(
        self, event_files: _Sequence[_Path]
    ) -> _Dict[str, str]:
        latest_event_file = max(event_files, key=lambda path: path.stat().st_mtime)
        accumulator = _EventAccumulator(str(latest_event_file))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))

        def scalars(tag: str):
            return accumulator.Scalars(tag) if tag in scalar_tags else []

        scalar_series = {
            "val_loss": scalars("val_loss"),
            "epoch": scalars("epoch"),
            "tpe": scalars("TPE"),
            "esr": scalars("ESR"),
            "esrpreemph": scalars("ESRPREEMPH"),
            "mse": scalars("MSE"),
            "mrstft": (
                scalars("MRSTFTPREEMPH")
                or scalars("Pre-emphasized MRSTFT")
                or scalars("MRSTFT")
            ),
        }

        reference_series = (
            scalar_series["val_loss"]
            or scalar_series["esr"]
            or scalar_series["esrpreemph"]
            or scalar_series["mse"]
            or scalar_series["mrstft"]
        )
        if len(reference_series) == 0:
            return {
                "epochs": "",
                "best_epoch": "",
                "training_time": "",
                "tpe": "",
                "esr": "",
                "mrstft": "",
                "mse": "",
            }

        best_index = min(
            range(len(reference_series)), key=lambda index: reference_series[index].value
        )
        best_reference = reference_series[best_index]
        epoch_scalar = self._series_value(scalar_series["epoch"], best_index)
        epoch_value = epoch_scalar.value if epoch_scalar is not None else best_reference.step
        max_epoch_value = epoch_value
        if len(scalar_series["epoch"]) > 0:
            max_epoch_value = scalar_series["epoch"][-1].value
        elif len(reference_series) > 0:
            max_epoch_value = reference_series[-1].step

        return {
            "epochs": self._format_epoch(max_epoch_value),
            "best_epoch": self._format_epoch(epoch_value),
            "training_time": self._format_duration(
                reference_series[-1].wall_time - reference_series[0].wall_time
            ),
            "tpe": self._format_metric(self._series_mean_value(scalar_series["tpe"])),
            "esr": self._format_metric(self._series_numeric_value(scalar_series["esr"], best_index)),
            "mrstft": self._format_metric(
                self._series_numeric_value(scalar_series["mrstft"], best_index)
            ),
            "mse": self._format_metric(self._series_numeric_value(scalar_series["mse"], best_index)),
        }

    @staticmethod
    def _read_saved_analysis_info(run_dir: _Path) -> _Optional[_Dict[str, _Any]]:
        info_path = run_dir / "analysis_info.json"
        if not info_path.is_file():
            return None
        try:
            with open(info_path, "r", encoding="utf-8") as fp:
                data = _json.load(fp)
        except (OSError, ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _infer_scheduler(self, run_dir: _Path) -> str:
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            return ""
        checkpoint_paths = list(checkpoints_dir.glob("*.ckpt"))
        if len(checkpoint_paths) == 0:
            return ""
        checkpoint_path = max(checkpoint_paths, key=lambda path: path.stat().st_mtime)
        try:
            checkpoint = self._load_checkpoint(checkpoint_path)
        except Exception:
            return ""
        schedulers = checkpoint.get("lr_schedulers") or []
        if len(schedulers) == 0:
            return ""
        if len(schedulers) > 1 and isinstance(schedulers[1], dict):
            if "factor" in schedulers[1] or "patience" in schedulers[1]:
                return "Linear Warmup + ReduceLROnPlateau"
        scheduler_state = schedulers[0]
        if not isinstance(scheduler_state, dict):
            return ""
        if "gamma" in scheduler_state:
            return "ExponentialLR"
        if "T_max" in scheduler_state:
            return "CosineAnnealingLR"
        if "_milestones" in scheduler_state or "_schedulers" in scheduler_state:
            return "Warmup + Cosine Decay"
        if "_schedule_phases" in scheduler_state:
            return "OneCycleLR"
        if "factor" in scheduler_state or "patience" in scheduler_state:
            return "ReduceLROnPlateau"
        if "T_0" in scheduler_state or "T_mult" in scheduler_state:
            return "CosineAnnealingWarmRestarts"
        for key in ("name", "class", "_class_name"):
            if key in scheduler_state and scheduler_state[key]:
                return str(scheduler_state[key])
        return ""

    @staticmethod
    def _load_checkpoint(checkpoint_path: _Path):
        try:
            return _torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
        except TypeError:
            return _torch.load(str(checkpoint_path), map_location="cpu")

    @staticmethod
    def _series_value(series, index: int):
        if len(series) == 0:
            return None
        return series[min(index, len(series) - 1)]

    @classmethod
    def _series_numeric_value(cls, series, index: int) -> _Optional[float]:
        value = cls._series_value(series, index)
        return None if value is None else float(value.value)

    @staticmethod
    def _series_mean_value(series) -> _Optional[float]:
        if len(series) == 0:
            return None
        return sum(float(item.value) for item in series) / len(series)

    @staticmethod
    def _format_metric(value: _Optional[float]) -> str:
        if value is None:
            return ""
        if abs(value) >= 0.001:
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:.3e}"

    @staticmethod
    def _format_epoch(value: _Optional[float]) -> str:
        if value is None:
            return ""
        return str(int(round(float(value))))

    @staticmethod
    def _format_duration(value: _Optional[_Any]) -> str:
        if value in (None, ""):
            return ""
        try:
            total_seconds = max(0, int(round(float(value))))
        except (TypeError, ValueError):
            return ""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _string_or_blank(value: _Any) -> str:
        return "" if value in (None, "") else str(value)

    @classmethod
    def _saved_or_path(
        cls, saved_info: _Optional[_Dict[str, _Any]], key: str, fallback: str
    ) -> str:
        if saved_info is None:
            return fallback
        return cls._string_or_blank(saved_info.get(key)) or fallback

    @classmethod
    def _compute_efficiency_scores(
        cls, esr: str, training_time: str
    ) -> _Tuple[str, str]:
        try:
            esr_value = float(esr)
        except (TypeError, ValueError):
            return "", ""
        training_seconds = _LightningAnalysisModal._duration_to_seconds(training_time)
        if training_seconds in (None, 0) or esr_value <= 0.0:
            return "", ""
        minutes = training_seconds / 60.0
        capture_efficiency = 1.0 / (esr_value * minutes)
        quality_weighted_efficiency = 1.0 / ((esr_value ** 1.5) * minutes)
        return (
            cls._format_metric(capture_efficiency),
            cls._format_metric(quality_weighted_efficiency),
        )

    @staticmethod
    def _sort_key_for_number(value: str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def _get_additional_options_frame(self):
        # Checkboxes
        # TODO get these definitions into __init__()
        _divider(self._frame_main).pack(fill=_tk.X, pady=(14, 0))
        self._frame_checkboxes = _tk.Frame(self._frame_main, bg=_GUI_BG)
        self._frame_checkboxes.pack(anchor="w", fill=_tk.X, pady=(10, 10))
        _eyebrow(self._frame_checkboxes, "Options").grid(
            row=0, column=0, sticky="w", padx=(0, 22)
        )

        def make_checkbox(
            key: _CheckboxKeys, text: str, default_value: bool
        ) -> Checkbox:
            variable = _tk.BooleanVar()
            variable.set(self._saved_checkbox_values.get(key.value, default_value))
            check_button = _FlatCheckbutton(
                self._frame_checkboxes,
                text=text,
                variable=variable,
            )
            self._checkboxes[key] = Checkbox(variable, check_button)
            self._widgets[key] = check_button  # For tracking in set-all-widgets ops
            variable.trace_add("write", lambda *_args: self._persist_gui_state())

        self._checkboxes: _Dict[_CheckboxKeys, Checkbox] = dict()
        make_checkbox(
            _CheckboxKeys.SILENT_TRAINING,
            "Silent run (suggested for batch training)",
            False,
        )
        make_checkbox(_CheckboxKeys.SAVE_PLOT, "Save ESR plot automatically", True)
        make_checkbox(
            _CheckboxKeys.ADVANCED_INIT_INFO,
            "Show advanced initialization info",
            False,
        )

        # Grid them:
        for column, v in enumerate(self._checkboxes.values()):
            v.check_button.grid(
                row=0,
                column=column + 1,
                sticky="w",
                padx=(0, 18),
                pady=_BUTTON_STACK_PADY,
            )
            self._frame_checkboxes.grid_columnconfigure(column + 1, weight=0)
        _divider(self._frame_main).pack(fill=_tk.X, pady=(0, 16))

    def mainloop(self):
        self._root.mainloop()

    def _disable(self):
        self._set_all_widget_states_to(_tk.DISABLED)

    def _open_advanced_options(self):
        """
        Open window for advanced options
        """

        existing = getattr(self, "_advanced_options_window", None)
        if existing is not None:
            try:
                existing.lift()
                existing.focus_force()
                return
            except _tk.TclError:
                self._advanced_options_window = None
        AdvancedOptionsGUI(self._check_button_states, self)

    def _open_metadata(self):
        """
        Open window for metadata
        """

        self._wait_while_func(lambda resume: UserMetadataGUI(resume, self))

    def _show_update_modal_if_update_available(self):
        class UpdateInfo(_NamedTuple):
            available: bool
            current_version: _Version
            new_version: _Optional[_Version]

        def get_info() -> UpdateInfo:
            current_version = _get_current_version()
            latest_version = _get_latest_version_from_github()
            update_available = (
                latest_version is not None and latest_version > current_version
            )
            return UpdateInfo(
                available=update_available,
                current_version=current_version,
                new_version=latest_version,
            )

        update_info = get_info()
        if not update_info.available or update_info.new_version is None:  # No news
            return
        # Now figure out what we've seen in the past
        update_settings = _settings.get_update_settings()

        settings_version = (
            _Version.from_string(update_settings["newest_available_version"])
            if update_settings["newest_available_version"] is not None
            else None
        )
        # Different new version since we last checked
        if settings_version is None or update_info.new_version > settings_version:
            _settings.set_update_settings(
                newest_available_version=str(update_info.new_version),
                never_show_again=False,
            )
            update_settings = _settings.get_update_settings()
        if update_settings["never_show_again"]:
            return
        else:
            self._wait_while_func(
                lambda resume: _UpdateAvailableModal(
                    resume, str(update_info.new_version)
                ),
            )

    def _resume(self):
        self._set_all_widget_states_to(_tk.NORMAL)
        self._check_button_states()

    def _set_all_widget_states_to(self, state):
        for widget in self._widgets.values():
            widget["state"] = state

    # TODO: Rename this function to "validate_input_audio_files" or something similar.
    def _train(self):
        # TODO: Replace every single instance of the use of "input" and "output", as well as "x" and "y", with "dry_audio" and "wet_audio", or something similar.
        # Original author used "output" to mean "the output of the external effect/amp/whatever", *not* the output of the model training process!
        job = self._create_training_job()
        if job is not None:
            self._validate_and_submit_training_job(job, start_immediately=True)

    def _train2(self, ignore_checks=False):
        job = self._create_training_job(ignore_checks=ignore_checks)
        if job is not None:
            self._submit_training_job(job, start_immediately=True)

    def _run_training_job(self, job: _TrainingJob) -> _Tuple[bool, str, str]:
        input_path = job.input_path
        options = job.advanced_options
        user_metadata = (
            job.user_metadata if job.user_metadata_flag else _UserMetadata()
        )
        success_results = []
        failure_count = 0
        # Run it
        for file in job.output_paths:
            self._post_to_gui(
                lambda job=job, file=file: self._set_current_training_file(job, file)
            )
            print(f"Now training {file}")
            basename = _job_export_basename(job, file)

            try:
                train_output = _core.train(
                    input_path,
                    file,
                    job.training_destination,
                    epochs=options.num_epochs,
                    lr=options.lr,
                    lr_decay=options.lr_decay,
                    lr_scheduler_type=options.lr_scheduler,
                    batch_size=options.batch_size,
                    ny=options.ny,
                    latency=options.latency,
                    architecture=options.architecture,
                    silent=job.checkbox_values.get(
                        _CheckboxKeys.SILENT_TRAINING, False
                    ),
                    save_plot=job.checkbox_values.get(_CheckboxKeys.SAVE_PLOT, True),
                    modelname=basename,
                    ignore_checks=job.ignore_checks,
                    local=True,
                    fit_mrstft=self.get_mrstft_fit(),
                    threshold_esr=options.threshold_esr,
                    user_metadata=user_metadata,
                    stage_mode=options.stage_mode,
                    stage2_epochs=options.stage2_epochs,
                    stage2_lr=options.stage2_lr,
                    stage2_lr_decay=options.stage2_lr_decay,
                    stage2_lr_scheduler_type=options.stage2_lr_scheduler,
                    stage2_focus=options.stage2_focus,
                    checkpoint_save_mode=options.checkpoint_save_mode,
                    checkpoint_path=job.resume_checkpoint_path,
                    show_ignore_checks_warning=not job.ignore_checks,
                    show_advanced_init_info=job.checkbox_values.get(
                        _CheckboxKeys.ADVANCED_INIT_INFO, False
                    ),
                    **self.core_train_kwargs(),
                )
            except Exception as exc:
                print(f"Training failed for {file}: {exc}")
                failure_count += 1
                continue

            if train_output is None:
                print("Model training failed! Skip exporting...")
                failure_count += 1
                continue
            if train_output.aborted:
                print("Training aborted. Returning to the trainer window.")
                return True, _SCHEDULE_STATUS_ABORTED, "aborted"
            if train_output.model is None:
                print("Model training failed! Skip exporting...")
                failure_count += 1
                continue
            print("Model training complete!")
            print("Exporting...")
            outdir = job.training_destination
            print(f"Exporting trained model to {outdir}...")
            try:
                train_output.model.net.export(
                    outdir,
                    basename=basename,
                    user_metadata=user_metadata,
                    other_metadata={
                        _metadata.TRAINING_KEY: train_output.metadata.model_dump()
                    },
                )
            except Exception as exc:
                print(f"Export failed for {file}: {exc}")
                failure_count += 1
                continue
            success_results.append(
                self._format_training_result_cell(train_output, options.architecture)
            )
            print("Done!")

        # Metadata was only valid for 1 run (possibly a batch), so make sure it's not
        # used again unless the user re-visits the window and clicks "ok".
        #self.user_metadata_flag = False
        # It's annoying to have to remember to do this every time I change a setting, because the user metadata gets *silently ignored*, so I've commented this out.
        if success_results:
            result = " / ".join(success_results)
            if failure_count > 0:
                result = f"partial: {result}"
            return False, _SCHEDULE_STATUS_FINISHED, result
        if failure_count > 0:
            return False, _SCHEDULE_STATUS_FAILED, "failed"
        return False, _SCHEDULE_STATUS_FAILED, "no output"

    def _validate_all_data(
        self,
        input_path: _Path,
        output_paths: _Sequence[_Path],
        advanced_options: _Optional[AdvancedOptions] = None,
        on_ignore: _Optional[_Callable[[], None]] = None,
        on_exclude: _Optional[_Callable[[_Sequence[str]], None]] = None,
    ) -> bool:
        """
        Validate all the data.
        If something doesn't pass, then alert the user and ask them whether they
        want to continue.

        :return: whether we passed (NOTE: Training in spite of failure is
            triggered by a modal that is produced on failure.)
        """

        def make_message_for_file(
            output_path: str, validation_output: _core.DataValidationOutput
        ) -> str:
            """
            State the file and explain what's wrong with it.
            """
            # TODO put this closer to what it looks at, i.e. core.DataValidationOutput
            msg = (
                f"\t{_Path(output_path).name}:\n"  # They all have the same directory so
            )
            if not validation_output.sample_rate.passed:
                msg += (
                    "\t\t There are different sample rates for the input ("
                    f"{validation_output.sample_rate.input}) and output ("
                    f"{validation_output.sample_rate.output}).\n"
                )
            if not validation_output.length.passed:
                msg += (
                    "\t\t* The input and output audio files are too different in length"
                )
                if validation_output.length.delta_seconds > 0:
                    msg += (
                        f" (the output is {validation_output.length.delta_seconds:.2f} "
                        "seconds longer than the input)\n"
                    )
                else:
                    msg += (
                        f" (the output is {-validation_output.length.delta_seconds:.2f}"
                        " seconds shorter than the input)\n"
                    )
            if validation_output.latency.manual is None:
                if validation_output.latency.calibration.warnings.matches_lookahead:
                    msg += (
                        "\t\t* The calibrated latency is the maximum allowed. This is "
                        "probably because the latency calibration was triggered by noise.\n"
                    )
                if validation_output.latency.calibration.warnings.disagreement_too_high:
                    msg += "\t\t* The calculated latencies are too different from each other.\n"
            if not validation_output.checks.passed:
                check_version = validation_output.checks.version
                if check_version in (2, 3, 5, 6, 7):
                    msg += (
                        "\t\t* The clip was flagged as faulty because the "
                        "self-ESR/repeatability check is too high.\n"
                    )
                else:
                    msg += "\t\t* A data consistency check failed.\n"
            if not validation_output.pytorch.passed:
                msg += "\t\t* PyTorch data set errors:\n"
                for split in _Split:
                    split_validation = getattr(validation_output.pytorch, split.value)
                    if not split_validation.passed:
                        msg += f"   * {split.value:10s}: {split_validation.msg}\n"
            return msg

        # Validate input
        input_validation = _core.validate_input(input_path)
        if not input_validation.passed:
            self._wait_while_func(
                (lambda resume, *args, **kwargs: _OkModal(resume, *args, **kwargs)),
                f"Input file {input_path} is not recognized as a standardized input "
                "file.\nTraining cannot proceed.",
            )
            return False

        options = self.advanced_options if advanced_options is None else advanced_options
        user_latency = options.latency
        file_validation_outputs = {
            output_path: _core.validate_data(
                input_path,
                output_path,
                user_latency,
                num_output_samples_per_datum=options.ny,
            )
            for output_path in output_paths
        }
        if any(not fv.passed for fv in file_validation_outputs.values()):
            failed_output_paths = [
                str(output_path)
                for output_path, fv in file_validation_outputs.items()
                if not fv.passed
            ]
            good_output_paths = [
                str(output_path)
                for output_path, fv in file_validation_outputs.items()
                if fv.passed
            ]
            can_ignore = all(fv.passed_critical for fv in file_validation_outputs.values())
            msg = "The following output files failed checks:\n" + "".join(
                [
                    make_message_for_file(output_path, fv)
                    for output_path, fv in file_validation_outputs.items()
                    if not fv.passed
                ]
            )
            if not can_ignore and len(good_output_paths) == 0:
                msg += "\nCritical errors found, cannot ignore."
                self._wait_while_func(
                    lambda resume, msg, **kwargs: _OkModal(resume, msg, **kwargs),
                    msg=msg,
                    label_kwargs={"justify": "left"},
                )
                return False

            def ignore_failed_clips():
                if on_ignore is None:
                    self._train2(ignore_checks=True)
                else:
                    on_ignore()

            def exclude_failed_clips():
                if on_exclude is not None:
                    on_exclude(good_output_paths)

            self._wait_while_func(
                (
                    lambda resume, *args, **kwargs: _FaultyClipsModal(
                        resume,
                        self._root,
                        *args,
                        **kwargs,
                    )
                ),
                msg=msg,
                failed_count=len(failed_output_paths),
                good_count=len(good_output_paths),
                can_ignore=can_ignore,
                on_exclude=exclude_failed_clips,
                on_ignore=ignore_failed_clips if can_ignore else None,
            )
            return False  # we still failed checks so say so.

        return True

    def _wait_while_func(self, func, *args, **kwargs):
        """
        Disable this GUI while something happens.
        That function _needs_ to call the provided self._resume when it's ready to
        release me!
        """
        self._disable()
        func(self._resume, *args, **kwargs)


# some typing functions
def _non_negative_int(val):
    val = int(val)
    if val < 0:
        val = 0
    return val


def _positive_int(val):
    val = int(val)
    if val < 1:
        val = 1
    return val


def _non_negative_float(val):
    val = float(val)
    if val < 0.0:
        val = 0.0
    return val


class _TypeOrNull(object):
    def __init__(self, T, null_str=""):
        """
        :param T: tpe to cast to on .forward()
        """
        self._T = T
        self._null_str = null_str

    @property
    def null_str(self) -> str:
        """
        What str is displayed when for "None"
        """
        return self._null_str

    def forward(self, val: str):
        val = val.rstrip()
        return None if val == self._null_str else self._T(val)

    def inverse(self, val) -> str:
        return self._null_str if val is None else str(val)


_int_or_null = _TypeOrNull(int)
_float_or_null = _TypeOrNull(float)


def _rstripped_str(val):
    return str(val).rstrip()


class _SettingWidget(_abc.ABC):
    """
    A widget for the user to interact with to set something
    """

    @_abc.abstractmethod
    def get(self):
        pass

    @_abc.abstractmethod
    def set_enabled(self, enabled: bool):
        pass


class LabeledOptionMenu(_SettingWidget):
    """
    Label (left) and radio buttons (right)
    """

    def __init__(
        self,
        frame: _tk.Frame,
        label: str,
        choices: _Enum,
        default: _Optional[_Enum] = None,
        on_change: _Optional[_Callable[[_Enum], None]] = None,
        label_minsize: int = _ADVANCED_OPTIONS_LABEL_MINSIZE,
        display_labels: _Optional[_Dict[_Any, str]] = None,
        choice_order: _Optional[_Sequence[_Enum]] = None,
        flat_dropdown: bool = False,
        visible_items: int = 12,
    ):
        """
        :param command: Called to propagate option selection. Is provided with the
            value corresponding to the radio button selected.
        """
        self._frame = frame
        self._choices = choices
        self._on_change = on_change
        self._flat_dropdown = flat_dropdown
        self._visible_items = visible_items
        self._dropdown = None
        self._state = _tk.NORMAL
        self._display_labels = {}
        if display_labels is not None:
            for key, display in display_labels.items():
                value = key.value if isinstance(key, _Enum) else str(key)
                self._display_labels[value] = display
        self._choice_members = list(choices if choice_order is None else choice_order)
        self._choice_values = [choice.value for choice in self._choice_members]
        self._value_to_display = {
            value: self._display_labels.get(value, value) for value in self._choice_values
        }
        self._display_to_value = {
            display: value for value, display in self._value_to_display.items()
        }
        self._choice_displays = [
            self._display_for_value(choice.value) for choice in self._choice_members
        ]
        frame.columnconfigure(0, minsize=label_minsize)
        frame.columnconfigure(1, weight=1)
        self._label = _ttk.Label(
            frame,
            anchor="w",
            text=label,
            style="Muted.TLabel",
        )
        self._label.grid(row=0, column=0, sticky="w", padx=(0, 14))

        self._selected_value = None
        default = (self._choice_members[0] if default is None else default).value
        self._value_var = _tk.StringVar(
            master=frame, value=self._display_for_value(default), name=label
        )
        if self._flat_dropdown:
            self._build_flat_dropdown(frame)
        else:
            self._menu = _ttk.Combobox(
                frame,
                textvariable=self._value_var,
                values=self._choice_displays,
                state="readonly",
                font=_FT_BODY,
                style="TCombobox",
                postcommand=lambda: _style_combobox_popdown(self._menu),
            )
            self._menu.bind(
                "<<ComboboxSelected>>", lambda _event: self._set(self._value_var.get())
            )
        self._menu.grid(row=0, column=1, sticky="ew")
        # Initialize
        self._set(default)
        if self._flat_dropdown:
            self._recolor_flat_dropdown()

    def get(self) -> _Enum:
        return self._selected_value

    @property
    def label(self) -> _tk.Label:
        return self._label

    def set(self, value):
        value = value.value if isinstance(value, _Enum) else str(value)
        self._value_var.set(self._display_for_value(value))
        self._set(value)

    def _display_for_value(self, value: str) -> str:
        return self._value_to_display.get(value, value)

    def _value_for_display(self, display: str) -> str:
        return self._display_to_value.get(display, display)

    def _build_flat_dropdown(self, frame: _tk.Frame):
        self._menu = _tk.Frame(
            frame,
            bg=_GUI_DIVIDER,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self._menu_inner = _tk.Frame(
            self._menu,
            bg=_GUI_SURFACE,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self._menu_inner.pack(fill=_tk.BOTH, expand=True, padx=1, pady=1)
        self._menu_value = _tk.Label(
            self._menu_inner,
            textvariable=self._value_var,
            bg=_GUI_SURFACE,
            fg=_GUI_TEXT,
            font=_FT_BODY,
            anchor="w",
            padx=7,
            pady=5,
            cursor="hand2",
        )
        self._menu_value.pack(side=_tk.LEFT, fill=_tk.X, expand=True)
        self._menu_arrow = _tk.Label(
            self._menu_inner,
            text="v",
            bg=_GUI_SURFACE,
            fg=_GUI_MUTED,
            font=_FT_BTN,
            padx=8,
            pady=5,
            cursor="hand2",
        )
        self._menu_arrow.pack(side=_tk.RIGHT)
        for widget in (
            self._menu,
            self._menu_inner,
            self._menu_value,
            self._menu_arrow,
        ):
            widget.bind("<Button-1>", self._toggle_flat_dropdown)

    def _toggle_flat_dropdown(self, _event=None):
        if self._state == _tk.DISABLED:
            return "break"
        if self._dropdown is not None:
            self._close_flat_dropdown()
        else:
            self._open_flat_dropdown()
        return "break"

    def _open_flat_dropdown(self):
        self._close_flat_dropdown()
        parent = self._frame.winfo_toplevel()
        dropdown = _tk.Toplevel(parent)
        self._dropdown = dropdown
        dropdown.withdraw()
        dropdown.overrideredirect(True)
        dropdown.transient(parent)
        dropdown.configure(bg=_GUI_DIVIDER)

        list_frame = _tk.Frame(dropdown, bg=_GUI_SURFACE)
        list_frame.pack(fill=_tk.BOTH, expand=True, padx=1, pady=1)
        listbox = _tk.Listbox(
            list_frame,
            bg=_GUI_SURFACE,
            fg=_GUI_TEXT,
            selectbackground=_GUI_ACCENT,
            selectforeground=_GUI_ACCENT_TEXT,
            exportselection=False,
            relief="flat",
            highlightthickness=0,
            borderwidth=0,
            font=_FT_BODY,
            activestyle="none",
            height=min(self._visible_items, len(self._choice_displays)),
        )
        listbox.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True, padx=(10, 0), pady=8)
        for display in self._choice_displays:
            listbox.insert(_tk.END, display)
        try:
            selected_index = self._choice_displays.index(self._value_var.get())
        except ValueError:
            selected_index = 0
        listbox.selection_set(selected_index)
        listbox.activate(selected_index)
        listbox.see(selected_index)

        if len(self._choice_displays) > self._visible_items:
            scrollbar = _FlatVerticalScrollbar(list_frame, command=listbox.yview)
            scrollbar.pack(side=_tk.RIGHT, fill=_tk.Y, pady=8, padx=(0, 8))
            listbox.configure(yscrollcommand=scrollbar.set)

        def select_current(_event=None):
            selection = listbox.curselection()
            if selection:
                display = listbox.get(selection[0])
                self._value_var.set(display)
            self._close_flat_dropdown()
            self._restore_parent_interaction()
            if selection:
                self._set(display)
            return "break"

        def scroll_list(event):
            delta = -1 if event.delta > 0 else 1
            listbox.yview_scroll(delta, "units")
            return "break"

        listbox.bind("<ButtonRelease-1>", select_current)
        listbox.bind("<Return>", select_current)
        listbox.bind("<Escape>", lambda _event: self._close_flat_dropdown())
        listbox.bind("<MouseWheel>", scroll_list)
        def close_if_focus_left():
            try:
                focus = dropdown.focus_get()
            except _tk.TclError:
                return
            if focus is None or not str(focus).startswith(str(dropdown)):
                self._close_flat_dropdown()

        dropdown.bind("<Escape>", lambda _event: self._close_flat_dropdown())
        dropdown.bind(
            "<FocusOut>",
            lambda _event: dropdown.after(120, close_if_focus_left),
        )

        self._menu.update_idletasks()
        dropdown.update_idletasks()
        x = self._menu.winfo_rootx()
        y = self._menu.winfo_rooty() + self._menu.winfo_height()
        width = max(self._menu.winfo_width(), dropdown.winfo_reqwidth())
        height = dropdown.winfo_reqheight()
        screen_h = dropdown.winfo_screenheight()
        if y + height > screen_h - 8:
            y = max(0, self._menu.winfo_rooty() - height)
        dropdown.geometry(f"{width}x{height}+{x}+{y}")
        dropdown.deiconify()
        dropdown.lift(parent)
        listbox.focus_set()

    def _close_flat_dropdown(self):
        dropdown = self._dropdown
        self._dropdown = None
        if dropdown is not None:
            try:
                dropdown.destroy()
            except _tk.TclError:
                pass

    def _restore_parent_interaction(self):
        parent = self._frame.winfo_toplevel()

        def restore():
            try:
                parent.grab_release()
            except _tk.TclError:
                pass
            try:
                parent.focus_force()
            except _tk.TclError:
                pass

        try:
            parent.after_idle(restore)
        except _tk.TclError:
            pass

    def _recolor_flat_dropdown(self):
        if not self._flat_dropdown:
            return
        enabled = self._state != _tk.DISABLED
        bg = _GUI_SURFACE if enabled else _GUI_DISABLED_BG
        text = _GUI_TEXT if enabled else _GUI_FAINT
        muted = _GUI_MUTED if enabled else _GUI_FAINT
        cursor = "hand2" if enabled else ""
        for widget in (self._menu, self._menu_inner, self._menu_value, self._menu_arrow):
            widget.configure(cursor=cursor)
        self._menu.configure(bg=_GUI_DIVIDER)
        self._menu_inner.configure(bg=bg)
        self._menu_value.configure(bg=bg, fg=text)
        self._menu_arrow.configure(bg=bg, fg=muted)

    def _set(self, val: str):
        """
        Set the value selected
        """
        val = self._value_for_display(val)
        self._selected_value = self._choices(val)
        if self._on_change is not None:
            self._on_change(self._selected_value)

    def set_enabled(self, enabled: bool):
        state = "readonly" if enabled else "disabled"
        label_fg = _GUI_MUTED if enabled else _GUI_FAINT
        self._label.configure(foreground=label_fg)
        if self._flat_dropdown:
            self._state = _tk.NORMAL if enabled else _tk.DISABLED
            if not enabled:
                self._close_flat_dropdown()
            self._recolor_flat_dropdown()
        else:
            self._menu.configure(
                state=state,
            )


class _Hovertip(Hovertip):
    """
    Adjustments:

    * Always black text (macOS)
    """

    def showcontents(self):
        # Override
        label = _tk.Label(
            self.tipwindow,
            text=self.text,
            justify=_tk.LEFT,
            background="#ffffe0",
            relief=_tk.SOLID,
            borderwidth=1,
            fg="black",
        )
        label.pack()


class LabeledText(_SettingWidget):
    """
    Label (left) and text input (right)
    """

    def __init__(
        self,
        frame: _tk.Frame,
        label: str,
        default=None,
        type=None,
        on_change: _Optional[_Callable[[], None]] = None,
        left_width=_ADVANCED_OPTIONS_LEFT_WIDTH,
        right_width=_ADVANCED_OPTIONS_RIGHT_WIDTH,
        label_minsize: int = _ADVANCED_OPTIONS_LABEL_MINSIZE,
        stretch: bool = False,
    ):
        """
        :param command: Called to propagate option selection. Is provided with the
            value corresponding to the radio button selected.
        :param type: If provided, casts value to given type
        :param left_width: How much space to use on the left side (text)
        :param right_width: How much space for the Text field
        """
        self._frame = frame
        frame.columnconfigure(0, minsize=label_minsize)
        frame.columnconfigure(1, weight=1)
        self._label = _ttk.Label(
            frame,
            anchor="w",
            text=label,
            style="Muted.TLabel",
        )
        self._label.grid(row=0, column=0, sticky="w", padx=(0, 14))

        self._value_var = _tk.StringVar(master=frame)
        self._text = _ttk.Entry(
            frame,
            textvariable=self._value_var,
            font=_FT_MONO,
            style="TEntry",
            width=right_width,
        )
        self._text.grid(row=0, column=1, sticky="ew" if stretch else "w")
        self._text.bind("<ButtonPress-1>", self._focus_text, add="+")
        self._text.bind("<ButtonRelease-1>", self._focus_text, add="+")

        self._type = (lambda x: x) if type is None else type
        self._on_change = on_change

        if default is not None:
            self._value_var.set(str(default))
        if self._on_change is not None:
            self._text.bind("<KeyRelease>", self._notify_change)
            self._text.bind("<FocusOut>", self._notify_change)

        # You can assign a tooltip for the label if you'd like.
        self.label_tooltip: _Optional[_Hovertip] = None

    @property
    def label(self) -> _tk.Label:
        return self._label

    def get(self):
        """
        Attempt to get and return the value.
        May throw a tk.TclError indicating something went wrong getting the value.
        """
        return self._type(self._value_var.get())

    def set(self, value):
        self._value_var.set(str(value))

    def focus_set(self):
        self._text.focus_force()

    def _focus_text(self, _event=None):
        if str(self._text.cget("state")) != _tk.DISABLED:
            self._text.focus_force()
            self._text.after_idle(self._text.focus_force)

    def _notify_change(self, _event=None):
        if self._on_change is not None:
            self._on_change()

    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        label_fg = _GUI_MUTED if enabled else _GUI_FAINT
        self._label.configure(foreground=label_fg)
        self._text.configure(
            state=state,
        )


class AdvancedOptionsGUI(object):
    """
    A window to hold advanced options (Architecture and number of epochs)
    """

    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._suspend_recommendations = False
        self._user_edited_stage1_lr_decay = False
        self._user_edited_stage2_lr_decay = False

        def resume_and_forget():
            self._parent._advanced_options_window = None
            self._parent._advanced_options_controller = None
            resume_main()

        self._root = _TopLevelWithOk(self.apply, resume_and_forget)
        self._root.withdraw()
        self._parent._advanced_options_window = self._root
        self._parent._advanced_options_controller = self
        self._root.title("Advanced Options")
        _apply_gui_theme(self._root, self._parent._P)
        advanced_width = 460
        advanced_height = 920
        _style_toplevel(self._root, width=advanced_width, height=advanced_height)
        self._root.transient(self._parent._root)
        self._root.protocol("WM_DELETE_WINDOW", self._root.destroy)
        self._auto_fill_recommendations_var = _tk.BooleanVar(
            master=self._root,
            value=bool(self._parent.advanced_options.auto_fill_recommendations),
        )
        self._body = _ttk.Frame(self._root, style="TFrame")
        self._body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))

        self.pack()

        self._frame_actions = _ttk.Frame(self._body, style="TFrame")
        self._frame_actions.pack(fill=_tk.X, pady=(10, 0))
        footer_button_width = 14
        self._button_save_preset = _ttk.Button(
            self._frame_actions,
            text="Save Preset...",
            command=self._save_preset,
            style="Ghost.TButton",
            width=footer_button_width,
        )
        self._button_save_preset.pack(side=_tk.LEFT, padx=(0, 8))
        self._button_load_preset = _ttk.Button(
            self._frame_actions,
            text="Load Preset...",
            command=self._load_preset,
            style="Ghost.TButton",
            width=footer_button_width,
        )
        self._button_load_preset.pack(side=_tk.LEFT, padx=(0, 8))
        self._button_ok = _ttk.Button(
            self._frame_actions,
            text="OK",
            command=lambda: self._root.destroy(pressed_ok=True),
            style="Primary.TButton",
            width=footer_button_width,
        )
        self._button_ok.pack(side=_tk.LEFT)
        _position_toplevel_right_of_parent(
            self._root, self._parent._root, advanced_width, advanced_height
        )
        self._root.deiconify()
        self._root.lift()
        self._root.focus_force()
        self._root.after_idle(self._focus_initial_field)

    def apply(self):
        """
        Set values to parent and destroy this object
        """

        def safe_apply(name):
            try:
                setattr(
                    self._parent.advanced_options, name, getattr(self, "_" + name).get()
                )
            except ValueError:
                pass

        # TODO could clean up more / see `.pack_options()`
        for name in _ADVANCED_OPTIONS_WIDGET_FIELDS:
            safe_apply(name)
        self._parent.advanced_options.auto_fill_recommendations = bool(
            self._auto_fill_recommendations_var.get()
        )
        self._parent._persist_gui_state()

    def _collect_current_advanced_options(self) -> AdvancedOptions:
        return AdvancedOptions(
            architecture=self._architecture.get(),
            num_epochs=self._num_epochs.get(),
            lr=self._lr.get(),
            lr_decay=self._lr_decay.get(),
            lr_scheduler=self._lr_scheduler.get(),
            batch_size=self._batch_size.get(),
            ny=self._ny.get(),
            latency=self._latency.get(),
            ignore_checks=self._parent.advanced_options.ignore_checks,
            threshold_esr=self._threshold_esr.get(),
            stage_mode=self._stage_mode.get(),
            stage2_epochs=self._stage2_epochs.get(),
            stage2_lr=self._stage2_lr.get(),
            stage2_lr_decay=self._stage2_lr_decay.get(),
            stage2_lr_scheduler=self._stage2_lr_scheduler.get(),
            stage2_focus=self._stage2_focus.get(),
            checkpoint_save_mode=self._checkpoint_save_mode.get(),
            auto_fill_recommendations=bool(self._auto_fill_recommendations_var.get()),
        )

    def _set_widgets_from_advanced_options(self, options: AdvancedOptions):
        self._suspend_recommendations = True
        self._user_edited_stage1_lr_decay = False
        self._user_edited_stage2_lr_decay = False
        try:
            self._architecture.set(options.architecture)
            self._num_epochs.set(options.num_epochs)
            self._lr.set(_fmt_float(options.lr))
            self._lr_decay.set(_fmt_float(options.lr_decay))
            self._lr_scheduler.set(options.lr_scheduler)
            self._batch_size.set(options.batch_size)
            self._ny.set(options.ny)
            self._latency.set(_int_or_null.inverse(options.latency))
            self._threshold_esr.set(_float_or_null.inverse(options.threshold_esr))
            self._stage_mode.set(options.stage_mode)
            self._stage2_epochs.set(options.stage2_epochs)
            self._stage2_lr.set(_fmt_float(options.stage2_lr))
            self._stage2_lr_decay.set(_fmt_float(options.stage2_lr_decay))
            self._stage2_lr_scheduler.set(options.stage2_lr_scheduler)
            self._stage2_focus.set(options.stage2_focus)
            self._checkpoint_save_mode.set(options.checkpoint_save_mode)
            self._auto_fill_recommendations_var.set(
                bool(options.auto_fill_recommendations)
            )
        finally:
            self._suspend_recommendations = False
        self._refresh_scheduler_tooltips()
        self._refresh_stage_mode_ui()

    def _focus_initial_field(self):
        try:
            self._num_epochs.focus_set()
        except _tk.TclError:
            pass

    def _prompt_preset_name(self) -> _Optional[str]:
        dialog = _tk.Toplevel(self._root)
        dialog.title("Save Preset")
        _apply_gui_theme(dialog, self._parent._P)
        preset_width = 440
        preset_height = 230
        _style_toplevel(
            dialog,
            width=preset_width,
            height=preset_height,
            min_width=360,
            min_height=230,
        )
        dialog.transient(self._root)
        _position_toplevel_right_of_parent(
            dialog, self._root, preset_width, preset_height
        )

        result = {"name": None}
        body = _tk.Frame(dialog, bg=_GUI_BG)
        body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))
        _ttk.Label(body, text="Save Preset", style="Display.TLabel").pack(anchor="w")
        _eyebrow(body, "Advanced option preset name", prefix="").pack(
            anchor="w", pady=(2, 14)
        )

        value = _tk.StringVar(master=dialog)
        entry = _ttk.Entry(body, textvariable=value, font=_FT_BODY, style="TEntry")
        entry.pack(fill=_tk.X)
        error = _ttk.Label(body, text="", style="Muted.TLabel")
        error.pack(anchor="w", pady=(8, 14))

        actions = _tk.Frame(body, bg=_GUI_BG)
        actions.pack(fill=_tk.X)

        def close():
            try:
                dialog.destroy()
            except _tk.TclError:
                pass

        def save():
            name = value.get().strip()
            if name == "":
                error.configure(text="Enter a preset name.")
                return
            result["name"] = name
            close()

        _ttk.Button(
            actions, text="Cancel", command=close, style="Ghost.TButton", width=14
        ).pack(side=_tk.LEFT, padx=(0, 8))
        _ttk.Button(
            actions, text="Save", command=save, style="Primary.TButton", width=14
        ).pack(side=_tk.LEFT)
        entry.bind("<Return>", lambda _event: save())
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.grab_set()
        entry.focus_set()
        self._root.wait_window(dialog)
        try:
            self._root.focus_force()
        except _tk.TclError:
            pass
        return result["name"]

    def _confirm_preset_overwrite(self, name: str) -> bool:
        dialog = _tk.Toplevel(self._root)
        dialog.title("Overwrite Preset")
        _apply_gui_theme(dialog, self._parent._P)
        _style_toplevel(dialog, width=460, height=240)
        dialog.transient(self._root)

        result = {"overwrite": False}
        body = _tk.Frame(dialog, bg=_GUI_BG)
        body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))
        _ttk.Label(body, text="Overwrite Preset", style="Display.TLabel").pack(anchor="w")
        _ttk.Label(
            body,
            text=f"A preset named '{name}' already exists.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(14, 18))

        actions = _tk.Frame(body, bg=_GUI_BG)
        actions.pack(fill=_tk.X)

        def close():
            try:
                dialog.destroy()
            except _tk.TclError:
                pass

        def overwrite():
            result["overwrite"] = True
            close()

        _ttk.Button(
            actions, text="Cancel", command=close, style="Ghost.TButton", width=14
        ).pack(side=_tk.LEFT, padx=(0, 8))
        _ttk.Button(
            actions,
            text="Overwrite",
            command=overwrite,
            style="Primary.TButton",
            width=14,
        ).pack(side=_tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.grab_set()
        self._root.wait_window(dialog)
        try:
            self._root.focus_force()
        except _tk.TclError:
            pass
        return bool(result["overwrite"])

    def _show_themed_message(self, title: str, message: str):
        dialog = _tk.Toplevel(self._root)
        dialog.title(title)
        _apply_gui_theme(dialog, self._parent._P)
        wraplength = 360
        estimated_lines = 0
        for raw_line in str(message).splitlines() or [""]:
            estimated_lines += max(1, (len(raw_line) + 49) // 50)
        dialog_width = 430
        dialog_height = max(172, min(300, 142 + estimated_lines * 22))
        _style_toplevel(
            dialog,
            width=dialog_width,
            height=dialog_height,
            min_width=380,
            min_height=dialog_height,
        )
        dialog.transient(self._root)
        _position_toplevel_right_of_parent(
            dialog, self._root, dialog_width, dialog_height
        )

        body = _tk.Frame(dialog, bg=_GUI_BG)
        body.pack(fill=_tk.BOTH, expand=True, padx=22, pady=(18, 16))
        _ttk.Label(body, text=title, style="Display.TLabel").pack(anchor="w")
        _ttk.Label(
            body,
            text=message,
            style="Muted.TLabel",
            wraplength=wraplength,
            justify=_tk.LEFT,
        ).pack(anchor="w", fill=_tk.X, pady=(12, 16))

        actions = _tk.Frame(body, bg=_GUI_BG)
        actions.pack(fill=_tk.X)

        def close():
            try:
                dialog.destroy()
            except _tk.TclError:
                pass

        _ttk.Button(
            actions, text="OK", command=close, style="Primary.TButton", width=14
        ).pack(side=_tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.grab_set()
        self._root.wait_window(dialog)
        try:
            self._root.focus_force()
        except _tk.TclError:
            pass

    def _save_preset(self):
        try:
            options = self._collect_current_advanced_options()
        except Exception as e:
            self._show_themed_message(
                "Save Preset",
                "The current advanced options contain an invalid value and could not "
                f"be saved as a preset.\n\n{e}",
            )
            return

        name = self._prompt_preset_name()
        if name is None:
            return

        presets = _settings.get_advanced_options_presets()
        if name in presets and not self._confirm_preset_overwrite(name):
            return

        _settings.set_advanced_options_preset(
            name, _serialize_advanced_options(options)
        )
        self._show_themed_message(
            "Preset Saved",
            f"Saved advanced options preset '{name}'.",
        )

    def _choose_preset_name(self, preset_names: _Sequence[str]) -> _Optional[str]:
        dialog = _tk.Toplevel(self._root)
        dialog.title("Load Preset")
        _apply_gui_theme(dialog, self._parent._P)
        preset_width = 440
        preset_height = 480
        _style_toplevel(dialog, width=preset_width, height=preset_height)
        dialog.transient(self._root)
        _position_toplevel_right_of_parent(
            dialog, self._root, preset_width, preset_height
        )

        result = {"name": None}
        body = _tk.Frame(dialog, bg=_GUI_BG)
        body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))
        _ttk.Label(body, text="Load Preset", style="Display.TLabel").pack(anchor="w")
        _eyebrow(body, "Saved advanced option presets", prefix="").pack(
            anchor="w", pady=(2, 14)
        )

        list_border = _tk.Frame(body, bg=_GUI_BORDER)
        list_border.pack(fill=_tk.BOTH, expand=True, pady=(0, 16))
        list_frame = _tk.Frame(list_border, bg=_GUI_SURFACE)
        list_frame.pack(fill=_tk.BOTH, expand=True, padx=1, pady=1)
        listbox = _tk.Listbox(
            list_frame,
            bg=_GUI_SURFACE,
            fg=_GUI_TEXT,
            selectbackground=_GUI_ACCENT,
            selectforeground=_GUI_ACCENT_TEXT,
            exportselection=False,
            relief="flat",
            highlightthickness=0,
            borderwidth=0,
            font=_FT_BODY,
            activestyle="none",
        )
        listbox.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True, padx=(10, 0), pady=8)
        scrollbar = _FlatVerticalScrollbar(list_frame, command=listbox.yview)
        scrollbar.pack(side=_tk.RIGHT, fill=_tk.Y, pady=8, padx=(0, 8))
        listbox.configure(yscrollcommand=scrollbar.set)

        for name in preset_names:
            listbox.insert(_tk.END, name)
        if len(preset_names) > 0:
            listbox.selection_set(0)
            listbox.activate(0)

        action_frame = _tk.Frame(body, bg=_GUI_BG)
        action_frame.pack(fill=_tk.X)

        def close():
            try:
                dialog.destroy()
            except _tk.TclError:
                pass

        def load_selected():
            selection = listbox.curselection()
            if not selection:
                return
            result["name"] = listbox.get(selection[0])
            close()

        _ttk.Button(
            action_frame,
            text="Cancel",
            command=close,
            style="Ghost.TButton",
            width=14,
        ).pack(side=_tk.LEFT, padx=(0, 8))
        _ttk.Button(
            action_frame,
            text="Load",
            command=load_selected,
            style="Primary.TButton",
            width=14,
        ).pack(side=_tk.LEFT)

        listbox.bind("<Double-Button-1>", lambda _event: load_selected())
        listbox.bind("<Return>", lambda _event: load_selected())
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.grab_set()
        listbox.focus_set()
        self._root.wait_window(dialog)
        try:
            self._root.focus_force()
        except _tk.TclError:
            pass
        return result["name"]

    def _load_preset(self):
        presets = _settings.get_advanced_options_presets()
        if len(presets) == 0:
            self._show_themed_message(
                "Load Preset",
                "No advanced-options presets have been saved yet.",
            )
            return

        preset_names = sorted(presets)
        name = self._choose_preset_name(preset_names)
        if name is None:
            return
        if name not in presets:
            self._show_themed_message(
                "Load Preset",
                f"Preset '{name}' was not found.",
            )
            return

        preset = presets[name]
        current_auto_fill = bool(self._auto_fill_recommendations_var.get())
        options = _advanced_options_from_dict(
            preset, fallback=self._parent.advanced_options
        )
        if "auto_fill_recommendations" not in preset:
            options.auto_fill_recommendations = current_auto_fill
        self._set_widgets_from_advanced_options(options)

    def _refresh_scheduler_tooltips(self, *_args):
        if not all(
            hasattr(self, name)
            for name in (
                "_lr",
                "_lr_decay",
                "_lr_scheduler",
                "_stage2_lr",
                "_stage2_lr_decay",
                "_stage2_lr_scheduler",
            )
        ):
            return
        lr_scheduler = self._lr_scheduler.get()
        stage2_lr_scheduler = self._stage2_lr_scheduler.get()
        try:
            stage1_epochs = self._num_epochs.get()
        except Exception:
            stage1_epochs = self._parent.advanced_options.num_epochs
        try:
            stage1_lr = self._lr.get()
        except Exception:
            stage1_lr = self._parent.advanced_options.lr
        self._lr.label_tooltip.text = _scheduler_lr_tooltip_text(
            lr_scheduler, "Stage 1"
        )
        self._lr_decay.label_tooltip.text = _scheduler_decay_tooltip_text(
            lr_scheduler, "Stage 1"
        )
        self._stage2_lr.label_tooltip.text = _stage2_lr_tooltip_text(
            stage2_lr_scheduler, stage1_epochs, stage1_lr
        )
        self._stage2_lr_decay.label_tooltip.text = _stage2_decay_tooltip_text(
            stage2_lr_scheduler, stage1_epochs
        )

    def _current_architecture(self) -> _core.Architecture:
        try:
            return self._architecture.get()
        except Exception:
            return self._parent.advanced_options.architecture

    def _auto_fill_recommendations_enabled(self) -> bool:
        try:
            return bool(self._auto_fill_recommendations_var.get())
        except Exception:
            return bool(self._parent.advanced_options.auto_fill_recommendations)

    def _apply_recommended_stage1_lr_decay(self, *, force: bool = False):
        if getattr(self, "_suspend_recommendations", False):
            self._refresh_scheduler_tooltips()
            return
        if not self._auto_fill_recommendations_enabled():
            self._refresh_scheduler_tooltips()
            return
        if self._user_edited_stage1_lr_decay and not force:
            self._refresh_scheduler_tooltips()
            return
        if not all(
            hasattr(self, name)
            for name in ("_architecture", "_num_epochs", "_lr_scheduler", "_lr", "_lr_decay")
        ):
            self._refresh_scheduler_tooltips()
            return
        try:
            epochs = self._num_epochs.get()
        except Exception:
            self._refresh_scheduler_tooltips()
            return
        architecture = self._current_architecture()
        scheduler = self._lr_scheduler.get()
        lr, decay = _recommended_lr_decay_for_scheduler(
            scheduler,
            epochs,
            architecture=architecture,
        )
        self._lr.set(_fmt_float(lr))
        self._lr_decay.set(_fmt_float(decay))
        self._user_edited_stage1_lr_decay = False
        self._refresh_scheduler_tooltips()

    def _apply_recommended_stage2_lr_decay(self, *, force: bool = False):
        if getattr(self, "_suspend_recommendations", False):
            self._refresh_scheduler_tooltips()
            return
        if not self._auto_fill_recommendations_enabled():
            self._refresh_scheduler_tooltips()
            return
        if self._user_edited_stage2_lr_decay and not force:
            self._refresh_scheduler_tooltips()
            return
        if not all(
            hasattr(self, name)
            for name in (
                "_architecture",
                "_stage2_epochs",
                "_stage2_lr_scheduler",
                "_stage2_lr",
                "_stage2_lr_decay",
            )
        ):
            self._refresh_scheduler_tooltips()
            return
        try:
            epochs = self._stage2_epochs.get()
        except Exception:
            self._refresh_scheduler_tooltips()
            return
        architecture = self._current_architecture()
        scheduler = self._stage2_lr_scheduler.get()
        lr, decay = _recommended_lr_decay_for_scheduler(
            scheduler,
            epochs,
            stage_two=True,
            architecture=architecture,
        )
        self._stage2_lr.set(_fmt_float(lr))
        self._stage2_lr_decay.set(_fmt_float(decay))
        self._user_edited_stage2_lr_decay = False
        self._refresh_scheduler_tooltips()

    def _on_architecture_changed(self, *_args):
        self._refresh_core_field_ui()
        self._apply_recommended_stage1_lr_decay(force=True)
        self._apply_recommended_stage2_lr_decay(force=True)

    def _on_stage1_scheduler_changed(self, *_args):
        self._apply_recommended_stage1_lr_decay(force=True)

    def _on_stage2_scheduler_changed(self, *_args):
        self._apply_recommended_stage2_lr_decay(force=True)

    def _on_stage1_epochs_changed(self):
        self._apply_recommended_stage1_lr_decay()

    def _on_stage2_epochs_changed(self):
        self._apply_recommended_stage2_lr_decay()

    def _on_stage1_lr_decay_edited(self):
        self._user_edited_stage1_lr_decay = True
        self._refresh_scheduler_tooltips()

    def _on_stage2_lr_decay_edited(self):
        self._user_edited_stage2_lr_decay = True
        self._refresh_scheduler_tooltips()

    def _refresh_core_field_ui(self):
        if not all(
            hasattr(self, name)
            for name in (
                "_num_epochs",
                "_lr_scheduler",
                "_lr",
                "_lr_decay",
                "_batch_size",
                "_ny",
                "_latency",
                "_threshold_esr",
            )
        ):
            return
        core_training_enabled = True
        if hasattr(self, "_stage_mode"):
            try:
                core_training_enabled = (
                    self._stage_mode.get() != _core.TrainingStageMode.REFINEMENT_ONLY
                )
            except Exception:
                core_training_enabled = True
        for widget in (
            self._num_epochs,
            self._lr_scheduler,
            self._lr,
            self._lr_decay,
        ):
            widget.set_enabled(core_training_enabled)
        for widget in (
            self._batch_size,
            self._ny,
            self._latency,
            self._threshold_esr,
        ):
            widget.set_enabled(True)

    def _refresh_stage_mode_ui(self, *_args):
        if not all(
            hasattr(self, name)
            for name in (
                "_stage_mode",
                "_stage2_epochs",
                "_stage2_lr",
                "_stage2_lr_decay",
                "_stage2_lr_scheduler",
                "_stage2_focus",
            )
        ):
            return
        mode = self._stage_mode.get()
        refinement_enabled = mode in (
            _core.TrainingStageMode.TWO_STAGE,
            _core.TrainingStageMode.REFINEMENT_ONLY,
        )
        for widget in (
            self._stage2_epochs,
            self._stage2_lr,
            self._stage2_lr_decay,
            self._stage2_lr_scheduler,
            self._stage2_focus,
        ):
            widget.set_enabled(refinement_enabled)
        self._refresh_core_field_ui()

    def pack(self):
        parent = getattr(self, "_body", self._root)

        def pack_row(section: _tk.Frame, frame: _tk.Frame):
            frame.pack(fill=_tk.X, pady=_ROW_PADY)

        # TODO things that are `_SettingWidget`s are named carefully, need to make this
        # easier to work with.

        _ttk.Label(
            parent,
            text="Advanced Options",
            style="Display.TLabel",
        ).pack(anchor="w")
        _eyebrow(parent, "Architecture · Schedule · Refinement", prefix="").pack(
            anchor="w", pady=(2, 14)
        )
        self._frame_auto_fill_recommendations = _tk.Frame(parent, bg=_GUI_BG)
        self._frame_auto_fill_recommendations.pack(fill=_tk.X, pady=(0, 14))
        self._auto_fill_recommendations = _FlatCheckbutton(
            self._frame_auto_fill_recommendations,
            text="Auto-fill recommended LR and LR decay values",
            variable=self._auto_fill_recommendations_var,
        )
        self._auto_fill_recommendations.pack(anchor="w")

        _eyebrow(parent, "Core").pack(anchor="w", pady=(0, 4))
        self._frame_core = _ttk.Frame(parent, style="TFrame")
        self._frame_core.pack(fill=_tk.X, pady=(0, 14))

        # Architecture: radio buttons
        self._frame_architecture = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_architecture)
        self._architecture = LabeledOptionMenu(
            self._frame_architecture,
            "Architecture",
            _core.Architecture,
            default=self._parent.advanced_options.architecture,
            on_change=self._on_architecture_changed,
            display_labels=_architecture_display_labels(),
            choice_order=_architecture_choices(),
            flat_dropdown=True,
            visible_items=12,
        )

        # Number of epochs: text box
        self._frame_epochs = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_epochs)
        self._num_epochs = LabeledText(
            self._frame_epochs,
            "Epochs",
            default=str(self._parent.advanced_options.num_epochs),
            type=_non_negative_int,
            on_change=self._on_stage1_epochs_changed,
        )

        self._frame_lr_scheduler = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_lr_scheduler)
        self._lr_scheduler = LabeledOptionMenu(
            self._frame_lr_scheduler,
            "LR scheduler",
            _core.LearningRateScheduler,
            default=self._parent.advanced_options.lr_scheduler,
            on_change=self._on_stage1_scheduler_changed,
            display_labels=_scheduler_display_labels(),
            flat_dropdown=True,
            visible_items=len(list(_core.LearningRateScheduler)),
        )

        # Learning Rate: text box
        self._frame_lr = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_lr)
        self._lr = LabeledText(
            self._frame_lr,
            "Learning rate",
            default=_fmt_float(self._parent.advanced_options.lr),
            type=_non_negative_float,
            on_change=self._on_stage1_lr_decay_edited,
        )
        self._lr.label_tooltip = _Hovertip(
            anchor_widget=self._lr.label,
            text="",
        )

        # Learning Rate Decay: text box
        self._frame_lr_decay = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_lr_decay)
        self._lr_decay = LabeledText(
            self._frame_lr_decay,
            "LR decay",
            default=_fmt_float(self._parent.advanced_options.lr_decay),
            type=_non_negative_float,
            on_change=self._on_stage1_lr_decay_edited,
        )
        self._lr_decay.label_tooltip = _Hovertip(
            anchor_widget=self._lr_decay.label,
            text="",
        )

        # Batch Size: text box
        self._frame_batch_size = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_batch_size)
        self._batch_size = LabeledText(
            self._frame_batch_size,
            "Batch size",
            default=str(self._parent.advanced_options.batch_size),
            type=_non_negative_int
        )

        # NY: text box
        self._frame_ny = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_ny)
        self._ny = LabeledText(
            self._frame_ny,
            "NY",
            default=str(self._parent.advanced_options.ny),
            type=_positive_int,
        )
        self._ny.label_tooltip = _Hovertip(
            anchor_widget=self._ny.label,
            text=(
                "Output samples per training datum. Larger values use more VRAM "
                "and can reduce steps per epoch for long clips."
            ),
        )

        # Delay: text box
        self._frame_latency = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_latency)
        self._latency = LabeledText(
            self._frame_latency,
            "Reamp latency",
            default=_int_or_null.inverse(self._parent.advanced_options.latency),
            type=_int_or_null.forward,
        )

        # Threshold ESR
        self._frame_threshold_esr = _tk.Frame(self._frame_core, bg=_GUI_BG)
        pack_row(self._frame_core, self._frame_threshold_esr)
        self._threshold_esr = LabeledText(
            self._frame_threshold_esr,
            "Threshold ESR",
            default=_float_or_null.inverse(self._parent.advanced_options.threshold_esr),
            type=_float_or_null.forward,
        )

        _eyebrow(parent, "Refinement").pack(anchor="w", pady=(0, 4))
        self._frame_stage2 = _ttk.Frame(parent, style="TFrame")
        self._frame_stage2.pack(fill=_tk.X, pady=(0, 10))

        self._frame_stage_mode = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage_mode)
        self._stage_mode = LabeledOptionMenu(
            self._frame_stage_mode,
            "Training mode",
            _core.TrainingStageMode,
            default=self._parent.advanced_options.stage_mode,
            on_change=self._refresh_stage_mode_ui,
            display_labels={
                _core.TrainingStageMode.SINGLE_STAGE: "Core Only",
                _core.TrainingStageMode.TWO_STAGE: "Core & Refinement",
                _core.TrainingStageMode.REFINEMENT_ONLY: "Refinement Only",
            },
            flat_dropdown=True,
            visible_items=len(list(_core.TrainingStageMode)),
        )

        self._frame_stage2_epochs = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage2_epochs)
        self._stage2_epochs = LabeledText(
            self._frame_stage2_epochs,
            "Refinement epochs",
            default=str(self._parent.advanced_options.stage2_epochs),
            type=_non_negative_int,
            on_change=self._on_stage2_epochs_changed,
        )

        self._frame_stage2_lr_scheduler = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage2_lr_scheduler)
        self._stage2_lr_scheduler = LabeledOptionMenu(
            self._frame_stage2_lr_scheduler,
            "Refinement scheduler",
            _core.LearningRateScheduler,
            default=self._parent.advanced_options.stage2_lr_scheduler,
            on_change=self._on_stage2_scheduler_changed,
            display_labels=_scheduler_display_labels(),
            flat_dropdown=True,
            visible_items=len(list(_core.LearningRateScheduler)),
        )

        self._frame_stage2_lr = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage2_lr)
        self._stage2_lr = LabeledText(
            self._frame_stage2_lr,
            "Refinement LR",
            default=_fmt_float(self._parent.advanced_options.stage2_lr),
            type=_non_negative_float,
            on_change=self._on_stage2_lr_decay_edited,
        )
        self._stage2_lr.label_tooltip = _Hovertip(
            anchor_widget=self._stage2_lr.label,
            text="",
        )

        self._frame_stage2_lr_decay = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage2_lr_decay)
        self._stage2_lr_decay = LabeledText(
            self._frame_stage2_lr_decay,
            "Refinement LR decay",
            default=_fmt_float(self._parent.advanced_options.stage2_lr_decay),
            type=_non_negative_float,
            on_change=self._on_stage2_lr_decay_edited,
        )
        self._stage2_lr_decay.label_tooltip = _Hovertip(
            anchor_widget=self._stage2_lr_decay.label,
            text="",
        )

        self._frame_stage2_focus = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_stage2_focus)
        self._stage2_focus = LabeledOptionMenu(
            self._frame_stage2_focus,
            "Refinement focus",
            _core.StageTwoFocus,
            default=self._parent.advanced_options.stage2_focus,
            flat_dropdown=True,
            visible_items=len(list(_core.StageTwoFocus)),
        )
        self._frame_checkpoint_save_mode = _tk.Frame(self._frame_stage2, bg=_GUI_BG)
        pack_row(self._frame_stage2, self._frame_checkpoint_save_mode)
        self._checkpoint_save_mode = LabeledOptionMenu(
            self._frame_checkpoint_save_mode,
            "Checkpoint saving",
            _core.CheckpointSaveMode,
            default=self._parent.advanced_options.checkpoint_save_mode,
            flat_dropdown=True,
            visible_items=len(list(_core.CheckpointSaveMode)),
        )
        self._refresh_scheduler_tooltips()
        self._refresh_stage_mode_ui()


class UserMetadataGUI(object):
    # Things that are auto-filled:
    # Model date
    # gain
    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._root = _TopLevelWithOk(self.apply, resume_main)
        self._root.title("Metadata")
        _apply_gui_theme(self._root, self._parent._P)
        _style_toplevel(self._root, width=1060, height=520)
        self._body = _tk.Frame(self._root, bg=_GUI_BG)
        self._body.pack(fill=_tk.BOTH, expand=True, padx=24, pady=(20, 18))

        # Pack all the widgets
        self.pack()

        # "Ok": apply and destroy
        self._frame_ok = _tk.Frame(self._body, bg=_GUI_BG)
        self._frame_ok.pack(fill=_tk.X, pady=(18, 24))
        self._button_ok = _ttk.Button(
            self._frame_ok,
            text="OK",
            command=lambda: self._root.destroy(pressed_ok=True),
            style="Primary.TButton",
            width=16,
        )
        self._button_ok.pack(side=_tk.RIGHT)

    def apply(self):
        """
        Set values to parent and destroy this object
        """

        def safe_apply(name):
            try:
                setattr(
                    self._parent.user_metadata, name, getattr(self, "_" + name).get()
                )
            except ValueError:
                pass

        # TODO could clean up more / see `.pack()`
        for name in (
            "name",
            "modeled_by",
            "gear_make",
            "gear_model",
            "gear_type",
            "tone_type",
            "input_level_dbu",
            "output_level_dbu",
        ):
            safe_apply(name)
        self._parent.user_metadata_flag = any(
            getattr(self._parent.user_metadata, name) is not None
            for name in _METADATA_FIELD_NAMES
        )
        self._parent._persist_gui_state()

    def pack(self):
        # TODO things that are `_SettingWidget`s are named carefully, need to make this
        # easier to work with.
        parent_frame = getattr(self, "_body", self._root)

        def pack_row(frame: _tk.Frame):
            frame.pack(fill=_tk.X, pady=_ROW_PADY)

        _tk.Label(
            parent_frame,
            text="Metadata",
            bg=_GUI_BG,
            fg=_GUI_TEXT,
            font=_DISPLAY_FONT,
        ).pack(anchor="w")
        _eyebrow(parent_frame, "Model and capture notes", prefix="").pack(
            anchor="w", pady=(2, 14)
        )

        LabeledText_ = _partial(
            LabeledText,
            left_width=_METADATA_LEFT_WIDTH,
            right_width=_METADATA_RIGHT_WIDTH,
            label_minsize=_METADATA_LABEL_MINSIZE,
            stretch=True,
        )
        parent = self._parent

        # Name
        self._frame_name = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_name)
        self._name = LabeledText_(
            self._frame_name,
            "NAM name",
            default=parent.user_metadata.name,
            type=_rstripped_str,
        )
        # Modeled by
        self._frame_modeled_by = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_modeled_by)
        self._modeled_by = LabeledText_(
            self._frame_modeled_by,
            "Modeled by",
            default=parent.user_metadata.modeled_by,
            type=_rstripped_str,
        )
        # Gear make
        self._frame_gear_make = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_gear_make)
        self._gear_make = LabeledText_(
            self._frame_gear_make,
            "Gear make",
            default=parent.user_metadata.gear_make,
            type=_rstripped_str,
        )
        # Gear model
        self._frame_gear_model = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_gear_model)
        self._gear_model = LabeledText_(
            self._frame_gear_model,
            "Gear model",
            default=parent.user_metadata.gear_model,
            type=_rstripped_str,
        )
        # Calibration: input & output dBu
        self._frame_input_dbu = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_input_dbu)
        self._input_level_dbu = LabeledText_(
            self._frame_input_dbu,
            "Reamp send level (dBu)",
            default=_float_or_null.inverse(parent.user_metadata.input_level_dbu),
            type=_float_or_null.forward,
        )
        self._input_level_dbu.label_tooltip = _Hovertip(
            anchor_widget=self._input_level_dbu.label,
            text=(
                "(Ok to leave blank)\n\n"
                "Play a sine wave with frequency 1kHz and peak amplitude 0dBFS. Use\n"
                "a multimeter to measure the RMS voltage of the signal at the jack\n"
                "that connects to your gear, and convert to dBu.\n"
                "Record the value here."
            ),
        )
        self._frame_output_dbu = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_output_dbu)
        self._output_level_dbu = LabeledText_(
            self._frame_output_dbu,
            "Reamp return level (dBu)",
            default=_float_or_null.inverse(parent.user_metadata.output_level_dbu),
            type=_float_or_null.forward,
        )
        self._output_level_dbu.label_tooltip = _Hovertip(
            anchor_widget=self._output_level_dbu.label,
            text=(
                "(Ok to leave blank)\n\n"
                "Play a sine wave with frequency 1kHz into your interface where\n"
                "you're recording your gear. Keeping the interface's input gain\n"
                "trimmed as you will use it when recording, adjust the sine wave\n"
                "until the input peaks at exactly 0dBFS in your DAW. Measure the RMS\n"
                "voltage and convert to dBu.\n"
                "Record the value here."
            ),
        )
        # Gear type
        self._frame_gear_type = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_gear_type)
        self._gear_type = LabeledOptionMenu(
            self._frame_gear_type,
            "Gear type",
            _GearType,
            default=parent.user_metadata.gear_type,
            label_minsize=_METADATA_LABEL_MINSIZE,
        )
        # Tone type
        self._frame_tone_type = _tk.Frame(parent_frame, bg=_GUI_BG)
        pack_row(self._frame_tone_type)
        self._tone_type = LabeledOptionMenu(
            self._frame_tone_type,
            "Tone type",
            _ToneType,
            default=parent.user_metadata.tone_type,
            label_minsize=_METADATA_LABEL_MINSIZE,
        )


def _install_error():
    window = _tk.Tk()
    _apply_gui_theme(window)
    window.title("ERROR")
    label = _tk.Label(
        window,
        width=45,
        height=2,
        text="The NAM training software has not been installed correctly.",
        bg=_GUI_BG,
        fg=_GUI_TEXT,
        font=_BODY_FONT,
    )
    label.pack(padx=20, pady=(18, 10))
    button = _ttk.Button(
        window,
        text="Quit",
        command=window.destroy,
        style="Primary.TButton",
    )
    button.pack(pady=(0, 18))
    window.mainloop()


def run():
    if _install_is_valid:
        _gui = GUI()
        _gui.mainloop()
        print("Shut down NAM trainer")
    else:
        _install_error()


if __name__ == "__main__":
    run()
