# File: core.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
The core of the "simplified trainer"

Used by the GUI and Colab trainers.
"""

import hashlib as _hashlib
import importlib.resources as _resources
import json as _json
import os as _os
import sys as _sys
import threading as _threading
import time
import tkinter as _tk

from copy import deepcopy as _deepcopy
from enum import Enum as _Enum
from functools import partial as _partial
from pathlib import Path as _Path
from time import time as _time
from typing import (
    Dict as _Dict,
    NamedTuple as _NamedTuple,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import matplotlib.pyplot as _plt
import numpy as _np
import pytorch_lightning as _pl
import torch as _torch

from pydantic import BaseModel as _BaseModel
from pytorch_lightning.utilities.warnings import PossibleUserWarning as _PossibleUserWarning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import Timer
from torch.utils.data import DataLoader as _DataLoader

if _sys.platform == "win32":
    import msvcrt as _msvcrt
else:
    _msvcrt = None

from ..data import (
    AbstractDataset as _AbstractDataset,
    apply_joint_dataset_hooks as _apply_joint_dataset_hooks,
    DataError as _DataError,
    get_joint_dataset_hooks as _get_joint_dataset_hooks,
    Split as _Split,
    init_dataset as _init_dataset,
    wav_to_np as _wav_to_np,
    wav_to_tensor as _wav_to_tensor,
)
from ..models.exportable import Exportable as _Exportable
from ..models.losses import esr as _ESR
from ..models.metadata import UserMetadata as _UserMetadata
from ..util import (
    filter_warnings as _filter_warnings,
    temporary_logging_levels as _temporary_logging_levels,
)
from ._version import PROTEUS_VERSION as _PROTEUS_VERSION, Version as _Version
from .lightning_module import LightningModule as _LightningModule
from .lightning_module import PackedBestCheckpoint as _PackedBestCheckpoint
from .lightning_module import PackedLightningModule as _PackedLightningModule
from .lightning_module import PackedMaskCallback as _PackedMaskCallback
from . import metadata as _metadata


# Force both matrix operations and convolutions to use full 32-bit float Tensor Cores
# using the PyTorch 2.9+ precision API when available.
# https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices
if hasattr(_torch.backends, "cuda") and hasattr(_torch.backends.cuda, "matmul"):
    if hasattr(_torch.backends.cuda.matmul, "fp32_precision"):
        _torch.backends.cuda.matmul.fp32_precision = "ieee"
if hasattr(_torch.backends, "cudnn"):
    if hasattr(_torch.backends.cudnn, "fp32_precision"):
        _torch.backends.cudnn.fp32_precision = "ieee"
    if hasattr(_torch.backends.cudnn, "conv") and hasattr(_torch.backends.cudnn.conv, "fp32_precision"):
        _torch.backends.cudnn.conv.fp32_precision = "ieee"

# NOTE: Additional callbacks are defined here:
# Stop if validation loss does not decrease by at least `min_delta` within `patience` epochs.
# Also, stop if the validation loss is `NaN` or infinite (this usually indicates an error due to a nonsensical value).
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    min_delta=0.00001,
    patience=20,
    check_finite=True,
    verbose=True,
)
# NOTE: This maxes out the depth, so it could be excessively verbose...
rich_model_summary = RichModelSummary(max_depth=-1)
# NOTE: `cpu_stats` really means CPU, GPU, and some other stuff:
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html
# Keep this disabled by default: it logs hundreds of scalars during training and can interrupt CUDA work.
_DEVICE_STATS_MONITOR_ENV = "NAM_TRAINER_DEVICE_STATS"

def _device_stats_monitor_enabled() -> bool:
    return _os.environ.get(_DEVICE_STATS_MONITOR_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
# NOTE: This logs time per training, validation, and test loops to the trainer's callback dictionary.
time_stats_monitor = Timer(duration=None, verbose=True)

# Training using the simplified trainers in NAM is done at 48k.
STANDARD_SAMPLE_RATE = 48_000.0

# Default number of samples per (mini)batch datum to pull from the model output.
# Lots more info here: https://www.thegearpage.net/board/index.php?threads/nam-hyper-accuracy-captures.2543142/post-43368443
#
# I can't remember if these figures were for 32-true or bf16-mixed...
# Also, I can't remember if these figures were with stock Adam or AdamW8bit...
# TODO: Collect more data and publish stats!
# NY = 64436 (2^16) instead of 8192 (2^13), so 8x larger
# batch_size = 32
# Number of steps per epoch becomes 3.
# ULTRA uses about 15-16 GB of VRAM.
# Complex uses about 21 GB of VRAM.
# Standard uses about 8 GB of VRAM.
#
# NY = 32768 (2^15) instead of 8192 (2^13), so 4x larger
# batch_size = 32
# Number of steps per epoch becomes 7.
# ULTRA uses about 8 GB of VRAM.
# Complex uses about 12 GB of VRAM.
# REVyHI uses about 5 GB of VRAM.
# REVxSTD uses about 4 GB of VRAM.
# Standard uses about 4 GB of VRAM.
_NY_DEFAULT = 8192

_CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
_CAB_MRSTFT_PRE_EMPH_COEF = 0.85

_DELAY_CALIBRATION_ABS_THRESHOLD = 0.0003
_DELAY_CALIBRATION_REL_THRESHOLD = 0.001
_DELAY_CALIBRATION_SAFETY_FACTOR = 0


_ANSI_RESET = "\033[0m"
_ANSI_CYAN = "\033[96m"
_ANSI_WHITE = "\033[97m"
_ANSI_PURPLE = "\033[35m"
_ANSI_MAGENTA = "\033[95m"
_ANSI_GREEN = "\033[92m"
_ANSI_ORANGE = "\033[33m"
_ANSI_RED = "\033[91m"


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_ANSI_RESET}"


def _format_replicate_esr(esr_replicate: float) -> str:
    value_color = _ANSI_WHITE if esr_replicate < 0.02 else _ANSI_PURPLE
    return (
        f"{_colorize('Replicate ESR is ', _ANSI_CYAN)}"
        f"{_colorize(f'{esr_replicate:.8f}', value_color)}."
    )


def _format_final_esr(esr: float) -> str:
    if esr < 0.01:
        value_color = _ANSI_GREEN
    elif esr < 0.02:
        value_color = _ANSI_ORANGE
    else:
        value_color = _ANSI_RED
    return (
        f"{_colorize('Error-signal ratio', _ANSI_MAGENTA)} = "
        f"{_colorize(f'{esr:.4g}', value_color)}"
    )


#TODO: Refactor all of these hard-coded values to simple JSON and/or YAML config files.
#TODO: Refactor all of the *data* into simple JSON and/or YAML config files, so it's separate from the *code*.
class Architecture(_Enum):
    ULTRA = "ULTRA"
    COMPLEX = "complex"
    COMPLEXRF300 = "ComplexRF300"
    COMPLEXRF300LITE = "ComplexRF300Lite"
    COMPLEXRF600 = "ComplexRF600"
    COMPLEXRF600LITE = "ComplexRF600Lite"
    XCOMPLEX = "xComplex"
    XCOMPLEX_LITE = "xComplex Lite"
    REVYHI = "revyhi"
    REVYSTD = 'revystd'
    REVYLO = "revylo"
    REVXSTD = "revxstd"
    XSTD = "xstd"
    XSTD3 = "xstd3"
    XHI3 = "xhi3"
    XHV_12 = "xHV_12"
    XHV_16 = "xHV_16"
    XHV_24 = "xHV24"
    XHQ = "XHQ"
    UHQ = "UHQ"
    LSTM_COMPRESSOR_HQ_48X3 = "LSTM Compressor HQ 48x3"
    LSTM_COMPRESSOR_LIGHT_30X3 = "LSTM Compressor Light 30x3"
    LSTM_UHQ = "LSTM UHQ"
    LSTM_COMPRESSOR_UHQ_64X4 = "LSTM UHQ"
    LSTM_TONEX_LIKE_16 = "LSTM TONEX-like 16"
    CAUSAL_CONV_LSTM_TONEX_128_16_2048 = "TONEX-like Causal Conv LSTM 128-16-2048"
    CAUSAL_CONV_LSTM_TONEX_HQ = "Tonex HQ"
    STANDARD = "standard"
    LITE = "lite"
    FEATHER = "feather"
    NANO = "nano"
    NANO64X4 = "Nano64x4"
    NANO125X3 = "Nano125x3"
    DOUBLE = "double"
    XDOUBLE = "xDouble"
    YDOUBLE = "yDouble"
    DOUBLEPLUS = "DoublePlus"
    ACOMPLEX = "aComplex"
    A2_FULL_LITE = "A2 Full+Lite"
    A2_COMPLEX_LITE = "A2 Complex+Lite"
    A2_COMPLEX_REVYLO = "A2 Complex+RevYLo"
    A2_COMPLEX_NANO64X4 = "A2 Complex+Nano64x4"
    A2_COMPLEX_NANO125X3 = "A2 Complex+Nano125x3"
    A2_DOUBLE_LITE = "A2 Double-Lite"
    A2_XDOUBLE_LITE = "A2 xDouble-Lite"
    A2_DOUBLEPLUS_LITE = "A2 DoublePlus-Lite"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and value == "LSTM Compressor UHQ 64x4":
            return cls.LSTM_UHQ
        return None


class TrainingStageMode(_Enum):
    SINGLE_STAGE = "single_stage"
    TWO_STAGE = "two_stage"
    REFINEMENT_ONLY = "refinement_only"


class LearningRateScheduler(_Enum):
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    WARMUP_COSINE_DECAY = "warmup_cosine_decay"
    ONE_CYCLE = "one_cycle"
    LINEAR_WARMUP_REDUCE_ON_PLATEAU = "linear_warmup_reduce_on_plateau"


class CheckpointSaveMode(_Enum):
    MINIMAL = "Minimal"
    MAXIMUM = "Maximal"
    FOUR_BEST_PLUS_CURRENT = "4 Best + Current"


class StageTwoFocus(_Enum):
    LOW = "Lows"
    MID = "Mids"
    HIGH = "Highs"


_LSTM_ONLY_ARCHITECTURES = {
    Architecture.LSTM_COMPRESSOR_HQ_48X3,
    Architecture.LSTM_COMPRESSOR_LIGHT_30X3,
    Architecture.LSTM_UHQ,
    Architecture.LSTM_TONEX_LIKE_16,
}

_CAUSAL_CONV_LSTM_ARCHITECTURES = {
    Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048,
    Architecture.CAUSAL_CONV_LSTM_TONEX_HQ,
}


def _is_lstm_only_architecture(architecture: _Union[Architecture, str]) -> bool:
    return Architecture(architecture) in _LSTM_ONLY_ARCHITECTURES


def _is_causal_conv_lstm_architecture(architecture: _Union[Architecture, str]) -> bool:
    return Architecture(architecture) in _CAUSAL_CONV_LSTM_ARCHITECTURES


def _is_a2_architecture(architecture: _Union[Architecture, str]) -> bool:
    return Architecture(architecture) in {
        Architecture.A2_FULL_LITE,
        Architecture.A2_COMPLEX_LITE,
        Architecture.A2_COMPLEX_REVYLO,
        Architecture.A2_COMPLEX_NANO64X4,
        Architecture.A2_COMPLEX_NANO125X3,
        Architecture.A2_DOUBLE_LITE,
        Architecture.A2_XDOUBLE_LITE,
        Architecture.A2_DOUBLEPLUS_LITE,
    }


class _InputValidationError(ValueError):
    pass


_TTS_SHORT_VERSION = _Version(7, 0, 0)
_SUPER_INPUT_V2_VERSION = _Version(8, 0, 0)


def _detect_input_version(
    input_path, verbose: bool = False
) -> _Tuple[_Version, bool]:
    """
    Check to see if the input matches any of the known inputs

    :return: version, strong match
    """

    def detect_strong(input_path) -> _Optional[_Version]:
        def assign_hash(path):
            # Use this to create hashes for new files
            md5 = _hashlib.md5()
            buffer_size = 65536
            with open(path, "rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    md5.update(data)
            file_hash = md5.hexdigest()
            return file_hash

        file_hash = assign_hash(input_path)
        if verbose:
            print(f"Strong hash: {file_hash}")

        version = {
            "4d54a958861bf720ec4637f43d44a7ef": _Version(1, 0, 0),
            "7c3b6119c74465f79d96c761a0e27370": _Version(1, 1, 1),
            "ede3b9d82135ce10c7ace3bb27469422": _Version(2, 0, 0),
            "36cd1af62985c2fac3e654333e36431e": _Version(3, 0, 0),
            # TTS input v10.wav uses the v3 input/validation/blip layout.
            "5b6ae5a15a40f69a28c2ae969a690bd4": _Version(3, 0, 0),
            # TTS input v11.wav also uses the v3 input/validation/blip layout.
            "e949d388461b97c8e5cbbdd0a214cda0": _Version(3, 0, 0),
            # TTS input v11 short.wav is a 4-minute v11 edit with the same
            # v3 front section and the same registered v11 validation tail.
            "add5697003f29062e47a51bca21d3436": _Version(3, 0, 0),
            # TTSv12.wav is a 4-minute TTS input v10 edit. It deletes one minute
            # from the repeated late sweep section and preserves the v3 validation tail.
            "89a6f8065e426ef48d105410da8e9302": _Version(3, 0, 0),
            # Short TTS-style inputs: same v3 front/latency layout as TTS input
            # v10, with a shortened late sweep section and a repeated validation
            # tail that is close enough for the normal v3 repeatability check.
            "0ea89c7bec54ef5d9175a3b4e94f7a19": _TTS_SHORT_VERSION,
            "2c03451301ca45babe62c47fba419324": _TTS_SHORT_VERSION,
            "eea7ff116eed2e5093734f2973d7aae7": _TTS_SHORT_VERSION,
            "b5ed96015172f72b26acc75074eff3d6": _TTS_SHORT_VERSION,
            # Custom AP106/AM4 captures use the v3-style blip layout, but not the
            # repeated-validation layout used by the standard v3 file.
            "ed8fc4aaf6707a45b04a62ba352d31d1": _Version(6, 0, 0),
            "81d2ff4a7d7d3b6f6fed0bc2f36a4070": _Version(6, 0, 0),
            "80e224bd5622fd6153ff1fd9f34cb3bd": _PROTEUS_VERSION,
            "3a126f4bee0627f0941a6e20ec3ea20d": _Version(5, 0, 0),
            "938388dbb092b1494a0d00d368fd3591": _SUPER_INPUT_V2_VERSION,
        }.get(file_hash)
        if version is None:
            print(
                f"Provided input file {input_path} does not strong-match any known "
                "standard input files."
            )
        return version

    def detect_weak(input_path) -> _Optional[_Version]:
        def assign_hash(path):
            Hash = _Optional[str]
            Hashes = _Tuple[Hash, Hash]

            def _hash(x: _np.ndarray) -> str:
                return _hashlib.md5(x).hexdigest()

            def assign_hashes_v1(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V1_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V1_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: t_blips + t_sweep + t_white])
                # v1 ends with validation signal
                end_hash = _hash(x[-t_validation:])
                return start_hash, end_hash

            def assign_hashes_v2(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V2_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V2_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: (t_blips + t_sweep + t_white)])
                # v2 ends with 2x validation & blips
                end_hash = _hash(x[-(2 * t_validation + t_blips) :])
                return start_hash, end_hash

            def assign_hashes_v3(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V3_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                # See below.
                end_of_start_interval = 17 * rate  # Start at 0
                start_of_end_interval = -9 * rate
                start_hash = _hash(x[:end_of_start_interval])
                end_hash = _hash(x[start_of_end_interval:])
                return start_hash, end_hash

            def assign_hash_v4(path) -> Hash:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V4_DATA_INFO.rate:
                    return None
                # I don't care about anything in the file except the starting blip and
                start_hash = _hash(x[: int(1 * _V4_DATA_INFO.rate)])
                return start_hash

            # TODO: I don't know exactly what is going on here lol
            def assign_hashes_v5(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V5_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                # See below.
                end_of_start_interval = 17 * rate  # Start at 0
                start_of_end_interval = -9 * rate
                start_hash = _hash(x[:end_of_start_interval])
                end_hash = _hash(x[start_of_end_interval:])
                return start_hash, end_hash

            start_hash_v1, end_hash_v1 = assign_hashes_v1(path)
            start_hash_v2, end_hash_v2 = assign_hashes_v2(path)
            start_hash_v3, end_hash_v3 = assign_hashes_v3(path)
            hash_v4 = assign_hash_v4(path)
            start_hash_v5, end_hash_v5 = assign_hashes_v5(path)

            return (
                start_hash_v1,
                end_hash_v1,
                start_hash_v2,
                end_hash_v2,
                start_hash_v3,
                end_hash_v3,
                hash_v4,
                start_hash_v5,
                end_hash_v5,
            )

        (
            start_hash_v1,
            end_hash_v1,
            start_hash_v2,
            end_hash_v2,
            start_hash_v3,
            end_hash_v3,
            hash_v4,
            start_hash_v5,
            end_hash_v5,
        ) = assign_hash(input_path)

        if verbose:
            print(
                "Weak hashes:\n"
                f" Start (v1) : {start_hash_v1}\n"
                f" End (v1)   : {end_hash_v1}\n"
                f" Start (v2) : {start_hash_v2}\n"
                f" End (v2)   : {end_hash_v2}\n"
                f" Start (v3) : {start_hash_v3}\n"
                f" End (v3)   : {end_hash_v3}\n"
                f" Proteus    : {hash_v4}\n"
                f" Start (v5) : {start_hash_v5}\n"
                f" End (v5)   : {end_hash_v5}\n"
            )

        # Check for matches, starting with most recent. Proteus last since its match is
        # the most permissive.
        version = {
            (
                "dadb5d62f6c3973a59bf01439799809b",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): _Version(3, 0, 0),
            (
                "3175717ee1cdc4a625736096420de853",
                "f76c2a5114fbe67e2107714872b0a2ad",
            ): _Version(3, 0, 0),
            (
                "3175717ee1cdc4a625736096420de853",
                "a623126d7f2e7903172ae81eb08c5d75",
            ): _TTS_SHORT_VERSION,
            (
                "3175717ee1cdc4a625736096420de853",
                "09f8d064ed8d5fb6e05ebfe390a85ba9",
            ): _TTS_SHORT_VERSION,
            (
                "3175717ee1cdc4a625736096420de853",
                "396373b9241710a0c33f04846e268fa6",
            ): _TTS_SHORT_VERSION,
            (
                "3175717ee1cdc4a625736096420de853",
                "8207f6824096fa18c3a385c6918136ce",
            ): _TTS_SHORT_VERSION,
            (
                "cbd32bd7031848a6b8eae703fd434d93",
                "fb7171aa6d8f13e95b54568df9c601e5",
            ): _SUPER_INPUT_V2_VERSION,
            (
                "5045683470a078257aba05c071d00aea",
                "da2ac193492c256e78dd6a7eedc3d0c5",
            ): _Version(6, 0, 0),
            (
                "5045683470a078257aba05c071d00aea",
                "f8e585df25402b83a53f39acb8c46627",
            ): _Version(6, 0, 0),
        }.get((start_hash_v3, end_hash_v3))
        if version is not None:
            return version
        version = {
            (
                "1c4d94fbcb47e4d820bef611c1d4ae65",
                "28694e7bf9ab3f8ae6ef86e9545d4663",
            ): _Version(2, 0, 0)
        }.get((start_hash_v2, end_hash_v2))
        if version is not None:
            return version
        version = {
            (
                "bb4e140c9299bae67560d280917eb52b",
                "9b2468fcb6e9460a399fc5f64389d353",
            ): _Version(
                1, 0, 0
            ),  # FIXME!
            (
                "9f20c6b5f7fef68dd88307625a573a14",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): _Version(1, 1, 1),
        }.get((start_hash_v1, end_hash_v1))
        if version is not None:
            return version
        version = {"46151c8030798081acc00a725325a07d": _PROTEUS_VERSION}.get(hash_v4)
        return version

    version = detect_strong(input_path)
    if version is not None:
        strong_match = True
        return version, strong_match
    if verbose:
        print("Falling back to weak-matching...")
    version = detect_weak(input_path)
    if version is None:
        raise _InputValidationError(
            f"Input file at {input_path} cannot be recognized as any known version!"
        )
    strong_match = False

    return version, strong_match


class _DataInfo(_BaseModel):
    """:param major_version: Data major version"""
    major_version: int
    rate: _Optional[float]
    t_blips: int
    first_blips_start: int
    t_validate: int
    train_start: int
    validation_start: int
    noise_interval: _Tuple[int, int]
    blip_locations: _Sequence[_Sequence[int]]


# ESR will be plotted from 10_000 to 101_000
_V1_DATA_INFO = _DataInfo(
    major_version=1,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=48_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-432_000,
    noise_interval=(0, 6000),
    blip_locations=((12_000, 36_000),),
)

# ESR will be plotted from 10_000 to 101_000
# V2:
# (0:00-0:02) Blips at 0:00.5 and 0:01.5
# (0:02-0:05) Chirps
# (0:05-0:07) Noise
# (0:07-2:50.5) General training data
# (2:50.5-2:51) Silence
# (2:51-3:00) Validation 1
# (3:00-3:09) Validation 2
# (3:09-3:11) Blips at 3:09.5 and 3:10.5
_V2_DATA_INFO = _DataInfo(
    major_version=2,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-960_000,  # 96_000 + 2 * 432_000
    noise_interval=(12_000, 18_000),
    blip_locations=((24_000, 72_000), (-72_000, -24_000)),
)

# ESR will be plotted from 10_000 to 101_000
# V3:
# (0:00-0:09) Validation 1
# (0:09-0:10) Silence
# (0:10-0:12) Blips at 0:10.5 and 0:11.5
# (0:12-0:15) Chirps
# (0:15-0:17) Noise
# (0:17-3:00.5) General training data
# (3:00.5-3:01) Silence
# (3:01-3:10) Validation 2
_V3_DATA_INFO = _DataInfo(
    major_version=3,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,            # time window in which to find the two blips: 2 seconds, with one blip in the middle of each second
    first_blips_start=480_000, # ((first blip position in seconds - .5 seconds) * sample rate)
    t_validate=432_000,        # first 9 seconds are the validation, which is repeated at the end
    train_start=480_000,
    validation_start=-432_000, # first 9 seconds are the validation, which is repeated at the end (note the negative index!)
    noise_interval=(492_000, 498_000), # silence for 125ms in length, 250ms before the first blip, establishing the wet "output" file's noise floor
    blip_locations=((504_000, 552_000),), # Actual locations of each of the two blips
)

# Short TTS-style captures A/B/B2/C:
# * Same first 181 seconds as TTS input v10 and the same v3 impulse locations.
# * Shorter late sweep section than TTS input v10.
# * Last 9 seconds are the intended validation tail; they are not byte-identical
#   to the stock v3/v10 tail, but they pass the normal v3 repeatability ESR check.
_TTS_SHORT_DATA_INFO = _DataInfo(
    major_version=7,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=_V3_DATA_INFO.t_blips,
    first_blips_start=_V3_DATA_INFO.first_blips_start,
    t_validate=_V3_DATA_INFO.t_validate,
    train_start=_V3_DATA_INFO.train_start,
    validation_start=_V3_DATA_INFO.validation_start,
    noise_interval=_V3_DATA_INFO.noise_interval,
    blip_locations=_V3_DATA_INFO.blip_locations,
)
# FYI, this is how the validation slicing is done in _check_v3:
#    y = _wav_to_tensor(output_path, rate=rate)
#    n = len(_wav_to_tensor(input_path)) # to End-crop output
#    y_val_1 = y[: _V3_DATA_INFO.t_validate]
#       = 0 : 432_000
#    y_val_2 = y[n - _V3_DATA_INFO.t_validate : n]
#       = (len(input.wav)  - 432_000) : len(input.wav)
#
# FYI, from _get_data_config:
# validation_start = data_info.validation_start
# train_stop = validation_start
# train_kwargs = {"start_samples": data_info.train_start, "stop_samples": train_stop}
# validation_kwargs = {"start_samples": validation_start}

# ESR will be plotted from 10_000 to 101_000
# V4 (aka GuitarML Proteus)
# https://github.com/GuitarML/Releases/releases/download/v1.0.0/Proteus_Capture_Utility.zip
# * 44.1k
# * Odd length...
# * There's a blip on sample zero. This has to be ignored or else over-compensated
#   latencies will come out wrong!
# (0:00-0:01) Blips at 0:00.0 and 0:00.5
# (0:01-0:09) Sine sweeps
# (0:09-0:17) White noise
# (0:17:0.20) Rising white noise (to 0:20.333 appx)
# (0:20-3:30.858) General training data (ends on sample 9,298,872)
# I'm arbitrarily assigning the last 10 seconds as validation data.
_V4_DATA_INFO = _DataInfo(
    major_version=4,
    rate=44_100.0,
    t_blips=44_099,  # Need to ignore the first blip!
    first_blips_start=1,  # Need to ignore the first blip!
    t_validate=441_000,
    # Blips are problematic for training because they don't have preceding silence
    train_start=44_100,
    validation_start=-441_000,
    noise_interval=(6_000, 12_000),
    blip_locations=((22_050,),),
)

# FIXME: Seriously, you can see how unnecessarily complicated this is.
#   What makes a lot more sense (SP = silence padding):
#       | SP | Validation 1 | SP Blips SP | Training data | SP | Validation 2 | SP |
#       | 0  | v1_start     | blips_start | train_start   |    | v1_start     | -1 |
#   If SP is the same length in all instances, then you can just make all splits be:
#       silence_padded_start_of_each_section = absolute_start_of_each section - (1/2 * len(SP))
#       silence_padded_end_of_each_section = absolute_end_of_each section + (1/2 * len(SP))
#
# TODO: Create user-friendly documentation for how everything works.
# TODO: Refactor all of this convoluted logic to be more obvious and conceptually simpler.
# V5 = S3 Sound's "acid test". This is meant for TB-303-style acid synths, not guitars!
#
# ESR will be plotted from 96_000 to 120_000.
# 120 BPM = 2 seconds per bar = 500 ms per 1/4 note = 125 ms per 1/16 note.
# 1/4 = 24_000, 1/8 = 12_000, 1/16 = 6_000 samples @ 48_000 Hz sample rate.
#
# Minutes:Seconds.Samples
# (00:00.000 - 00:02.000) = Silence [0:96_000]
# (00:02.000 - 01:10.000) = Validation 1 [96_000:3_360_000] 1:08 length, 70 * 48000 = 3360000
# (01:10.000 - 01:10.499) = Silence before Blip 1 [3_360_000:3_384_000]
# (01:10.500)             = Blip 1 @ 3_384_000
# (01:10.501 - 01:11:499) = Silence between Blip 1 and Blip 2
# (01:11.500)             = Blip 2 @ 3_432_000
# (01:11.501 - 01:12.000) = Silence between Blip 2 and Operator white noise C3 (discontinuity?)
# (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
# (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
# (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
# (10:12.500 - 10:14.000) = Silence
# (10:14.000)             = EOF @ 29_472_000
_V5_DATA_INFO = _DataInfo(
    major_version=5,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,               # time window in which to find the two blips: 2 seconds, with one blip in the middle of each second
    first_blips_start=3_360_000, # ((first blip position in seconds - .5 seconds) * sample rate)
    t_validate=3_264_000,         # 01:08.000 = 68 * 48000 = 3_264_000
    train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
    validation_start=-3_360_000,  # -01:09.500 = 1:08.000 + 1.5 seconds trailing silence. ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position. (29_472_000 - (((9 * 60) + 4.5) * 48000)) / 48000 = 69.5
    noise_interval=(3_372_000, 3_378_000), # silence for 125ms in length, 250ms before the first blip, establishing the wet "output" file's noise floor.
    blip_locations=((3_384_000, 3_432_000),), # Actual locations of each of the two blips
)

# AP106/AM4 custom capture layout:
# * Uses a dedicated AP106 latency matcher so the long AP106 startup pulse maps
#   back onto NAM's standard v3 latency convention instead of tripping on its
#   low-level pre-ringing.
# * Does not provide the repeated start/end validation block that stock v3 expects.
# * Uses the tail of the file as validation and skips the pre-train silence
#   requirement used by standard v3.
_V6_DATA_INFO = _DataInfo(
    major_version=6,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=48_000,
    first_blips_start=0,
    t_validate=480_000,
    train_start=720_000,
    validation_start=-480_000,
    noise_interval=(12_000, 24_000),
    blip_locations=((4_571,),),
)


class _SuperInputV2Anchor(_NamedTuple):
    name: str
    template_start: int
    template_stop: int
    expected_peak: int
    lookahead: int
    lookback: int
    coarse_hop: int
    min_score: float


# Super Input v2 has no NAM-style impulse pair. These windows are stable,
# distinctive dry moments found in the file itself. The 180s burst is the most
# precise single anchor; the later tail pattern is a longer confidence anchor.
_SUPER_INPUT_V2_ANCHORS = (
    _SuperInputV2Anchor(
        name="post-quiet burst at 180s",
        template_start=8_649_600,  # 180.200000s
        template_stop=8_668_800,   # 180.600000s
        expected_peak=8_656_172,   # 180.336917s
        lookahead=4_800,
        lookback=24_000,
        coarse_hop=16,
        min_score=0.35,
    ),
    _SuperInputV2Anchor(
        name="tail transient pattern at 226-230s",
        template_start=10_857_600,  # 226.200000s
        template_stop=11_080_800,   # 230.850000s
        expected_peak=10_955_014,   # 228.229458s
        lookahead=4_800,
        lookback=24_000,
        coarse_hop=64,
        min_score=0.25,
    ),
    _SuperInputV2Anchor(
        name="post-quiet onset at 202s",
        template_start=9_704_640,  # 202.180000s
        template_stop=9_777_600,   # 203.700000s
        expected_peak=9_736_186,   # 202.837208s
        lookahead=4_800,
        lookback=24_000,
        coarse_hop=32,
        min_score=0.25,
    ),
)

_SUPER_INPUT_V2_DATA_INFO = _DataInfo(
    major_version=8,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=0,
    first_blips_start=0,
    t_validate=_V3_DATA_INFO.t_validate,
    train_start=0,
    validation_start=-_V3_DATA_INFO.t_validate,
    noise_interval=(0, 0),
    blip_locations=(),
)

_AP106_SELF_VALIDATION_WINDOW_SAMPLES = 24_000
_AP106_SELF_VALIDATION_HOP_SAMPLES = 24_000
_AP106_SELF_VALIDATION_MIN_RMS = 0.05
_AP106_SELF_VALIDATION_DRY_ESR_THRESHOLD = 0.05
_AP106_SELF_VALIDATION_WET_ESR_THRESHOLD = 0.65
_AP106_SELF_VALIDATION_SIMILARITY_TOLERANCE = 1.0e-5
_AP106_LATENCY_TEMPLATE_PRE_SAMPLES = 300
_AP106_LATENCY_TEMPLATE_POST_SAMPLES = 300
_AP106_LATENCY_SEARCH_PRE_SAMPLES = 200
_AP106_LATENCY_SEARCH_POST_SAMPLES = 400
# Stock NAM's v3 trigger fires about 32 samples before the peak of a sharp
# reamped blip. Subtract the same offset so AP106 latencies stay on the same
# scale as the standard NAM files.
_AP106_LATENCY_V3_COMPATIBILITY_OFFSET = 32


class _AP106SelfValidation(_NamedTuple):
    window_seconds: float
    start_1_seconds: float
    start_2_seconds: float
    dry_scale_invariant_esr: float
    dry_rms_ratio: float
    wet_esr: float


def _warn_lookaheads(indices: _Sequence[int]) -> str:
    return (
        f"WARNING: delays from some blips ({','.join([str(i) for i in indices])}) are "
        "at the minimum value possible. This usually means that something is "
        "wrong with your data. Check if training ends with a poor result!"
    )


def _calibrate_latency_v_all(
    data_info: _DataInfo,
    y,
    manual_available: bool,
    show_plots: bool,
    verbose: bool = False,
    abs_threshold=_DELAY_CALIBRATION_ABS_THRESHOLD,
    rel_threshold=_DELAY_CALIBRATION_REL_THRESHOLD,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
    _override_suppress_plots: bool = False,
) -> _metadata.LatencyCalibration:
    """
    Calibrate the delay in teh input-output pair based on blips.
    This only uses the blips in the first set of blip locations!

    :param y: The output audio, in complete.
    """

    def report_any_latency_warnings(
        delays: _Sequence[int],
    ) -> _metadata.LatencyCalibrationWarnings:
        # Warnings associated with any single delay:

        if len(delays) == 0:
            return _metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            )

        # "Lookahead warning": if the delay is equal to the lookahead, then it's
        # probably an error.
        lookahead_warnings = [i for i, d in enumerate(delays, 1) if d == -lookahead]
        matches_lookahead = len(lookahead_warnings) > 0
        if matches_lookahead:
            print(_warn_lookaheads(lookahead_warnings))

        # Ensemble warnings

        # If they're _really_ different, then something might be wrong.
        max_disagreement_threshold = 20
        max_disagreement_too_high = (
            _np.max(delays) - _np.min(delays) >= max_disagreement_threshold
        )
        if max_disagreement_too_high:
            print(
                "WARNING: Latencies are anomalously different from each other (more "
                f"than {max_disagreement_threshold} samples). If this model turns out "
                "badly, then you might need to provide the latency manually."
            )

        return _metadata.LatencyCalibrationWarnings(
            matches_lookahead=matches_lookahead,
            disagreement_too_high=max_disagreement_too_high,
            not_detected=False,
        )

    lookahead = 1_000
    lookback = 10_000
    # Calibrate the level for the trigger:
    end_of_blip_search_range = data_info.first_blips_start + data_info.t_blips
    if verbose:
        n_blips = len(data_info.blip_locations[0])
        blip_label = "dirac pulse blip" if n_blips == 1 else "dirac pulse blips"
        print(
            f"Scanning for {n_blips} {blip_label} in sample range "
            f"{data_info.first_blips_start} to {end_of_blip_search_range}..."
        )
    y = y[data_info.first_blips_start:end_of_blip_search_range]
    # Calculate the max amplitude of the background noise within data_info.noise_interval
    noise_interval_before_dirac_pulse_blip = data_info.noise_interval[0] - data_info.first_blips_start
    noise_interval_after_dirac_pulse_blip = data_info.noise_interval[1] - data_info.first_blips_start
    background_level = _np.max(
        _np.abs(
            y[noise_interval_before_dirac_pulse_blip:noise_interval_after_dirac_pulse_blip]
        )
    )
    if verbose:
        print(
            f"background_level = {background_level} based on a scan from "
            f"{noise_interval_before_dirac_pulse_blip} to "
            f"{noise_interval_after_dirac_pulse_blip}"
        )
        print(f"rel_threshold is {rel_threshold}")
    # Which is larger? The background noise from the previous step + abs_threshold, or (1.0 + rel_threshold) * the background noise?
    trigger_threshold = max(
        background_level + abs_threshold,
        (1.0 + rel_threshold) * background_level,
    )
    if verbose:
        print(f"Trigger threshold is {trigger_threshold}")

    blip_search_ranges = []
    for blip_index, blip_location in enumerate(data_info.blip_locations[0], 1):
        # Relative to start of the data
        i_rel = blip_location - data_info.first_blips_start
        if verbose:
            print(
                f"Blip {blip_index} expected to be at {i_rel} relative to "
                f"data_info.first_blips_start value of {data_info.first_blips_start}"
            )
        # Start at 1_000 samples before the blip location
        start_looking = i_rel - lookahead
        # Stop at 10_000 samples after the blip location
        stop_looking = i_rel + lookback
        if verbose:
            print(
                f"Scanning for blip {blip_index} between {start_looking} and "
                f"{stop_looking}"
            )
        blip_search_ranges.append(y[start_looking:stop_looking])
    y_scan_average = _np.mean(_np.stack(blip_search_ranges), axis=0)
    triggered = _np.where(_np.abs(y_scan_average) > trigger_threshold)[0]
    if len(triggered) == 0:  # No impulse responses were detected; can't calibrate
        msg = (
            "No response activated the trigger in response to input spikes. "
            "Is something wrong with the reamp? "
            "Check for background noise just before the first blip!"
        )
        if (show_plots or not manual_available) and not _override_suppress_plots:
            print(msg)
            print("SHARE THIS PLOT IF YOU ASK FOR HELP")
            _plt.figure()
            _plt.plot(
                _np.arange(-lookahead, lookback),
                y_scan_average,
                color="C0",
                label="Signal average",
            )
            for blip_search_range in blip_search_ranges:
                _plt.plot(
                    _np.arange(-lookahead, lookback), blip_search_range, color="C0", alpha=0.2
                )
            _plt.axvline(x=0, color="C1", linestyle="--", label="Trigger")
            _plt.axhline(
                y=-trigger_threshold, color="k", linestyle="--", label="Threshold"
            )
            _plt.axhline(y=trigger_threshold, color="k", linestyle="--")
            _plt.xlim((-lookahead, lookback))
            _plt.xlabel("Samples")
            _plt.ylabel("Response")
            _plt.legend()
            _plt.title("SHARE THIS PLOT IF YOU ASK FOR HELP")
            _plt.show()
        delays = []
        recommended = None
    else:
        delay = triggered[0] + start_looking - i_rel
        delays = [delay]
        recommended = delay - safety_factor
        if verbose:
            print(
                f"Delay based on average _np.mean(_np.stack(blip_search_ranges), "
                f"axis=0)` is {delay}"
            )
            print(
                f"Delay is based on triggered[0] {triggered[0]} + start_looking "
                f"{start_looking} - relative index = {i_rel}"
            )
            print(
                f"After applying safety factor of {safety_factor}, the final delay "
                f"is {recommended} samples"
            )

    warnings = report_any_latency_warnings(delays)

    return _metadata.LatencyCalibration(
        algorithm_version=1,
        delays=delays,
        safety_factor=safety_factor,
        recommended=recommended,
        warnings=warnings,
    )


_calibrate_latency_v1 = _partial(_calibrate_latency_v_all, _V1_DATA_INFO)
_calibrate_latency_v2 = _partial(_calibrate_latency_v_all, _V2_DATA_INFO)
_calibrate_latency_v3 = _partial(_calibrate_latency_v_all, _V3_DATA_INFO)
_calibrate_latency_v4 = _partial(_calibrate_latency_v_all, _V4_DATA_INFO)
_calibrate_latency_v5 = _partial(_calibrate_latency_v_all, _V5_DATA_INFO)


def _get_v6_latency_template(input_path: str) -> _Tuple[int, _np.ndarray]:
    x = _wav_to_np(input_path)[
        _V6_DATA_INFO.first_blips_start : _V6_DATA_INFO.first_blips_start
        + _V6_DATA_INFO.t_blips
    ]
    pulse_peak = int(_np.argmax(_np.abs(x)))
    template_start = max(0, pulse_peak - _AP106_LATENCY_TEMPLATE_PRE_SAMPLES)
    template_stop = min(len(x), pulse_peak + _AP106_LATENCY_TEMPLATE_POST_SAMPLES)
    template = x[template_start:template_stop].astype(_np.float64)
    template = template - _np.mean(template)
    template_norm = _np.linalg.norm(template)
    if template_norm == 0.0:
        raise RuntimeError("Failed to build an AP106 latency template from the input file.")
    return template_start, template / template_norm


def _calibrate_latency_v6(
    input_path: str,
    output_path: str,
    manual_available: bool,
    show_plots: bool,
    verbose: bool = False,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
    _override_suppress_plots: bool = False,
) -> _metadata.LatencyCalibration:
    """
    Calibrate AP106 latency by correlating the known AP106 startup pulse against
    the wet signal, then translating the result onto NAM's standard v3 latency
    scale.
    """
    template_start, template = _get_v6_latency_template(input_path)
    y = _wav_to_np(output_path)[
        _V6_DATA_INFO.first_blips_start : _V6_DATA_INFO.first_blips_start
        + _V6_DATA_INFO.t_blips
    ].astype(_np.float64)
    search_start = max(0, template_start - _AP106_LATENCY_SEARCH_PRE_SAMPLES)
    search_stop = min(
        len(y) - len(template) + 1,
        template_start + _AP106_LATENCY_SEARCH_POST_SAMPLES + 1,
    )
    if verbose:
        print(
            "Calibrating AP106 latency by matched-filtering the known AP106 "
            f"startup pulse in sample range {search_start} to {search_stop}..."
        )
    if search_stop <= search_start:
        return _metadata.LatencyCalibration(
            algorithm_version=2,
            delays=[],
            safety_factor=safety_factor,
            recommended=None,
            warnings=_metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            ),
        )

    correlations = []
    for start in range(search_start, search_stop):
        window = y[start : start + len(template)]
        window = window - _np.mean(window)
        window_norm = _np.linalg.norm(window)
        correlation = 0.0 if window_norm == 0.0 else float(_np.dot(window, template) / window_norm)
        correlations.append(correlation)
    correlations = _np.asarray(correlations, dtype=_np.float64)
    if len(correlations) == 0 or not _np.isfinite(correlations).any():
        return _metadata.LatencyCalibration(
            algorithm_version=2,
            delays=[],
            safety_factor=safety_factor,
            recommended=None,
            warnings=_metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            ),
        )

    peak_index = int(_np.argmax(_np.abs(correlations)))
    peak_start = search_start + peak_index
    raw_delay = peak_start - template_start
    delay = raw_delay - _AP106_LATENCY_V3_COMPATIBILITY_OFFSET
    recommended = delay - safety_factor
    if verbose:
        print(
            f"AP106 matched-filter lag is {raw_delay} samples. "
            f"Subtracting the v3 compatibility offset of "
            f"{_AP106_LATENCY_V3_COMPATIBILITY_OFFSET} gives {delay}."
        )
        print(
            f"After applying safety factor of {safety_factor}, the final delay "
            f"is {recommended} samples"
        )

    return _metadata.LatencyCalibration(
        algorithm_version=2,
        delays=[delay],
        safety_factor=safety_factor,
        recommended=recommended,
        warnings=_metadata.LatencyCalibrationWarnings(
            matches_lookahead=False,
            disagreement_too_high=False,
            not_detected=False,
        ),
    )


class _SuperInputV2LatencyMatch(_NamedTuple):
    anchor: _SuperInputV2Anchor
    delay: int
    score: float


def _super_input_v2_feature(x: _np.ndarray, window: int = 64) -> _np.ndarray:
    x = _np.asarray(x, dtype=_np.float64)
    if len(x) == 0:
        return x
    if window <= 1 or len(x) < window:
        envelope = _np.abs(x)
    else:
        kernel = _np.ones(window, dtype=_np.float64) / float(window)
        envelope = _np.sqrt(_np.convolve(x * x, kernel, mode="same"))
    onset = _np.maximum(_np.diff(envelope, prepend=envelope[0]), 0.0)
    envelope_max = float(_np.max(envelope))
    onset_max = float(_np.max(onset))
    if envelope_max > 0.0:
        envelope = envelope / envelope_max
    if onset_max > 0.0:
        onset = onset / onset_max
    return envelope + onset


def _normalized_template_match(
    template: _np.ndarray, search: _np.ndarray
) -> _Optional[_Tuple[int, float]]:
    template = _np.asarray(template, dtype=_np.float64)
    search = _np.asarray(search, dtype=_np.float64)
    if len(template) == 0 or len(search) < len(template):
        return None

    template = template - _np.mean(template)
    template_norm = float(_np.linalg.norm(template))
    if template_norm == 0.0:
        return None

    n = len(template)
    ones = _np.ones(n, dtype=_np.float64)
    numerator = _np.correlate(search, template, mode="valid")
    search_sum = _np.correlate(search, ones, mode="valid")
    search_sum_squares = _np.correlate(search * search, ones, mode="valid")
    search_variance = search_sum_squares - (search_sum * search_sum / n)
    search_variance = _np.maximum(search_variance, 0.0)
    denominator = _np.sqrt(search_variance) * template_norm
    scores = _np.divide(
        numerator,
        denominator,
        out=_np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    best_index = int(_np.argmax(scores))
    return best_index, float(scores[best_index])


def _match_super_input_v2_anchor(
    x: _np.ndarray,
    y: _np.ndarray,
    anchor: _SuperInputV2Anchor,
    verbose: bool = False,
) -> _Optional[_SuperInputV2LatencyMatch]:
    if anchor.template_stop > len(x):
        if verbose:
            print(f"Super Input v2 anchor '{anchor.name}' is outside the dry file.")
        return None

    search_start = max(0, anchor.template_start - anchor.lookahead)
    search_stop = min(len(y), anchor.template_stop + anchor.lookback)
    if search_stop <= search_start:
        return None

    template_feature = _super_input_v2_feature(
        x[anchor.template_start : anchor.template_stop]
    )[:: anchor.coarse_hop]
    search_feature = _super_input_v2_feature(y[search_start:search_stop])[
        :: anchor.coarse_hop
    ]
    coarse_match = _normalized_template_match(template_feature, search_feature)
    if coarse_match is None:
        return None
    coarse_index, coarse_score = coarse_match
    coarse_delay = search_start + coarse_index * anchor.coarse_hop - anchor.template_start

    refine_radius = 4 * anchor.coarse_hop
    refine_start = max(anchor.template_start, anchor.expected_peak - 7_200)
    refine_stop = min(anchor.template_stop, anchor.expected_peak + 7_200)
    refine_search_start = max(0, refine_start + coarse_delay - refine_radius)
    refine_search_stop = min(len(y), refine_stop + coarse_delay + refine_radius)
    refined_delay = coarse_delay
    refined_score = coarse_score
    refined_match = _normalized_template_match(
        _super_input_v2_feature(x[refine_start:refine_stop]),
        _super_input_v2_feature(y[refine_search_start:refine_search_stop]),
    )
    if refined_match is not None:
        refined_index, candidate_score = refined_match
        candidate_delay = refine_search_start + refined_index - refine_start
        if candidate_score >= max(0.10, coarse_score * 0.70):
            refined_delay = candidate_delay
            refined_score = candidate_score

    if verbose:
        print(
            f"Super Input v2 anchor '{anchor.name}' delay {refined_delay} "
            f"samples, score {refined_score:.3f}."
        )

    if refined_score < anchor.min_score:
        if verbose:
            print(
                f"Super Input v2 anchor '{anchor.name}' rejected because score "
                f"{refined_score:.3f} is below {anchor.min_score:.3f}."
            )
        return None
    return _SuperInputV2LatencyMatch(anchor, int(refined_delay), refined_score)


def _calibrate_latency_v8(
    input_path: str,
    output_path: str,
    manual_available: bool,
    show_plots: bool,
    verbose: bool = False,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
    _override_suppress_plots: bool = False,
) -> _metadata.LatencyCalibration:
    """
    Calibrate Super Input v2 with known dry transient/onset anchors.

    Super Input v2 does not contain the stock NAM impulse pair. Instead, this
    correlates robust envelope features around several distinctive dry events.
    """
    x = _wav_to_np(input_path)
    y = _wav_to_np(output_path)
    matches = [
        match
        for anchor in _SUPER_INPUT_V2_ANCHORS
        for match in [_match_super_input_v2_anchor(x, y, anchor, verbose=verbose)]
        if match is not None
    ]

    if len(matches) == 0:
        msg = (
            "Super Input v2 latency could not be detected from the known anchor "
            "points. Provide latency manually or check that the wet file is the "
            "matching reamp of this input."
        )
        if (show_plots or not manual_available) and not _override_suppress_plots:
            print(msg)
        return _metadata.LatencyCalibration(
            algorithm_version=3,
            delays=[],
            safety_factor=safety_factor,
            recommended=None,
            warnings=_metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            ),
        )

    delays = [match.delay for match in matches]
    best_match = max(matches, key=lambda match: match.score)
    inliers = [
        match
        for match in matches
        if abs(match.delay - best_match.delay) <= 256
    ]
    score_sum = sum(match.score for match in inliers)
    delay = (
        int(round(sum(match.delay * match.score for match in inliers) / score_sum))
        if score_sum > 0.0
        else best_match.delay
    )
    recommended = delay - safety_factor
    disagreement_too_high = max(delays) - min(delays) > 256
    if disagreement_too_high:
        print(
            "WARNING: Super Input v2 latency anchors disagree by more than 256 "
            "samples. The highest-confidence inlier set was used."
        )
    if verbose:
        print(
            "Super Input v2 anchor delays: "
            + ", ".join(
                f"{match.anchor.name}={match.delay} ({match.score:.3f})"
                for match in matches
            )
        )
        print(
            f"After applying safety factor of {safety_factor}, the final delay "
            f"is {recommended} samples"
        )

    return _metadata.LatencyCalibration(
        algorithm_version=3,
        delays=delays,
        safety_factor=safety_factor,
        recommended=recommended,
        warnings=_metadata.LatencyCalibrationWarnings(
            matches_lookahead=any(
                match.delay <= -match.anchor.lookahead + match.anchor.coarse_hop
                for match in matches
            ),
            disagreement_too_high=disagreement_too_high,
            not_detected=False,
        ),
    )


def _plot_latency_v_all(
    data_info: _DataInfo, latency: int, input_path: str, output_path: str, _nofail=True
):
    print("\nPlotting the latency for manual inspection...")
    print(f"data_info.major_version == {data_info.major_version}")
    print(f"Truncating x and y to range [{data_info.first_blips_start}:{data_info.first_blips_start + data_info.t_blips}]")
    x = _wav_to_np(input_path)[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    y = _wav_to_np(output_path)[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    print(f"len(x) = {len(x)} loaded from {input_path}")
    print(f"len(y) = {len(y)} loaded from {output_path}")
    # Only get the blips we really want.
    detected_blip_locations = _np.where(_np.abs(x) > 0.5 * _np.abs(x).max())[0]
    print(f"detected_blip_locations = {detected_blip_locations}")
    if len(detected_blip_locations) == 0:
        print(f"Failed to find any blips in {input_path}")
        print("Plotting the input and output; there should be spikes at around the marked locations.")
        t = _np.arange(data_info.first_blips_start, data_info.first_blips_start + data_info.t_blips)
        expected_spikes = data_info.blip_locations[0]  # For v1 specifically
        fig, axs = _plt.subplots(len((x, y)), 1)
        for ax, curve in zip(axs, (x, y)):
            ax.plot(t, curve)
            [ax.axvline(x=es, color="C1", linestyle="--") for es in expected_spikes]
        _plt.show()
        if _nofail:
            raise RuntimeError("Failed to plot delay")
    else:
        _plt.figure(figsize=(10.8, 7.2), edgecolor='black')
        plot_range_in_samples = 32
        # V1's got not a spike but a longer plateau; take the front of it.
        if data_info.major_version == 1:
            detected_blip_locations = [detected_blip_locations[0]]
        # figsize in inches * dpi default of 100
        #   (16, 5)      = 1600 x 500 pixels
        #   (19.2, 10.8) = 1920 x 1080 pixels
        #   (9.6, 5.4)   = 960 x 540 pixels
        #   10.8, 7.2)   = 1080 x 720 pixels
        # (0, (1, 1))) = "densely dotted line/curve"
        #NOTE: Enable this if you need to debug.
        #print(f"_np.arange(-plot_range_in_samples, plot_range_in_samples) = {_np.arange(-plot_range_in_samples, plot_range_in_samples)}")
        for blip_number, blip_location in enumerate(detected_blip_locations, 1):
            # detected_blip_locations = [24000 72000]
            # _np.arange(-plot_range_in_samples, plot_range_in_samples) = [-24000 ... 23999]
            # 24000 + -2 - 12000 = 11998
            # [(-32 + delay) + blip_location : (31 + delay)]
            # [(-32 + -2 + 24000) : (32 + -2 + 24000)]
            start_time_in_samples = int(-plot_range_in_samples + latency + blip_location)
            print(f"start_time_in_samples = {start_time_in_samples}")
            end_time_in_samples = int(plot_range_in_samples + latency + blip_location)
            print(f"end_time_in_samples = {end_time_in_samples}")
            #NOTE: Enable this if you need to debug (e.g., if the blip location doesn't show up in the plot).
            #print(f"y[start_time_in_samples : end_time_in_samples] for blip {blip_number} = {y[start_time_in_samples : end_time_in_samples]}")
            _plt.plot(
                _np.arange(-plot_range_in_samples, plot_range_in_samples),
                y[start_time_in_samples:end_time_in_samples],
                linestyle=(0, (2, 2)),
                alpha=0.9,
                label=f"Wet {blip_number}"
            )
            _plt.plot(
                _np.arange(-plot_range_in_samples, plot_range_in_samples),
                x[(start_time_in_samples - latency):(end_time_in_samples - latency)],
                linestyle=(1, (2, 2)),
                alpha=0.9,
                label=f"Dry {blip_number}"
            )
        _plt.axvline(x=0, linestyle="--", color="k")
        _plt.ylabel('Amplitude normalized to range [1, -1]')
        _plt.xlabel(f"Sample range from {-plot_range_in_samples} to {plot_range_in_samples} (relative to first detected blip location)")
        _plt.legend(fancybox=True, shadow=True)
        _plt.grid()
        _plt.show()  # This doesn't freeze the notebook


_plot_latency_v1 = _partial(_plot_latency_v_all, _V1_DATA_INFO)
_plot_latency_v2 = _partial(_plot_latency_v_all, _V2_DATA_INFO)
_plot_latency_v3 = _partial(_plot_latency_v_all, _V3_DATA_INFO)
_plot_latency_v4 = _partial(_plot_latency_v_all, _V4_DATA_INFO)
_plot_latency_v5 = _partial(_plot_latency_v_all, _V5_DATA_INFO)
_plot_latency_v7 = _partial(_plot_latency_v_all, _TTS_SHORT_DATA_INFO)


def _plot_latency_v6(latency: int, input_path: str, output_path: str, _nofail=True):
    print("\nPlotting the AP106 latency for manual inspection...")
    template_start, template = _get_v6_latency_template(input_path)
    raw_delay = latency + _AP106_LATENCY_V3_COMPATIBILITY_OFFSET
    template_len = len(template)
    x = _wav_to_np(input_path)[template_start : template_start + template_len]
    y = _wav_to_np(output_path)[
        template_start + raw_delay : template_start + raw_delay + template_len
    ]
    if len(x) != template_len or len(y) != template_len:
        print("Failed to plot AP106 delay")
        if _nofail:
            raise RuntimeError("Failed to plot AP106 delay")
        return
    _plt.figure(figsize=(10.8, 7.2), edgecolor='black')
    _plt.plot(_np.arange(template_len), y, linestyle=(0, (2, 2)), alpha=0.9, label="Wet")
    _plt.plot(_np.arange(template_len), x, linestyle=(1, (2, 2)), alpha=0.9, label="Dry")
    _plt.axvline(x=int(_np.argmax(_np.abs(x))), linestyle="--", color="k")
    _plt.ylabel('Amplitude normalized to range [1, -1]')
    _plt.xlabel('Sample index within the AP106 startup-pulse template')
    _plt.legend(fancybox=True, shadow=True)
    _plt.grid()
    _plt.show()


def _plot_latency_v8(latency: int, input_path: str, output_path: str, _nofail=True):
    print("\nPlotting the Super Input v2 latency for manual inspection...")
    anchor = _SUPER_INPUT_V2_ANCHORS[0]
    x = _wav_to_np(input_path)
    y = _wav_to_np(output_path)
    y_start = anchor.template_start + latency
    y_stop = anchor.template_stop + latency
    if y_start < 0 or y_stop > len(y):
        print("Failed to plot Super Input v2 delay")
        if _nofail:
            raise RuntimeError("Failed to plot Super Input v2 delay")
        return

    dry = _super_input_v2_feature(x[anchor.template_start : anchor.template_stop])
    wet = _super_input_v2_feature(y[y_start:y_stop])
    samples = _np.arange(anchor.template_start, anchor.template_stop)
    _plt.figure(figsize=(10.8, 7.2), edgecolor='black')
    _plt.plot(samples, wet, linestyle=(0, (2, 2)), alpha=0.9, label="Wet")
    _plt.plot(samples, dry, linestyle=(1, (2, 2)), alpha=0.9, label="Dry")
    _plt.axvline(x=anchor.expected_peak, linestyle="--", color="k")
    _plt.ylabel('Normalized envelope/onset feature')
    _plt.xlabel('Sample index in Super Input v2 primary anchor window')
    _plt.legend(fancybox=True, shadow=True)
    _plt.grid()
    _plt.show()


class _AnalyzeLatencyError(RuntimeError):
    """Raised when the latency analysis fails."""
    pass


def _analyze_latency(
    user_latency: _Optional[int],
    input_version: _Version,
    input_path: str,
    output_path: str,
    silent: bool = False,
    verbose: bool = False,
    _override_suppress_plots: bool = False,
) -> _metadata.Latency:
    """
    Use an automatic algorithm to calibrate the latency of the output audio.

    Return alongside the manual latency that the user provided if applicable.
    """
    if input_version.major == 1:
        calibrate, plot = _calibrate_latency_v1, _plot_latency_v1
    elif input_version.major == 2:
        calibrate, plot = _calibrate_latency_v2, _plot_latency_v2
    elif input_version.major == 3:
        calibrate, plot = _calibrate_latency_v3, _plot_latency_v3
    elif input_version.major == 4:
        calibrate, plot = _calibrate_latency_v4, _plot_latency_v4
    elif input_version.major == 5:
        calibrate, plot = _calibrate_latency_v5, _plot_latency_v5
    elif input_version.major == 6:
        calibrate, plot = _calibrate_latency_v6, _plot_latency_v6
    elif input_version.major == 7:
        calibrate, plot = _calibrate_latency_v3, _plot_latency_v7
    elif input_version.major == 8:
        calibrate, plot = _calibrate_latency_v8, _plot_latency_v8
    else:
        raise NotImplementedError(
            f"Input calibration not implemented for input version {input_version}"
        )
    if user_latency is not None:
        print(f"Delay is specified as {user_latency}")
    if input_version.major in (6, 8):
        calibration_output = calibrate(
            input_path=input_path,
            output_path=output_path,
            manual_available=user_latency is not None,
            show_plots=not silent,
            verbose=verbose,
            _override_suppress_plots=_override_suppress_plots,
        )
    else:
        calibration_output = calibrate(
            _wav_to_np(output_path),
            manual_available=user_latency is not None,
            show_plots=not silent,
            verbose=verbose,
            _override_suppress_plots=_override_suppress_plots,
        )
    if not silent and calibration_output.recommended is not None:
        plot(calibration_output.recommended, input_path, output_path)
    return _metadata.Latency(manual=user_latency, calibration=calibration_output)


def get_lstm_config(architecture):
    return {
        Architecture.STANDARD: {
            "num_layers": 1,
            "hidden_size": 24,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.LITE: {
            "num_layers": 2,
            "hidden_size": 8,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.FEATHER: {
            "num_layers": 1,
            "hidden_size": 16,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.NANO: {
            "num_layers": 1,
            "hidden_size": 12,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.LSTM_COMPRESSOR_HQ_48X3: {
            "input_size": 1,
            "num_layers": 3,
            "hidden_size": 48,
            "train_burn_in": 16384,
            "train_truncate": 16384,
        },
        Architecture.LSTM_COMPRESSOR_LIGHT_30X3: {
            "input_size": 1,
            "num_layers": 3,
            "hidden_size": 30,
            "train_burn_in": 16384,
            "train_truncate": 8192,
        },
        Architecture.LSTM_UHQ: {
            "input_size": 1,
            "num_layers": 4,
            "hidden_size": 64,
            "train_burn_in": 32768,
            "train_truncate": 32768,
        },
        Architecture.LSTM_TONEX_LIKE_16: {
            "input_size": 1,
            "num_layers": 1,
            "hidden_size": 16,
            "train_burn_in": 4096,
            "train_truncate": 2048,
        },
    }[architecture]


def get_causal_conv_lstm_config(architecture):
    return {
        Architecture.CAUSAL_CONV_LSTM_TONEX_128_16_2048: {
            "input_size": 1,
            "num_layers": 1,
            "hidden_size": 16,
            "input_conv_kernel_size": 128,
            "output_conv_kernel_size": 2048,
            "train_burn_in": 4096,
            "train_truncate": 2048,
        },
        Architecture.CAUSAL_CONV_LSTM_TONEX_HQ: {
            "input_size": 1,
            "num_layers": 2,
            "hidden_size": 48,
            "input_conv_kernel_size": 256,
            "output_conv_kernel_size": 4096,
            "train_burn_in": 8192,
            "train_truncate": 4096,
        },
    }[architecture]


def _esr_validation_replicate_msg(threshold: float) -> str:
    return (
        f"Validation replicates have a self-ESR of over {threshold}. "
        "Your gear doesn't sound like itself when played twice!\n\n"
        "Possible causes:\n"
        " * Your signal chain is too noisy.\n"
        " * There's a time-based effect (chorus, delay, reverb) turned on.\n"
        " * Some knob got moved while reamping.\n"
        " * You started reamping before the amp had time to warm up fully."
    )


def _check_v1(*args, **kwargs) -> _metadata.DataChecks:
    return _metadata.DataChecks(version=1, passed=True)


def _check_v2(
    input_path,
    output_path,
    delay: int,
    silent: bool,
    show_replicate_info: bool = False,
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V2 checks...")
        rate = _V2_DATA_INFO.rate
        y = _wav_to_tensor(output_path, rate=rate)
        t_blips = _V2_DATA_INFO.t_blips
        t_validate = _V2_DATA_INFO.t_validate
        y_val_1 = y[-(t_blips + 2 * t_validate) : -(t_blips + t_validate)]
        y_val_2 = y[-(t_blips + t_validate) : -t_blips]
        esr_replicate = _ESR(y_val_1, y_val_2).item()
        if show_replicate_info:
            print(_format_replicate_esr(esr_replicate))
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))

        # Do the blips line up?
        # If the ESR is too bad, then flag it.
        print("Checking blips...")

        def get_blips(y):
            """
            :return: [start/end,replicate]
            """
            i0, i1 = _V2_DATA_INFO.blip_locations[0]
            j0, j1 = _V2_DATA_INFO.blip_locations[1]

            i0, i1, j0, j1 = [i + delay for i in (i0, i1, j0, j1)]
            start = -10
            end = 1000
            blips = _torch.stack(
                [
                    _torch.stack([y[i0 + start : i0 + end], y[i1 + start : i1 + end]]),
                    _torch.stack([y[j0 + start : j0 + end], y[j1 + start : j1 + end]]),
                ]
            )
            return blips

        blips = get_blips(y)
        esr_0 = _ESR(blips[0][0], blips[0][1]).item()  # Within start
        esr_1 = _ESR(blips[1][0], blips[1][1]).item()  # Within end
        esr_cross_0 = _ESR(blips[0][0], blips[1][0]).item()  # 1st repeat, start vs end
        esr_cross_1 = _ESR(blips[0][1], blips[1][1]).item()  # 2nd repeat, start vs end

        print("  ESRs:")
        print(f"    Start     : {esr_0}")
        print(f"    End       : {esr_1}")
        print(f"    Cross (1) : {esr_cross_0}")
        print(f"    Cross (2) : {esr_cross_1}")

        esr_threshold = 1.0e-2

        def plot_esr_blip_error(
            show_plot: bool,
            msg: str,
            arrays: _Sequence[_Sequence[float]],
            labels: _Sequence[str],
        ):
            """
            :param silent: Whether to make and show a plot about it
            """
            if show_plot:
                _plt.figure()
                [_plt.plot(array, label=label) for array, label in zip(arrays, labels)]
                _plt.xlabel("Sample")
                _plt.ylabel("Output")
                _plt.legend()
                _plt.grid()
            print(msg)
            if show_plot:
                _plt.show()
            print(
                "This is known to be a very sensitive test, so training will continue. "
                "If the model doesn't look good, then this may be why!"
            )

        # Check consecutive blips
        show_blip_plots = False
        for e, blip_pair, when in zip((esr_0, esr_1), blips, ("start", "end")):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed consecutive blip check at {when} of training signal. The "
                    "target tone doesn't seem to be replicable over short timespans."
                    "\n\n"
                    "  Possible causes:\n\n"
                    "    * Your recording setup is really noisy.\n"
                    "    * There's a noise gate that's messing things up.\n"
                    "    * There's a time-based effect (chorus, delay, reverb) in "
                    "the signal chain",
                    blip_pair,
                    ("Replicate 1", "Replicate 2"),
                )
                return _metadata.DataChecks(version=2, passed=False)
        # Check blips between start & end of train signal
        for e, blip_pair, replicate in zip(
            (esr_cross_0, esr_cross_1), blips.permute(1, 0, 2), (1, 2)
        ):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed start-to-end blip check for blip replicate {replicate}. "
                    "The target tone doesn't seem to be same at the end of the reamp "
                    "as it was at the start. Did some setting change during reamping?",
                    blip_pair,
                    (f"Start, replicate {replicate}", f"End, replicate {replicate}"),
                )
                return _metadata.DataChecks(version=2, passed=False)
        return _metadata.DataChecks(version=2, passed=True)


def _check_v3(
    input_path,
    output_path,
    silent: bool,
    *args,
    show_replicate_info: bool = False,
    **kwargs,
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V3 checks...")
        rate = _V3_DATA_INFO.rate
        y = _wav_to_tensor(output_path, rate=rate)
        n = len(_wav_to_tensor(input_path)) # to End-crop output
        # first 9 seconds
        y_val_1 = y[: _V3_DATA_INFO.t_validate]
        # last 9 seconds
        y_val_2 = y[n - _V3_DATA_INFO.t_validate : n]
        esr_replicate = _ESR(y_val_1, y_val_2).item()
        if show_replicate_info:
            print(_format_replicate_esr(esr_replicate))
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                _plt.figure()
                t = _np.arange(len(y_val_1)) / rate
                _plt.plot(t, y_val_1, label="Validation 1")
                _plt.plot(t, y_val_2, label="Validation 2")
                _plt.xlabel("Time (sec)")
                _plt.legend()
                _plt.title("V3 check: Validation replicate FAILURE")
                _plt.show()
            return _metadata.DataChecks(version=3, passed=False)
    return _metadata.DataChecks(version=3, passed=True)


def _check_v4(
    input_path, output_path, silent: bool, *args, **kwargs
) -> _metadata.DataChecks:
    # Things we can't check:
    # Latency compensation agreement
    # Data replicability
    print("Using Proteus audio file. Standard data checks aren't possible!")
    signal, info = _wav_to_np(output_path, info=True)
    passed = True
    if info.rate != _V4_DATA_INFO.rate:
        print(
            f"Output signal has sample rate {info.rate}; expected {_V4_DATA_INFO.rate}!"
        )
        passed = False
    # I don't care what's in the files except that they're long enough to hold the blip
    # and the last 10 seconds I decided to use as validation
    required_length = int((1.0 + 10.0) * _V4_DATA_INFO.rate)
    if len(signal) < required_length:
        print(
            "File doesn't meet the minimum length requirements for latency compensation and validation signal!"
        )
        passed = False
    return _metadata.DataChecks(version=4, passed=passed)


def _check_v5(
    input_path,
    output_path,
    silent: bool,
    *args,
    show_replicate_info: bool = False,
    **kwargs,
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V5 checks...")
        rate = _V5_DATA_INFO.rate

        wet_audio_tensor = _wav_to_tensor(output_path, rate=rate)
        print(f"Wet audio input Tensor length = {len(wet_audio_tensor)}")
        # Minutes:Seconds.Samples
        # (00:00.000 - 00:02.000) = Silence [0:96_000]
        # (00:02.000 - 01:10.000) = Validation 1 [96_000:3_360_000] 1:08 length, 70 * 48000 = 3360000
        # (01:10.000 - 01:10.499) = Silence before Blip 1 [3_360_000:3_384_000]
        # (01:10.500)             = Blip 1 @ 3_384_000
        # (01:10.501 - 01:11:499) = Silence between Blip 1 and Blip 2
        # (01:11.500)             = Blip 2 @ 3_432_000
        # (01:11.501 - 01:12.000) = Silence between Blip 2 and Operator white noise C3 (discontinuity?)
        # (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
        # (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
        # (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
        # (10:12.500 - 10:14.000) = Silence
        # (10:14.000)             = EOF @ 29_472_000
        #
        # t_blips=96_000
        # first_blips_start = 3_360_000
        # t_validate=3_264_000,         # 01:08.000 = 68 * 48000 = 3_264_000
        # train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
        # validation_start=-3_360_000,  # ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position.
        # noise_interval=(3_372_000, 3_378_000), # silence for 125ms in length, 250ms before the first blip.
        # blip_locations=((3_384_000, 3_432_000),), # Actual locations of each of the two blips.
        dry_audio_tensor_len = len(_wav_to_tensor(input_path))
        # 00:01.500 - 01:10.000
        wet_audio_validation_segment_1 = wet_audio_tensor[72_000:3_360_000]
        # 00:00.500 of silence after 09:04:000
        # 01:08.000 of validation #2
        # 00:01.500 of silence til EOF
        # = total time of 01:10:000 = 3_360_000
        # 29_472_000 - 3_360_000 = 26_112_000
        validation_start_time_in_samples = dry_audio_tensor_len + _V5_DATA_INFO.validation_start
        validation_stop_time_in_samples = dry_audio_tensor_len - 72_000
        wet_audio_validation_segment_2 = wet_audio_tensor[validation_start_time_in_samples:validation_stop_time_in_samples]
        # Validation (replicate) ESR =
        #   * how much does wet_audio_validation_segment_1 differ from wet_audio_validation_segment_2?
        #   * does either segment drift off-grid from their expected positions relative to the start and end timings of the dry/wet audio files?
        #
        # If there is a big difference, it's because these two copies of the same exact input audio segment run through the physical signal chain at two different times produces inconsistent results.
        # Or it's because there's something mucking up the timing, like an echo or delay effect.
        # There is a threshold for this because obviously analog gear is not deterministic, and maybe your tubes warmed up a bit between the two takes.
        esr_replicate = _ESR(wet_audio_validation_segment_1, wet_audio_validation_segment_2).item()
        if show_replicate_info:
            print(_format_replicate_esr(esr_replicate))
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                _plt.figure()
                t = _np.arange(len(wet_audio_validation_segment_1)) / rate
                _plt.plot(t, wet_audio_validation_segment_1, label="Validation 1")
                _plt.plot(t, wet_audio_validation_segment_2, label="Validation 2")
                _plt.xlabel("Time (sec)")
                _plt.legend()
                _plt.title("V5 check: Validation replicate FAILURE")
                _plt.show()
            return _metadata.DataChecks(version=5, passed=False)
    return _metadata.DataChecks(version=5, passed=True)


def _apply_delay_to_pair(
    dry_audio_tensor: _torch.Tensor,
    wet_audio_tensor: _torch.Tensor,
    delay: int,
) -> _Tuple[_torch.Tensor, _torch.Tensor]:
    # Mirror nam.data.Dataset._apply_delay_int() so this check sees the same
    # alignment as the training dataset.
    if delay > 0:
        dry_audio_tensor = dry_audio_tensor[:-delay]
        wet_audio_tensor = wet_audio_tensor[delay:]
    elif delay < 0:
        dry_audio_tensor = dry_audio_tensor[-delay:]
        wet_audio_tensor = wet_audio_tensor[:delay]
    return dry_audio_tensor, wet_audio_tensor


def _match_level(
    reference: _torch.Tensor, target: _torch.Tensor
) -> _Tuple[_torch.Tensor, float]:
    denom = _torch.dot(reference, reference).item()
    scale = 1.0 if denom == 0.0 else _torch.dot(reference, target).item() / denom
    return reference * scale, scale


def _find_v6_self_validation(
    input_path: str, output_path: str, delay: int
) -> _Optional[_AP106SelfValidation]:
    rate = int(_V6_DATA_INFO.rate)
    dry_audio_tensor = _wav_to_tensor(input_path, rate=rate)
    wet_audio_tensor = _wav_to_tensor(output_path, rate=rate)
    pair_length = min(len(dry_audio_tensor), len(wet_audio_tensor))
    train_stop = pair_length + _V6_DATA_INFO.validation_start
    if train_stop <= _V6_DATA_INFO.train_start:
        return None

    dry_audio_tensor = dry_audio_tensor[_V6_DATA_INFO.train_start:train_stop]
    wet_audio_tensor = wet_audio_tensor[_V6_DATA_INFO.train_start:train_stop]
    dry_audio_tensor, wet_audio_tensor = _apply_delay_to_pair(
        dry_audio_tensor, wet_audio_tensor, delay
    )

    window_samples = _AP106_SELF_VALIDATION_WINDOW_SAMPLES
    if len(dry_audio_tensor) < 2 * window_samples:
        return None

    dry_windows = []
    wet_windows = []
    starts = []
    rms_values = []
    for start in range(
        0,
        len(dry_audio_tensor) - window_samples + 1,
        _AP106_SELF_VALIDATION_HOP_SAMPLES,
    ):
        dry_window = dry_audio_tensor[start : start + window_samples]
        rms = _torch.sqrt(_torch.mean(_torch.square(dry_window))).item()
        if rms < _AP106_SELF_VALIDATION_MIN_RMS:
            continue
        dry_windows.append(dry_window)
        wet_windows.append(wet_audio_tensor[start : start + window_samples])
        starts.append(start)
        rms_values.append(rms)

    if len(dry_windows) < 2:
        return None

    dry_windows = _torch.stack(dry_windows)
    wet_windows = _torch.stack(wet_windows)
    norms = _torch.linalg.vector_norm(dry_windows, dim=1, keepdim=True)
    dry_windows = dry_windows / norms
    similarities = dry_windows @ dry_windows.T
    starts_tensor = _torch.tensor(starts, device=similarities.device)
    overlap_mask = (
        _torch.abs(starts_tensor[:, None] - starts_tensor[None, :]) < window_samples
    )
    similarities[overlap_mask] = float("-inf")
    if not _torch.isfinite(similarities).any().item():
        return None

    best_similarity = _torch.max(similarities).item()
    candidate_pairs = _torch.nonzero(
        similarities
        >= best_similarity - _AP106_SELF_VALIDATION_SIMILARITY_TOLERANCE,
        as_tuple=False,
    )
    best_candidate = None
    for i, j in candidate_pairs.tolist():
        wet_reference, _ = _match_level(wet_windows[i], wet_windows[j])
        wet_esr = _ESR(wet_reference, wet_windows[j]).item()
        if best_candidate is None or wet_esr < best_candidate[0]:
            best_candidate = (wet_esr, i, j)

    if best_candidate is None:
        return None

    wet_esr, i, j = best_candidate
    start_offset_samples = _V6_DATA_INFO.train_start + max(-delay, 0)
    return _AP106SelfValidation(
        window_seconds=window_samples / rate,
        start_1_seconds=(start_offset_samples + starts[i]) / rate,
        start_2_seconds=(start_offset_samples + starts[j]) / rate,
        dry_scale_invariant_esr=max(0.0, 1.0 - best_similarity**2),
        dry_rms_ratio=max(rms_values[i], rms_values[j]) / min(rms_values[i], rms_values[j]),
        wet_esr=wet_esr,
    )


def _format_v6_self_validation(self_validation: _AP106SelfValidation) -> str:
    return (
        "AP106 approximate self-validation: "
        f"best repeated {self_validation.window_seconds:.2f}s dry window at "
        f"{self_validation.start_1_seconds:.2f}s and "
        f"{self_validation.start_2_seconds:.2f}s, "
        f"dry self-ESR {self_validation.dry_scale_invariant_esr:.8f}, "
        f"dry RMS ratio {self_validation.dry_rms_ratio:.3f}, "
        f"gain-matched wet self-ESR {self_validation.wet_esr:.8f}."
    )


def _esr_validation_v6_msg(threshold: float) -> str:
    return (
        f"AP106 approximate self-validation wet ESR is over {threshold}. "
        "The best repeated dry phrase does not reamp consistently enough.\n\n"
        "Possible causes:\n"
        " * Your signal chain is too noisy.\n"
        " * There's a time-based effect (chorus, delay, reverb) turned on.\n"
        " * Some knob got moved while reamping.\n"
        " * The amp or cab changed audibly as it warmed up."
    )



def _check_v6(
    input_path,
    output_path,
    delay: _Optional[int],
    silent: bool,
    *args,
    show_replicate_info: bool = False,
    **kwargs,
) -> _metadata.DataChecks:
    print(
        "Using AP106/AM4 audio file. "
        "Latency calibration correlates the known AP106 startup pulse against "
        "the wet file and reports the result on NAM's standard v3 latency scale."
    )
    signal, info = _wav_to_np(output_path, info=True)
    passed = True
    if info.rate != _V6_DATA_INFO.rate:
        print(
            f"Output signal has sample rate {info.rate}; expected {_V6_DATA_INFO.rate}!"
        )
        passed = False
    required_length = _V6_DATA_INFO.train_start + _V6_DATA_INFO.t_validate
    if len(signal) < required_length:
        print(
            "File doesn't meet the minimum length requirements for AP106/AM4 validation!"
        )
        passed = False

    if not passed:
        return _metadata.DataChecks(version=6, passed=False)

    if delay is None:
        if show_replicate_info:
            print(
                "AP106 approximate self-validation skipped because latency could not "
                "be determined."
            )
        return _metadata.DataChecks(version=6, passed=True)

    with _torch.no_grad():
        self_validation = _find_v6_self_validation(input_path, output_path, delay)
    if self_validation is None:
        if show_replicate_info:
            print(
                "AP106 approximate self-validation skipped because no two "
                "non-overlapping, non-silent repeated dry windows were found."
            )
        return _metadata.DataChecks(version=6, passed=True)

    if show_replicate_info:
        print(_format_v6_self_validation(self_validation))
    if (
        self_validation.dry_scale_invariant_esr
        <= _AP106_SELF_VALIDATION_DRY_ESR_THRESHOLD
        and self_validation.wet_esr > _AP106_SELF_VALIDATION_WET_ESR_THRESHOLD
    ):
        print(_esr_validation_v6_msg(_AP106_SELF_VALIDATION_WET_ESR_THRESHOLD))
        return _metadata.DataChecks(version=6, passed=False)
    return _metadata.DataChecks(version=6, passed=True)


def _check_v7(
    input_path,
    output_path,
    delay: _Optional[int],
    silent: bool,
    *args,
    show_replicate_info: bool = False,
    **kwargs,
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("Short TTS-style checks...")
        rate = _TTS_SHORT_DATA_INFO.rate
        y = _wav_to_tensor(output_path, rate=rate)
        n = len(_wav_to_tensor(input_path))
        y_val_1 = y[: _TTS_SHORT_DATA_INFO.t_validate]
        y_val_2 = y[n - _TTS_SHORT_DATA_INFO.t_validate : n]
        esr_replicate = _ESR(y_val_1, y_val_2).item()
        if show_replicate_info:
            print(_format_replicate_esr(esr_replicate))
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                _plt.figure()
                t = _np.arange(len(y_val_1)) / rate
                _plt.plot(t, y_val_1, label="Validation 1")
                _plt.plot(t, y_val_2, label="Validation 2")
                _plt.xlabel("Time (sec)")
                _plt.legend()
                _plt.title("Short TTS-style check: Validation replicate FAILURE")
                _plt.show()
            return _metadata.DataChecks(version=7, passed=False)
    return _metadata.DataChecks(version=7, passed=True)


def _check_v8(
    input_path,
    output_path,
    delay: _Optional[int],
    silent: bool,
    *args,
    show_replicate_info: bool = False,
    **kwargs,
) -> _metadata.DataChecks:
    print(
        "Using Super Input v2 audio file. Latency calibration uses known "
        "dry envelope anchors because this file has no NAM impulse pair."
    )
    signal, info = _wav_to_np(output_path, info=True)
    passed = True
    if info.rate != _SUPER_INPUT_V2_DATA_INFO.rate:
        print(
            f"Output signal has sample rate {info.rate}; expected "
            f"{_SUPER_INPUT_V2_DATA_INFO.rate}!"
        )
        passed = False
    required_length = max(
        max(anchor.template_stop for anchor in _SUPER_INPUT_V2_ANCHORS),
        _SUPER_INPUT_V2_DATA_INFO.t_validate * 2,
    )
    if len(signal) < required_length:
        print(
            "File does not meet the minimum length requirements for Super "
            "Input v2 anchor calibration and validation!"
        )
        passed = False
    if show_replicate_info:
        print(
            "Super Input v2 has no repeated validation block; standard "
            "repeatability ESR checks are skipped."
        )
    return _metadata.DataChecks(version=8, passed=passed)


def _check_data(
    input_path: str,
    output_path: str,
    input_version: _Version,
    delay: int,
    silent: bool,
    show_replicate_info: bool = False,
) -> _Optional[_metadata.DataChecks]:
    """
    Ensure that everything should go smoothly

    :return: True if looks good
    """
    if input_version.major == 1:
        f = _check_v1
    elif input_version.major == 2:
        f = _check_v2
    elif input_version.major == 3:
        f = _check_v3
    elif input_version.major == 4:
        f = _check_v4
    elif input_version.major == 5:
        f = _check_v5
    elif input_version.major == 6:
        f = _check_v6
    elif input_version.major == 7:
        f = _check_v7
    elif input_version.major == 8:
        f = _check_v8
    else:
        print(f"Checks not implemented for input version {input_version}; skip")
        return None
    out = f(
        input_path,
        output_path,
        delay,
        silent,
        show_replicate_info=show_replicate_info,
    )
    # Issue 442: Deprecate inputs
    if input_version.major in [1, 2]:
        print(
            f"Input version {input_version} is deprecated and will be removed in "
            "version 0.11 of the trainer. To continue using it, you must ignore checks."
        )
        out.passed = False
    return out


def get_wavenet_config(architecture):
    architecture = Architecture(architecture)
    return {
        Architecture.ULTRA: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 24,
                    "head_size": 20,
                    "kernel_size": 5,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                    ],
                    "activation": {
                        "name": "PairMultiply",
                        "primary": "Tanh",
                        "secondary": "Sigmoid",
                    },
                    "bottleneck": 16,
                    "head_bias": False,
                },
                {
                    "input_size": 24,
                    "condition_size": 1,
                    "channels": 20,
                    "head_size": 10,
                    "kernel_size": 5,
                    "dilations": [
                        1,
                        5,
                        25,
                        125,
                        625,
                        3125,
                        1,
                        5,
                        25,
                        125,
                        625,
                    ],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "input_size": 20,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                    ],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        # head_size -> channels
        # channels  -> input_size
        #
        # head 3 -> channels 3 -> input 3
        Architecture.COMPLEX: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 32,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 32,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.COMPLEXRF300: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 32,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 32,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 512, 256, 128, 64, 32, 32, 32, 2],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.COMPLEXRF300LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 26,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 26,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 512, 256, 128, 64, 32, 32, 32, 2],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.COMPLEXRF600: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 32,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 32,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 1, 2, 4, 8, 16, 32, 4096],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.COMPLEXRF600LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 20,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 20,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 1, 2, 4, 8, 16, 32, 4096],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XCOMPLEX: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 32,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 32,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XCOMPLEX_LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 20,
                    "head_size": 6,
                    "kernel_size": 3,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 20,
                    "condition_size": 1,
                    "channels": 6,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.REVYHI: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                }
            ],
            "head_scale": 0.99
        },
        # Recommended settings: http://coginthemachine.ddns.net/mnt/_namhtml/wnet/namconfig.html
        # "pre_emph_mrstft_weight": 0.0002 # Same as the default of _CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
        # "pre_emph_mrstft_coef": 0.85     # Same as the default of _CAB_MRSTFT_PRE_EMPH_COEF = 0.85
        # "lr_scheduler": {{"gamma": 0.9985}
        Architecture.REVYSTD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True
                }
            ],
            "head_scale": 0.99
        },
        Architecture.REVYLO: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 4,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 4,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 4,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 4,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                }
            ],
            "head_scale": 0.99
        },
        # Original uses different settings:
        # https://github.com/sdatkinson/neural-amp-modeler/compare/main...38github:neural-amp-modeler:main#diff-2ca4268576a15c724ac7227c5ae5e17ed1e70edd2a86ce943497e82973f91030R921
        # "head_scale": 0.50,
        # _CAB_MRSTFT_PRE_EMPH_WEIGHT = 1.5e-3 # Different from the default of _CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
        # _CAB_MRSTFT_PRE_EMPH_COEF = 0.15     # Different from the default of _CAB_MRSTFT_PRE_EMPH_COEF = 0.85
        # "loss": {
        #     "type": "combined_loss",
        #     "loss_weights": {"spectral_mse": 0.9, "perceptual_loss": 0.1, "low_freq_weight": 2.0, "high_freq_weight": 1.0},
        # },
        Architecture.REVXSTD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True
                }
            ],
            "head_scale": 0.99
        },
        Architecture.XSTD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [1, 3, 9, 27, 81, 243, 729],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [1, 3, 9, 27, 81, 243, 729],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 15,
                    "dilations": [1, 16],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [1, 3, 9, 27, 81, 243, 729],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.99,
        },
        Architecture.XSTD3: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XHI3: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XHV_12: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 12,
                    "head_size": 12,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 12,
                    "condition_size": 1,
                    "channels": 12,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XHV_16: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 16,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 16,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XHV_24: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 24,
                    "head_size": 24,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 24,
                    "condition_size": 1,
                    "channels": 24,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.XHQ: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 16,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 16,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.UHQ: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 22,
                    "head_size": 23,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125, 625, 3125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 22,
                    "condition_size": 1,
                    "channels": 23,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        1, 5, 25, 125,
                        625,
                    ],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.STANDARD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 16,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 12,
                    "head_size": 6,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 12,
                    "channels": 6,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.FEATHER: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 4,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 4,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.NANO: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 2,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 2,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.NANO64X4: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 7,
                    "dilations": [1, 8, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 7,
                    "dilations": [1, 8, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 4,
                    "head_size": 2,
                    "kernel_size": 7,
                    "dilations": [1, 8, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 2,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [512, 1, 8, 64, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.36,
        },
        Architecture.NANO125X3: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 4,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 4,
                    "head_size": 2,
                    "kernel_size": 6,
                    "dilations": [1, 5, 25, 125],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 2,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [625, 1, 5, 25, 125, 625],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.36,
        },
        Architecture.DOUBLE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 379, 482],
                    "activation": "LeakyReLU",
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": True,
                    },
                },
            ],
            "head_scale": 0.01,
        },
        Architecture.XDOUBLE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 643, 953],
                    "activation": "LeakyReLU",
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": True,
                    },
                },
            ],
            "head_scale": 0.01,
        },
        Architecture.YDOUBLE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 19,
                    "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 643, 953],
                    "activation": "LeakyReLU",
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": True,
                    },
                },
            ],
            "head_scale": 0.01,
        },
        Architecture.DOUBLEPLUS: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 15, 6, 6],
                    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 11, 23, 643, 953],
                    "activation": "LeakyReLU",
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": True,
                    },
                },
            ],
            "head_scale": 0.01,
        },
        # Standalone A2 Complex branch: copied from the channels_16_complex
        # submodel in config_model_packed_complex.json without the Lite branch.
        Architecture.ACOMPLEX: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 577, 811],
                    "activation": "LeakyReLU",
                    "gated": False,
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": True,
                    },
                },
            ],
            "head_scale": 0.01,
        },
    }[architecture]


def _get_data_config(
    input_version: _Version,
    input_path: _Path,
    output_path: _Path,
    ny: int,
    latency: int,
) -> dict:
    def get_split_kwargs(data_info: _DataInfo):
        if data_info.major_version == 1:
            train_val_split = data_info.validation_start
            train_kwargs = {"stop_samples": train_val_split}
            validation_kwargs = {"start_samples": train_val_split}
        elif data_info.major_version == 2:
            validation_start = data_info.validation_start
            train_stop = validation_start
            validation_stop = validation_start + data_info.t_validate
            train_kwargs = {"stop_samples": train_stop}
            validation_kwargs = {
                "start_samples": validation_start,
                "stop_samples": validation_stop,
            }
        elif data_info.major_version in (3, 7):
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {
                "start_samples": data_info.train_start,
                "stop_samples": train_stop,
            }
            # The repeated validation block at the end of these v3-style inputs
            # does not have the same preceding silence guarantee as the training
            # split.
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": None,
            }
        elif data_info.major_version == 4:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"stop_samples": train_stop}
            # Proteus doesn't have silence to get a clean split. Bite the bullet.
            print(
                "Using Proteus files:\n"
                " * There isn't a silent point to split the validation set, so some of "
                "your gear's response from the train set will leak into the start of "
                "the validation set and impact validation accuracy (Bypassing data "
                "quality check)\n"
                " * Since the validation set is different, the ESRs reported for this "
                "model aren't comparable to those from the other 'NAM' training files."
            )
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": False,
            }
        elif data_info.major_version == 5:
            validation_start = data_info.validation_start
            train_stop = validation_start
            # validation_start=-3_360_000,  # -01:09.500 = 1:08.000 + 1.5 seconds trailing silence.
            # ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position.
            # (29_472_000 - (((9 * 60) + 4.5) * 48000)) / 48000 = 69.5
            #
            # train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
            #
            # Minutes:Seconds.Samples
            # (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
            # (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
            # (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
            print(f"train_kwargs start_samples = {data_info.train_start} and stop_samples = {train_stop}")
            train_kwargs = {"start_samples": data_info.train_start, "stop_samples": train_stop}
            validation_kwargs = {"start_samples": validation_start}
        elif data_info.major_version == 6:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {
                "start_samples": data_info.train_start,
                "stop_samples": train_stop,
                "require_input_pre_silence": False,
            }
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": None,
            }
        elif data_info.major_version == 8:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"stop_samples": train_stop}
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": None,
            }
        else:
            raise NotImplementedError(f"kwargs for input version {input_version}")
        return train_kwargs, validation_kwargs

    data_info = {
        1: _V1_DATA_INFO,
        2: _V2_DATA_INFO,
        3: _V3_DATA_INFO,
        4: _V4_DATA_INFO,
        5: _V5_DATA_INFO,
        6: _V6_DATA_INFO,
        7: _TTS_SHORT_DATA_INFO,
        8: _SUPER_INPUT_V2_DATA_INFO,
    }[input_version.major]

    train_kwargs, validation_kwargs = get_split_kwargs(data_info)
    data_config = {
        "train": {"ny": ny, **train_kwargs},
        "validation": {"ny": None, **validation_kwargs},
        "common": {
            "x_path": input_path,
            "y_path": output_path,
            "delay": latency,
            "allow_unequal_lengths": True,
        },
    }
    return data_config


def _get_packed_model_config(
    architecture: _Union[Architecture, str] = Architecture.A2_FULL_LITE,
) -> dict:
    resource_name = {
        Architecture.A2_FULL_LITE: "config_model_packed.json",
        Architecture.A2_COMPLEX_LITE: "config_model_packed_complex.json",
        Architecture.A2_COMPLEX_REVYLO: "config_model_packed_complex_revylo.json",
        Architecture.A2_COMPLEX_NANO64X4: "config_model_packed_complex_nano64x4.json",
        Architecture.A2_COMPLEX_NANO125X3: "config_model_packed_complex_nano125x3.json",
        Architecture.A2_DOUBLE_LITE: "config_model_packed_double_lite.json",
        Architecture.A2_XDOUBLE_LITE: "config_model_packed_xdouble_lite.json",
        Architecture.A2_DOUBLEPLUS_LITE: "config_model_packed_doubleplus_lite.json",
    }[Architecture(architecture)]
    resource = _resources.files("nam.train._resources").joinpath(resource_name)
    with resource.open() as fp:
        return _deepcopy(_json.load(fp))


def _get_lightning_module_cls(model_config: _Dict):
    return (
        _PackedLightningModule
        if model_config["net"]["name"] == "PackedWaveNet"
        else _LightningModule
    )


def _get_configs(
    input_version: _Version,
    input_path: str,
    output_path: str,
    latency: int,
    epochs: int,
    model_type: str,
    architecture: Architecture,
    ny: int,
    lr: float,
    lr_decay: float,
    batch_size: int,
    fit_mrstft: bool,
    lr_scheduler_type: _Union[LearningRateScheduler, str],
    checkpoint_path: _Optional[_Union[str, _Sequence[_Optional[str]]]] = None,
    checkpoint_paths_by_submodel: _Optional[_Sequence[_Optional[str]]] = None,
    stage_two_focus: _Optional[_Union[StageTwoFocus, str]] = None,
):
    def _scheduler_warmup_epochs(
        scheduler_type: LearningRateScheduler,
        total_epochs: int,
        scheduler_architecture: Architecture,
    ) -> int:
        total_epochs = max(1, int(total_epochs))
        if scheduler_type == LearningRateScheduler.ONE_CYCLE:
            fraction = 0.18
            max_warmup = 80
        elif _is_lstm_only_architecture(scheduler_architecture):
            fraction = 0.08
            max_warmup = 35
        elif _is_causal_conv_lstm_architecture(scheduler_architecture):
            fraction = 0.10
            max_warmup = 45
        else:
            fraction = 0.06
            max_warmup = 50
        return min(max_warmup, max(1, int(round(total_epochs * fraction))))

    def _plateau_patience(total_epochs: int) -> int:
        total_epochs = max(1, int(total_epochs))
        if total_epochs <= 150:
            return 8
        if total_epochs <= 400:
            return 12
        if total_epochs <= 800:
            return 18
        return 25

    def _scheduler_monitor(default_monitor: str) -> str:
        return "val_loss" if default_monitor == "" else default_monitor

    def _scheduler_config_for(
        scheduler_type: LearningRateScheduler,
        total_epochs: int,
        scheduler_lr: float,
        scheduler_lr_decay: float,
        scheduler_architecture: Architecture,
        monitor: str,
    ) -> _Dict:
        scheduler_lr_decay = max(1.0e-8, float(scheduler_lr_decay))
        monitor = _scheduler_monitor(monitor)
        if scheduler_type == LearningRateScheduler.EXPONENTIAL:
            return {
                "class": "ExponentialLR",
                "name": "ExponentialLR",
                "kwargs": {"gamma": 1.0 - scheduler_lr_decay},
            }
        if scheduler_type == LearningRateScheduler.COSINE_ANNEALING:
            return {
                "class": "CosineAnnealingLR",
                "name": "CosineAnnealingLR",
                "kwargs": {
                    "T_max": max(1, total_epochs),
                    "eta_min": max(0.0, min(scheduler_lr, scheduler_lr * scheduler_lr_decay)),
                },
            }
        if scheduler_type == LearningRateScheduler.COSINE_ANNEALING_WARM_RESTARTS:
            return {
                "class": "CosineAnnealingWarmRestarts",
                "name": "CosineAnnealingWarmRestarts",
                "kwargs": {
                    "T_0": max(10, total_epochs // 10),
                    "T_mult": 2,
                    "eta_min": max(0.0, min(scheduler_lr, scheduler_lr * scheduler_lr_decay)),
                },
            }
        if scheduler_type == LearningRateScheduler.WARMUP_COSINE_DECAY:
            warmup_epochs = min(
                max(0, total_epochs - 1),
                _scheduler_warmup_epochs(
                    scheduler_type, total_epochs, scheduler_architecture
                ),
            )
            return {
                "class": "WarmupCosineDecay",
                "name": "Warmup + Cosine Decay",
                "kwargs": {
                    "warmup_epochs": warmup_epochs,
                    "total_epochs": max(1, total_epochs),
                    "start_factor": 0.10,
                    "eta_min": max(0.0, min(scheduler_lr, scheduler_lr * scheduler_lr_decay)),
                },
                "interval": "epoch",
            }
        if scheduler_type == LearningRateScheduler.ONE_CYCLE:
            warmup_epochs = _scheduler_warmup_epochs(
                scheduler_type, total_epochs, scheduler_architecture
            )
            pct_start = max(0.05, min(0.35, warmup_epochs / max(1, total_epochs)))
            div_factor = 10.0
            if _is_lstm_only_architecture(scheduler_architecture):
                div_factor = 8.0
            elif _is_causal_conv_lstm_architecture(scheduler_architecture):
                div_factor = 12.0
            final_div_factor = max(1.01, 1.0 / (div_factor * scheduler_lr_decay))
            return {
                "class": "OneCycleLR",
                "name": "OneCycleLR",
                "kwargs": {
                    "max_lr": scheduler_lr,
                    "pct_start": pct_start,
                    "anneal_strategy": "cos",
                    "div_factor": div_factor,
                    "final_div_factor": final_div_factor,
                    "use_estimated_stepping_batches": True,
                    "fallback_total_steps": max(1, total_epochs),
                },
                "interval": "step",
                "frequency": 1,
            }
        if scheduler_type == LearningRateScheduler.LINEAR_WARMUP_REDUCE_ON_PLATEAU:
            warmup_epochs = _scheduler_warmup_epochs(
                scheduler_type, total_epochs, scheduler_architecture
            )
            return {
                "class": "LinearWarmupReduceLROnPlateau",
                "name": "Linear Warmup + ReduceLROnPlateau",
                "kwargs": {
                    "warmup_epochs": min(max(0, total_epochs - 1), warmup_epochs),
                    "start_factor": 0.10,
                    "mode": "min",
                    "factor": min(0.99, max(0.01, scheduler_lr_decay)),
                    "patience": _plateau_patience(total_epochs) + warmup_epochs,
                    "cooldown": max(0, warmup_epochs // 2),
                    "min_lr": max(1.0e-8, scheduler_lr * 0.001),
                },
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            }
        return {
            "class": "ReduceLROnPlateau",
            "name": "ReduceLROnPlateau",
            "kwargs": {
                "mode": "min",
                "factor": scheduler_lr_decay,
                "patience": 5,
            },
            "monitor": monitor,
        }

    def _scheduler_uses_plateau(config: _Dict) -> bool:
        return config.get("class") in (
            "ReduceLROnPlateau",
            "LinearWarmupReduceLROnPlateau",
        )

    def _get_stage_two_focus_config(focus: _Union[StageTwoFocus, str]) -> _Dict:
        focus = StageTwoFocus(focus)
        custom_loss_name = {
            StageTwoFocus.LOW: "LOW_BAND_MSE",
            StageTwoFocus.MID: "MID_BAND_MSE",
            StageTwoFocus.HIGH: "HIGH_BAND_MSE",
        }[focus]
        band_kwargs = {
            StageTwoFocus.LOW: {"low_hz": None, "high_hz": 250.0, "weight": 0.5},
            StageTwoFocus.MID: {"low_hz": 250.0, "high_hz": 2_000.0, "weight": 0.35},
            StageTwoFocus.HIGH: {"low_hz": 2_000.0, "high_hz": None, "weight": 0.25},
        }[focus]
        return {
            "custom_loss_name": custom_loss_name,
            "custom_losses": {
                custom_loss_name: {
                    "name": "nam.models.losses.band_mse",
                    "kwargs": {
                        "sample_rate": STANDARD_SAMPLE_RATE,
                        "low_hz": band_kwargs["low_hz"],
                        "high_hz": band_kwargs["high_hz"],
                    },
                    "weight": band_kwargs["weight"],
                }
            },
            "val_loss": "esr",
            "monitor": "ESR",
            "disable_preemph_mrstft": focus in (StageTwoFocus.LOW, StageTwoFocus.MID),
        }

    # DATA CONFIG
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=ny,
        latency=latency,
    )
    lr_scheduler_type = LearningRateScheduler(lr_scheduler_type)
    if _is_a2_architecture(architecture):
        model_type = "PackedWaveNet"
        data_config["joint"] = [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ]
    elif _is_causal_conv_lstm_architecture(architecture):
        model_type = "CausalConvLSTM"
    elif model_type == "WaveNet" and _is_lstm_only_architecture(architecture):
        model_type = "LSTM"
    if checkpoint_paths_by_submodel is not None and model_type != "PackedWaveNet":
        raise ValueError(
            "Submodel checkpoint pairs are only supported for A2 PackedWaveNet."
        )
    # MODEL CONFIG
    if model_type == "PackedWaveNet":
        model_config = _get_packed_model_config(architecture)
        model_config["optimizer"]["lr"] = lr
        model_config["lr_scheduler"] = _scheduler_config_for(
            lr_scheduler_type,
            epochs,
            lr,
            lr_decay,
            architecture,
            monitor="val_loss",
        )
        if checkpoint_paths_by_submodel is not None:
            model_config["checkpoint_paths_by_submodel"] = list(checkpoint_paths_by_submodel)
        elif checkpoint_path is not None:
            model_config["checkpoint_path"] = checkpoint_path
        if stage_two_focus is not None:
            stage_two_config = _get_stage_two_focus_config(stage_two_focus)
            model_config["loss"]["val_loss"] = stage_two_config["val_loss"]
            model_config["loss"]["custom_losses"] = stage_two_config["custom_losses"]
            if _scheduler_uses_plateau(model_config["lr_scheduler"]):
                model_config["lr_scheduler"]["monitor"] = stage_two_config["monitor"]
        fit_mrstft = False
    elif model_type == "WaveNet":
        lr_scheduler_config = _scheduler_config_for(
            lr_scheduler_type,
            epochs,
            lr,
            lr_decay,
            architecture,
            monitor="ESRPREEMPH",
        )
        model_config = {
            "net": {
                "name": "WaveNet",
                # This should do decently. If you really want a nice model, try turning up
                # "channels" in the first block and "input_size" in the second from 12 to 16.
                "config": get_wavenet_config(architecture),
            },
            "loss": {"val_loss": "esrpreemph"},
            #"loss": {"val_loss": "esr"},
            "optimizer": {"lr": lr},
            "lr_scheduler": lr_scheduler_config,
        }
        if checkpoint_path is not None:
            model_config["checkpoint_path"] = checkpoint_path
        if stage_two_focus is not None:
            stage_two_config = _get_stage_two_focus_config(stage_two_focus)
            model_config["loss"]["val_loss"] = stage_two_config["val_loss"]
            model_config["loss"]["custom_losses"] = stage_two_config["custom_losses"]
            if _scheduler_uses_plateau(model_config["lr_scheduler"]):
                model_config["lr_scheduler"]["monitor"] = stage_two_config["monitor"]
            if stage_two_config["disable_preemph_mrstft"]:
                fit_mrstft = False
    elif model_type == "CausalConvLSTM":
        lstm_config = get_causal_conv_lstm_config(architecture)
        model_config = {
            "net": {
                "name": "CausalConvLSTM",
                "config": lstm_config,
            },
            "loss": {
                "val_loss": "mse",
                "mask_first": lstm_config.get("train_burn_in", 4096),
                "pre_emph_weight": 1.0,
                "pre_emph_coef": 0.85,
            },
            "optimizer": {"lr": lr},
            "lr_scheduler": _scheduler_config_for(
                lr_scheduler_type,
                epochs,
                lr,
                lr_decay,
                architecture,
                monitor="MSE",
            ),
        }
    else:
        lstm_config = get_lstm_config(architecture)
        model_config = {
            "net": {
                "name": "LSTM",
                "config": lstm_config,
            },
            "loss": {
                "val_loss": "mse",
                "mask_first": lstm_config.get("train_burn_in", 4096),
                "pre_emph_weight": 1.0,
                "pre_emph_coef": 0.85,
            },
            "optimizer": {"lr": lr},
            "lr_scheduler": _scheduler_config_for(
                lr_scheduler_type,
                epochs,
                lr,
                lr_decay,
                architecture,
                monitor="MSE",
            ),
        }
    if fit_mrstft:
        model_config["loss"]["pre_emph_mrstft_weight"] = _CAB_MRSTFT_PRE_EMPH_WEIGHT
        model_config["loss"]["pre_emph_mrstft_coef"] = _CAB_MRSTFT_PRE_EMPH_COEF

    # DEVICE CONFIG (part of learning_config)
    if _torch.cuda.is_available():
        device_config = {"accelerator": "gpu", "devices": 1}
    elif _torch.backends.mps.is_available():
        device_config = {"accelerator": "mps", "devices": 1}
    else:
        print("WARNING: No GPU was found. Training will be very slow!")
        device_config = {}
    # LEARNING CONFIG
    # Set `num_workers` to the same number of P-cores in your computer.
    learning_config = {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 6,
            "persistent_workers": True
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": epochs, **device_config},
    }
    return data_config, model_config, learning_config


def _get_dataloaders(
    data_config: _Dict, learning_config: _Dict, model: _LightningModule
) -> _Tuple[_DataLoader, _DataLoader]:
    data_config, learning_config = [
        _deepcopy(c) for c in (data_config, learning_config)
    ]
    data_config["common"]["nx"] = model.net.receptive_field
    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)
    _apply_joint_dataset_hooks(
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )
    model.net.sample_rate = dataset_train.sample_rate
    dataset_train.handshake(model.net)
    dataset_validation.handshake(model.net)
    model.net.handshake(dataset_train)
    model.net.handshake(dataset_validation)
    train_dataloader = _DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = _DataLoader(dataset_validation, **learning_config["val_dataloader"])
    return train_dataloader, val_dataloader


def _esr(pred: _torch.Tensor, target: _torch.Tensor) -> float:
    return (_torch.mean(_torch.square(pred - target)).item() / _torch.mean(_torch.square(target)).item())


def _plot(
    model,
    ds,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
    filepath: _Optional[str] = None,
    silent: bool = False,
    packed_esrs: _Optional[list] = None,
) -> float:
    """:return: The ESR"""

    print("Plotting a comparison of your model with the target output...")
    net = getattr(model, "net", model)
    num_submodels = getattr(net, "num_submodels", 1)
    with _torch.no_grad():
        print(f"Dataset x size in samples = {len(ds.x)}")
        print(f"Dataset y size in samples = {len(ds.y)}")
        dataset_x_size_in_seconds = len(ds.x) / 48_000
        print(f"Dataset x size in seconds = {dataset_x_size_in_seconds:.2f}")
        t0 = _time()
        output = model(ds.x).cpu().numpy()
        print(f"model output shape = {output.shape}")
        t1 = _time()
        print(f"Took {t1 - t0:.2f} sec ({dataset_x_size_in_seconds / (t1 - t0):.2f}x)")

    if num_submodels > 1 and output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    is_packed = output.ndim == 2 and (num_submodels > 1 or output.shape[0] > 1)
    if is_packed:
        submodel_names = getattr(net, "submodel_names", ())
        labels = [
            submodel_names[i] if i < len(submodel_names) else f"submodel {i}"
            for i in range(output.shape[0])
        ]
        esrs = [_esr(_torch.Tensor(output_i), ds.y) for output_i in output]
        if packed_esrs is not None:
            packed_esrs[:] = [float(submodel_esr) for submodel_esr in esrs]
        aggregate_esr = sum(esrs)
        for label, submodel_esr in zip(labels, esrs):
            print(f"Error-signal ratio ({label}) = {submodel_esr:.4g}")
        print(_format_final_esr(aggregate_esr))

        _plt.figure(figsize=(9.6, 5.4), edgecolor='black')
        for label, output_i, submodel_esr in zip(labels, output, esrs):
            _plt.plot(
                output_i[window_start:window_end],
                linestyle='--',
                label=f"Prediction {label} (ESR={submodel_esr:.4g})",
            )
        _plt.plot(ds.y[window_start:window_end], color='xkcd:blue', linestyle='solid', label='Target')
        _plt.title(
            "Aggregate ESR="
            f"{aggregate_esr:.4g} ("
            + ", ".join(
                f"{label}: {submodel_esr:.4g}" for label, submodel_esr in zip(labels, esrs)
            )
            + ")"
        )
        _plt.xlabel(f"Time in samples from {window_start} to {window_end}")
        _plt.ylabel('Amplitude normalized to range [1, -1]')
        _plt.legend(fancybox=True, shadow=True)
        _plt.grid()
        if filepath is not None:
            _plt.savefig(f"{filepath}_{int(time.time())}.png")
        if not silent:
            _plt.show()
        return aggregate_esr

    output = output.flatten()
    esr = _esr(_torch.Tensor(output), ds.y)
    # Trying my best to put numbers to it...
    if esr < 0.001:
        esr_comment = "HOLY SHIT BRO!!! Niiiiiiice! 🤘😎"
    elif esr < 0.01:
        esr_comment = "Great!"
    elif esr < 0.035:
        esr_comment = "Not bad!"
    elif esr < 0.1:
        esr_comment = "...This *might* sound ok!"
    elif esr < 0.3:
        esr_comment = "...This probably won't sound great :("
    else:
        esr_comment = "...Something seems to have gone wrong."
    print(_format_final_esr(esr))
    print(esr_comment)

    # figsize in inches * dpi default of 100
    #   (16, 5)      = 1600 x 500 pixels
    #   (19.2, 10.8) = 1920 x 1080 pixels
    #   (9.6, 5.4)   = 960 x 540 pixels
    _plt.figure(figsize=(9.6, 5.4), edgecolor='black')
    _plt.plot(output[window_start:window_end], color='xkcd:orange', linestyle='--', label='Prediction')
    _plt.plot(ds.y[window_start:window_end], color='xkcd:blue', linestyle='solid', label='Target')
    _plt.title(f"ESR={esr:.4g}")
    _plt.xlabel(f"Time in samples from {window_start} to {window_end}")
    _plt.ylabel('Amplitude normalized to range [1, -1]')
    _plt.legend(fancybox=True, shadow=True)
    _plt.grid()
    if filepath is not None:
        # E.g.:
        #   ./RLROP_ESRPREEMPH/2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono_1773770799.png
        # Where ./RLROP_ESRPREEMPH is the trainer default output path
        # and 2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono is the model name
        # and 2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono_1773770799 is the checkpoint basename
        # defined in:
        #   def _save_checkpoint(self, trainer: _pl.Trainer, filepath: str):
        _plt.savefig(f"{filepath}_{int(time.time())}.png")
    if not silent:
        _plt.show()
    return esr


def _print_nasty_checks_warning():
    print(
        "\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        "X                                                                          X\n"
        "X                                WARNING:                                  X\n"
        "X                                                                          X\n"
        "X       You are ignoring the checks! Your model might turn out bad!        X\n"
        "X                                                                          X\n"
        "X                              I warned you!                               X\n"
        "X                                                                          X\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )


def _nasty_checks_modal():
    msg = "You are ignoring the checks!\nYour model might turn out bad!"

    root = _tk.Tk()
    root.withdraw()  # hide the root window
    modal = _tk.Toplevel(root)
    modal.geometry("300x100")
    modal.title("Warning!")
    label = _tk.Label(modal, text=msg)
    label.pack(pady=10)
    ok_button = _tk.Button(
        modal,
        text="I can only blame myself!",
        command=lambda: [modal.destroy(), root.quit()],
    )
    ok_button.pack()
    modal.grab_set()  # disable interaction with root window while modal is open
    modal.mainloop()


class _ValidationStopping(_pl.callbacks.EarlyStopping):
    """
    Callback to indicate to stop training if the validation metric is good enough,
    without the other conditions that EarlyStopping usually forces like patience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = _np.inf


class _UserAbortRequested(RuntimeError):
    """Raised when the interactive console controller requests an abort."""


class _UserSkipStageRequested(RuntimeError):
    """Raised when the interactive console controller requests a stage skip."""


class _InteractiveTrainingState(object):
    def __init__(self):
        self._lock = _threading.Lock()
        self._paused = False
        self._abort_requested = False
        self._skip_stage_requested = False
        self._shutdown_requested = False

    @property
    def abort_requested(self) -> bool:
        with self._lock:
            return self._abort_requested

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def skip_stage_requested(self) -> bool:
        with self._lock:
            return self._skip_stage_requested

    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    def request_abort(self):
        with self._lock:
            self._abort_requested = True
            self._paused = False

    def request_skip_stage(self):
        with self._lock:
            if self._abort_requested or self._shutdown_requested:
                return False
            self._skip_stage_requested = True
            self._paused = False
            return True

    def request_shutdown(self):
        with self._lock:
            self._shutdown_requested = True
            self._paused = False

    def consume_skip_stage_request(self) -> bool:
        with self._lock:
            requested = self._skip_stage_requested
            self._skip_stage_requested = False
            return requested

    def toggle_pause(self) -> _Optional[bool]:
        with self._lock:
            if self._abort_requested or self._shutdown_requested:
                return None
            self._paused = not self._paused
            return self._paused


class _InteractiveConsoleController(object):
    """
    Windows console hotkeys for local trainer sessions.

    Ctrl+P toggles pause/resume.
    Ctrl+A requests a graceful abort of the current training run.
    Ctrl+F skips the current training stage.
    """

    _CTRL_A = "\x01"
    _CTRL_F = "\x06"
    _CTRL_P = "\x10"

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.state = _InteractiveTrainingState() if enabled else None
        self._thread: _Optional[_threading.Thread] = None

    @classmethod
    def create(cls, enabled: bool):
        has_console = bool(
            enabled
            and _msvcrt is not None
            and getattr(_sys.stdin, "isatty", lambda: False)()
        )
        return cls(enabled=has_console)

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        print(
            "Interactive trainer hotkeys active in this console: "
            "Ctrl+P = pause/resume, Ctrl+F = skip current stage, "
            "Ctrl+A = abort current run."
        )
        self._thread = _threading.Thread(
            target=self._run,
            name="nam-interactive-console-controller",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        if not self.enabled or self.state is None:
            return
        self.state.request_shutdown()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        assert self.state is not None
        while not self.state.shutdown_requested:
            try:
                if not _msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                ch = _msvcrt.getwch()
                if ch in ("\x00", "\xe0") and _msvcrt.kbhit():
                    _msvcrt.getwch()
                    continue
            except OSError:
                break

            if ch == self._CTRL_P:
                paused = self.state.toggle_pause()
                if paused is True:
                    print(
                        "[Trainer] Pause requested. "
                        "Press Ctrl+P again to resume, Ctrl+F to skip this stage, "
                        "or Ctrl+A to abort."
                    )
                elif paused is False:
                    print("[Trainer] Resume requested.")
            elif ch == self._CTRL_F:
                if self.state.request_skip_stage():
                    print(
                        "[Trainer] Skip requested. "
                        "The trainer will finish this stage using the best checkpoint "
                        "found so far, then move to the next stage or finish the run."
                    )
            elif ch == self._CTRL_A and not self.state.abort_requested:
                self.state.request_abort()
                print(
                    "[Trainer] Abort requested. "
                    "The trainer will stop after the current batch finishes."
                )


class _InteractiveControlCallback(_pl.Callback):
    def __init__(self, controller: _InteractiveConsoleController):
        super().__init__()
        self._controller = controller

    def _check_for_user_controls(self):
        state = self._controller.state
        if state is None:
            return
        while state.paused:
            if state.abort_requested:
                raise _UserAbortRequested("Abort requested while paused.")
            if state.consume_skip_stage_request():
                raise _UserSkipStageRequested("Stage skip requested while paused.")
            time.sleep(0.1)
        if state.abort_requested:
            raise _UserAbortRequested("Abort requested by user.")
        if state.consume_skip_stage_request():
            raise _UserSkipStageRequested("Stage skip requested by user.")

    def on_train_epoch_start(self, trainer, pl_module):
        self._check_for_user_controls()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._check_for_user_controls()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._check_for_user_controls()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check_for_user_controls()


class _EpochTimeLogger(_pl.Callback):
    def __init__(self):
        self._epoch_start_time: _Optional[float] = None
        self._epoch_durations: _Sequence[float] = []

    @property
    def mean_tpe_seconds(self) -> _Optional[float]:
        if len(self._epoch_durations) == 0:
            return None
        return sum(self._epoch_durations) / len(self._epoch_durations)

    @property
    def last_tpe_seconds(self) -> _Optional[float]:
        if len(self._epoch_durations) == 0:
            return None
        return self._epoch_durations[-1]

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = _time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start_time is None:
            return
        elapsed_seconds = _time() - self._epoch_start_time
        self._epoch_durations = [*self._epoch_durations, elapsed_seconds]
        self._epoch_start_time = None
        if trainer.logger is not None:
            trainer.logger.log_metrics({"TPE": elapsed_seconds}, step=trainer.current_epoch)


class _ModelCheckpoint(_pl.callbacks.model_checkpoint.ModelCheckpoint):
    """
    Extension to model checkpoint to save a .nam file as well as the .ckpt file.
    """

    def __init__(
        self,
        *args,
        user_metadata: _Optional[_UserMetadata] = None,
        settings_metadata: _Optional[_metadata.Settings] = None,
        data_metadata: _Optional[_metadata.Data] = None,
        state_key_suffix: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._user_metadata = user_metadata
        self._settings_metadata = settings_metadata
        self._data_metadata = data_metadata
        self._state_key_suffix = state_key_suffix

    _NAM_FILE_EXTENSION = _Exportable.FILE_EXTENSION

    @classmethod
    def _get_nam_filepath(cls, filepath: str) -> _Path:
        """
        Given a .ckpt filepath, figure out a .nam for it.
        """
        if not filepath.endswith(cls.FILE_EXTENSION):
            raise ValueError(
                f"Checkpoint filepath {filepath} doesn't end in expected extension "
                f"{cls.FILE_EXTENSION}"
            )
        return _Path(filepath[: -len(cls.FILE_EXTENSION)] + cls._NAM_FILE_EXTENSION)

    @property
    def _include_other_metadata(self) -> bool:
        return self._settings_metadata is not None and self._data_metadata is not None

    @property
    def state_key(self) -> str:
        base_key = super().state_key
        return (
            base_key
            if self._state_key_suffix == ""
            else f"{base_key}|{self._state_key_suffix}"
        )

    def _save_checkpoint(self, trainer: _pl.Trainer, filepath: str):
        # Save the .ckpt:
        super()._save_checkpoint(trainer, filepath)
        # Save the .nam:
        nam_filepath = self._get_nam_filepath(filepath)
        pl_model: _LightningModule = trainer.model
        nam_model = pl_model.net
        outdir = nam_filepath.parent
        # HACK: Assume the extension
        basename = f"{nam_filepath.name[: -len(self._NAM_FILE_EXTENSION)]}"
        #print(f"trainer.callback_metrics = {trainer.callback_metrics}")
        #print(f"trainer.logged_metrics = {trainer.logged_metrics}")
        logged_metrics = {key: value.item() for key, value in trainer.logged_metrics.items()}
        other_metadata = (
            None
            if not self._include_other_metadata
            else {
                _metadata.TRAINING_KEY: _metadata.TrainingMetadata(
                    settings=self._settings_metadata,
                    data=self._data_metadata,
                    validation_esr=logged_metrics.get("ESR"),
                    logged_metrics=logged_metrics,
                    #TODO: embed the *final* plotted *full* validation ESR in the checkpoint
                ).model_dump()
            }
        )
        nam_model.export(
            outdir,
            basename=basename,
            user_metadata=self._user_metadata,
            other_metadata=other_metadata,
        )

    def _remove_checkpoint(self, trainer: _pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        nam_path = self._get_nam_filepath(filepath)
        if nam_path.exists():
            nam_path.unlink()


def get_callbacks(
    threshold_esr: _Optional[float],
    dirpath: _Optional[str] = None,
    user_metadata: _Optional[_UserMetadata] = None,
    settings_metadata: _Optional[_metadata.Settings] = None,
    data_metadata: _Optional[_metadata.Data] = None,
    interactive_controller: _Optional[_InteractiveConsoleController] = None,
    show_advanced_init_info: bool = False,
    checkpoint_save_mode: _Union[CheckpointSaveMode, str] = CheckpointSaveMode.MINIMAL,
    packed: bool = False,
):
    checkpoint_save_mode = CheckpointSaveMode(checkpoint_save_mode)
    epoch_time_logger = _EpochTimeLogger()
    keep_four_best_plus_current = (
        checkpoint_save_mode == CheckpointSaveMode.FOUR_BEST_PLUS_CURRENT
    )
    best_checkpoint_filename = (
        "checkpoint_best_{epoch:04d}_{step}_{ESR:.4g}_{ESRPREEMPH:.4g}_{MSE:.4g}"
    )
    last_checkpoint_filename = (
        "checkpoint_last_{epoch:04d}_{step}_{ESR:.4g}_{ESRPREEMPH:.4g}_{MSE:.4g}"
    )
    best_checkpoint_callback = _ModelCheckpoint(
        dirpath=dirpath,
        filename=best_checkpoint_filename,
        save_top_k=4 if keep_four_best_plus_current else 3,
        monitor="val_loss",
        every_n_epochs=1,
        user_metadata=user_metadata,
        settings_metadata=settings_metadata,
        data_metadata=data_metadata,
        state_key_suffix="top4_best" if keep_four_best_plus_current else "top3_best",
    )
    last_checkpoint_callback = _ModelCheckpoint(
        dirpath=dirpath,
        filename=last_checkpoint_filename,
        every_n_epochs=1,
        user_metadata=user_metadata,
        settings_metadata=settings_metadata,
        data_metadata=data_metadata,
        state_key_suffix="last_epoch",
    )
    callbacks = [
        best_checkpoint_callback,
        last_checkpoint_callback,
        # FIXME: Early stopping is too aggressive / stops too early
        #early_stopping,
        time_stats_monitor,
        epoch_time_logger,
    ]
    if _device_stats_monitor_enabled():
        callbacks.insert(2, DeviceStatsMonitor(cpu_stats=True))
    preferred_checkpoint_callback = best_checkpoint_callback
    if checkpoint_save_mode == CheckpointSaveMode.MAXIMUM:
        current_best_checkpoint_callback = _ModelCheckpoint(
            dirpath=dirpath,
            filename="checkpoint_current_best",
            save_top_k=1,
            monitor="val_loss",
            every_n_epochs=1,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
            state_key_suffix="current_best",
        )
        callbacks.insert(1, current_best_checkpoint_callback)
        preferred_checkpoint_callback = current_best_checkpoint_callback
    if show_advanced_init_info:
        callbacks.insert(3, rich_model_summary)
    if interactive_controller is not None and interactive_controller.enabled:
        callbacks.append(_InteractiveControlCallback(interactive_controller))
    if packed:
        callbacks.extend(
            [
                _PackedBestCheckpoint(dirpath=dirpath),
                _PackedMaskCallback(),
            ]
        )
    # Stop training if stopping threshold ESR validation loss is reached
    if threshold_esr is not None:
        callbacks.append(
            _ValidationStopping(monitor="ESR", stopping_threshold=threshold_esr)
        )
    return callbacks, preferred_checkpoint_callback, epoch_time_logger


class TrainOutput(_NamedTuple):
    """
    :param model: The trained model
    :param simplified_trainer_metadata: The metadata summarizing training with the
        simplified trainer.
    """
    model: _Optional[_LightningModule]
    metadata: _metadata.TrainingMetadata
    aborted: bool = False


class _FinalLatencyError(ValueError):
    """Raised when the final latency cannot be determined."""
    pass


def _get_final_latency(latency_analysis: _metadata.Latency) -> int:
    """
    Make a decision based on automatic and manual latency values what will be
    used for training.
    """
    user = latency_analysis.manual
    analyzed = latency_analysis.calibration.recommended

    if user is not None:
        if analyzed is not None:
            if user == analyzed:
                print(f"The user latency is same as the analyzed latency ({user}).")
            else:
                print(
                    f"The user latency is different from the analyzed latency ({user} vs {analyzed})."
                )
                print(f"Override the analyzed latency with the user latency.")
        else:
            print(
                f"Cannot automatically analyze the latency. Use the user latency ({user})."
            )

        return user

    if analyzed is not None:
        print(f"Cannot use the user latency. Use the analyzed latency ({analyzed}).")
        return analyzed

    raise _FinalLatencyError(
        "No latency provided and cannot automatically analyze the latency."
    )


def _apply_packed_best_checkpoints(
    model: _PackedLightningModule,
    model_config: _Dict,
    checkpoint_paths: _Sequence[_Optional[str]],
) -> None:
    if not checkpoint_paths or not any(checkpoint_paths):
        return
    for submodel_index, checkpoint_path in enumerate(checkpoint_paths):
        if checkpoint_path is None:
            continue
        if submodel_index >= model.net.num_submodels:
            print(
                "WARNING: Ignoring packed-best checkpoint for unknown submodel "
                f"{submodel_index}: {checkpoint_path}"
            )
            continue
        try:
            source = _PackedLightningModule.load_from_checkpoint(
                checkpoint_path, **_PackedLightningModule.parse_config(model_config)
            )
            source.cpu()
            source.eval()
            if source.net.sample_rate is None:
                source.net.sample_rate = model.net.sample_rate
            model.net.import_submodel(
                submodel_index, source.net.extract_submodel(submodel_index)
            )
        except Exception as e:
            print(
                "WARNING: Failed to load packed-best checkpoint for submodel "
                f"{submodel_index} from {checkpoint_path}: {e}"
            )
    model.net.apply_mask()


class _StageRunOutput(_NamedTuple):
    model: _LightningModule
    train_dataloader: _DataLoader
    val_dataloader: _DataLoader
    best_checkpoint: str
    plot_basename: str
    logged_metrics: _Dict[str, float]
    run_dir: str
    packed_best_checkpoints: _Tuple[_Optional[str], ...] = ()
    skipped: bool = False


def _get_complete_a2_packed_best_checkpoint_pair(
    checkpoint_paths: _Optional[_Sequence[_Optional[str]]],
) -> _Optional[_Tuple[str, str]]:
    if not checkpoint_paths:
        return None
    paths = tuple(
        None if path in (None, "") else str(path) for path in checkpoint_paths
    )
    if len(paths) != 2 or any(path is None for path in paths):
        return None
    missing = [path for path in paths if not _Path(path).is_file()]
    if missing:
        print(
            "WARNING: A2 packed-best handoff checkpoint is missing; "
            "falling back to the aggregate best checkpoint for refinement: "
            + ", ".join(str(path) for path in missing)
        )
        return None
    return paths


def _teardown_dataloaders(*dataloaders):
    for dl in dataloaders:
        assert isinstance(dl.dataset, _AbstractDataset)
        dl.dataset.teardown()


def _write_analysis_info(run_dir: _Path, info: _Dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "analysis_info.json", "w", encoding="utf-8") as fp:
        _json.dump(info, fp, indent=2)


def _scheduler_name_from_model_config(model_config: _Dict) -> str:
    scheduler_config = model_config.get("lr_scheduler") or {}
    scheduler_name = scheduler_config.get("name") or scheduler_config.get("class", "")
    return "" if scheduler_name is None else str(scheduler_name)


def _window_kwargs(version: _Version):
    if version.major == 1:
        return dict(window_start=100_000, window_end=101_000)
    elif version.major == 2:
        return dict(window_start=100_000, window_end=101_000)
    elif version.major == 5:
        return dict(window_start=96_000, window_end=120_000)
    return dict(window_start=100_000, window_end=101_000)


def _run_training_stage(
    *,
    stage_name: str,
    input_path: str,
    output_path: str,
    data_config: _Dict,
    model_config: _Dict,
    learning_config: _Dict,
    train_path: str,
    modelname: str,
    architecture: Architecture,
    precision: str,
    ny: int,
    batch_size: int,
    stage_epochs: int,
    stage_lr: float,
    stage_lr_decay: float,
    threshold_esr: _Optional[float],
    user_metadata: _Optional[_UserMetadata],
    settings_metadata: _metadata.Settings,
    data_metadata: _metadata.Data,
    interactive_controller: _InteractiveConsoleController,
    fast_dev_run: _Union[bool, int],
    show_advanced_init_info: bool,
    checkpoint_save_mode: _Union[CheckpointSaveMode, str],
):
    assert (
        "fast_dev_run" not in learning_config
    ), "fast_dev_run is set as a kwarg to train()"

    print(f"Starting {stage_name.replace('_', ' ')} training...")
    lightning_cls = _get_lightning_module_cls(model_config)
    is_packed = model_config["net"]["name"] == "PackedWaveNet"
    model = lightning_cls.init_from_config(model_config)
    source_checkpoint_paths_by_submodel = model_config.get("checkpoint_paths_by_submodel")
    if is_packed and source_checkpoint_paths_by_submodel:
        print("Initializing A2 packed submodels from checkpoint pair:")
        for submodel_index, source_checkpoint_path in enumerate(source_checkpoint_paths_by_submodel):
            print(f"  submodel {submodel_index}: {source_checkpoint_path}")
        _apply_packed_best_checkpoints(
            model, model_config, source_checkpoint_paths_by_submodel
        )
    train_dataloader, val_dataloader = _get_dataloaders(
        data_config, learning_config, model
    )
    run_dir: _Optional[_Path] = None
    analysis_info: _Optional[_Dict] = None
    try:
        if train_dataloader.dataset.sample_rate != val_dataloader.dataset.sample_rate:
            raise RuntimeError(
                "Train and validation data loaders have different data set sample "
                "rates: "
                f"{train_dataloader.dataset.sample_rate}, "
                f"{val_dataloader.dataset.sample_rate}"
            )
        sample_rate = train_dataloader.dataset.sample_rate
        model.net.sample_rate = sample_rate

        save_dir = _Path(train_path).joinpath(modelname, stage_name)
        str_epochs = str(stage_epochs)
        str_hparams = (
            f"{precision}_ny{str(ny)}_batch{str(batch_size)}_lr{str(stage_lr)}"
            f"_lrdecay{str(stage_lr_decay)}"
        )
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name=architecture.value,
            version=str_epochs,
            sub_dir=str_hparams,
        )
        run_dir = _Path(logger.log_dir)
        scheduler_name = _scheduler_name_from_model_config(model_config)
        analysis_info = {
            "model_name": modelname,
            "input_file": _Path(input_path).name,
            "output_file": _Path(output_path).name,
            "stage": stage_name,
            "architecture": architecture.value,
            "epochs": stage_epochs,
            "learning_rate": stage_lr,
            "lr_decay": stage_lr_decay,
            "scheduler": scheduler_name,
            "checkpoint_save_mode": CheckpointSaveMode(checkpoint_save_mode).value,
            "status": "initialized",
        }
        if model_config.get("checkpoint_path"):
            analysis_info["source_checkpoint"] = model_config["checkpoint_path"]
        if model_config.get("checkpoint_paths_by_submodel"):
            analysis_info["source_checkpoints_by_submodel"] = model_config[
                "checkpoint_paths_by_submodel"
            ]
        _write_analysis_info(run_dir, analysis_info)
        dirpath = _Path(logger.log_dir).joinpath("checkpoints")
        callbacks, preferred_checkpoint_callback, epoch_time_logger = get_callbacks(
            threshold_esr,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
            dirpath=dirpath,
            interactive_controller=interactive_controller,
            show_advanced_init_info=show_advanced_init_info,
            checkpoint_save_mode=checkpoint_save_mode,
            packed=is_packed,
        )
        packed_best_callback = next(
            (
                callback
                for callback in callbacks
                if isinstance(callback, _PackedBestCheckpoint)
            ),
            None,
        )
        def make_trainer():
            return _pl.Trainer(
                callbacks=callbacks,
                default_root_dir=train_path,
                fast_dev_run=fast_dev_run,
                logger=logger,
                precision=precision,
                benchmark=True,
                enable_model_summary=show_advanced_init_info,
                **learning_config["trainer"],
            )

        if show_advanced_init_info:
            with _filter_warnings(
                "ignore",
                category=UserWarning,
                message=r"Please use the new API settings to control TF32 behavior.*",
            ):
                trainer = make_trainer()
        else:
            with _filter_warnings(
                "ignore",
                message=r"Checkpoint directory .* exists and is not empty\.",
            ):
                with _filter_warnings(
                    "ignore",
                    category=UserWarning,
                    message=r"Please use the new API settings to control TF32 behavior.*",
                ):
                    with _temporary_logging_levels(
                        [
                            "pytorch_lightning.utilities.rank_zero",
                            "pytorch_lightning",
                        ],
                        level=30,
                    ):
                        trainer = make_trainer()

        fit_start_time = _time()
        fit_elapsed_seconds: _Optional[float] = None
        stage_skipped = False
        try:
            with _filter_warnings("ignore", category=_PossibleUserWarning):
                with _filter_warnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=r".*Tensor\.pin_memory\(\) is deprecated.*",
                ):
                    with _filter_warnings(
                        "ignore",
                        category=DeprecationWarning,
                        message=r".*Tensor\.is_pinned\(\) is deprecated.*",
                    ):
                        with _filter_warnings(
                            "ignore",
                            category=UserWarning,
                            message=r"Please use the new API settings to control TF32 behavior.*",
                        ):
                            if show_advanced_init_info:
                                trainer.fit(model, train_dataloader, val_dataloader)
                            else:
                                with _temporary_logging_levels(
                                    [
                                        "pytorch_lightning.utilities.rank_zero",
                                        "pytorch_lightning",
                                    ],
                                    level=30,
                                ):
                                    trainer.fit(model, train_dataloader, val_dataloader)
            fit_elapsed_seconds = _time() - fit_start_time
        except _UserSkipStageRequested:
            fit_elapsed_seconds = _time() - fit_start_time
            stage_skipped = True
            print(
                f"\n{stage_name.replace('_', ' ').title()} skip requested. "
                "Proceeding with the best checkpoint found so far."
            )
        except _UserAbortRequested:
            fit_elapsed_seconds = _time() - fit_start_time
            analysis_info.update(
                {
                    "status": "aborted",
                    "training_time_seconds": fit_elapsed_seconds,
                }
            )
            _write_analysis_info(run_dir, analysis_info)
            raise
        except KeyboardInterrupt:
            fit_elapsed_seconds = _time() - fit_start_time
            print("\nTraining interrupted by user.")

        best_checkpoint = preferred_checkpoint_callback.best_model_path
        if best_checkpoint == "":
            last_checkpoint_callbacks = [
                callback
                for callback in callbacks
                if isinstance(callback, _ModelCheckpoint)
                and getattr(callback, "_state_key_suffix", "") == "last_epoch"
            ]
            if len(last_checkpoint_callbacks) > 0:
                best_checkpoint = last_checkpoint_callbacks[0].last_model_path
        if best_checkpoint == "":
            forced_checkpoint = dirpath / f"{stage_name}_skip_fallback.ckpt"
            trainer.save_checkpoint(str(forced_checkpoint))
            best_checkpoint = str(forced_checkpoint)
        if best_checkpoint != "":
            model = lightning_cls.load_from_checkpoint(
                best_checkpoint,
                **lightning_cls.parse_config(model_config),
            )
        model.cpu()
        model.eval()
        model.net.sample_rate = sample_rate
        packed_best_checkpoints: _Tuple[_Optional[str], ...] = ()
        if (
            is_packed
            and isinstance(model, _PackedLightningModule)
            and packed_best_callback is not None
        ):
            packed_best_checkpoints = tuple(packed_best_callback.checkpoint_paths)
            _apply_packed_best_checkpoints(
                model, model_config, packed_best_checkpoints
            )

        logged_metrics = {
            key: value.item() for key, value in trainer.logged_metrics.items()
        }
        if epoch_time_logger.last_tpe_seconds is not None:
            logged_metrics["TPE"] = epoch_time_logger.last_tpe_seconds
        analysis_info.update(
            {
                "status": "skipped" if stage_skipped else "completed",
                "best_checkpoint": best_checkpoint,
                "training_time_seconds": fit_elapsed_seconds,
                "training_time_per_epoch_seconds": epoch_time_logger.mean_tpe_seconds,
                "logged_metrics": logged_metrics,
            }
        )
        if packed_best_checkpoints:
            analysis_info["packed_best_checkpoints_by_submodel"] = list(
                packed_best_checkpoints
            )
        _write_analysis_info(run_dir, analysis_info)
        return _StageRunOutput(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            best_checkpoint=best_checkpoint,
            plot_basename=(
                f"{modelname}_{stage_name}_{architecture.value}_{str_epochs}_{str_hparams}"
            ),
            logged_metrics=logged_metrics,
            run_dir=str(run_dir),
            packed_best_checkpoints=packed_best_checkpoints,
            skipped=stage_skipped,
        )
    except Exception:
        if analysis_info is not None and run_dir is not None:
            analysis_info["status"] = "failed"
            _write_analysis_info(run_dir, analysis_info)
        _teardown_dataloaders(train_dataloader, val_dataloader)
        raise


def train(
    input_path: str,
    output_path: str,
    train_path: str,
    epochs=300,
    latency: _Optional[int] = None,
    model_type: str = "WaveNet",
    architecture: _Union[Architecture, str] = Architecture.STANDARD,
    precision: str = '32-true',
    batch_size: int = 16,
    ny: int = _NY_DEFAULT,
    lr=0.002,
    lr_decay=0.004,
    lr_scheduler_type: _Union[LearningRateScheduler, str] = LearningRateScheduler.EXPONENTIAL,
    seed: _Optional[int] = 0,
    save_plot: bool = True,
    silent: bool = False,
    modelname: str = "model",
    ignore_checks: bool = False,
    local: bool = False,
    fit_mrstft: bool = True,
    threshold_esr: _Optional[bool] = None,
    user_metadata: _Optional[_UserMetadata] = None,
    fast_dev_run: _Union[bool, int] = False,
    stage_mode: _Union[TrainingStageMode, str] = TrainingStageMode.SINGLE_STAGE,
    stage2_epochs: int = 0,
    stage2_lr: _Optional[float] = None,
    stage2_lr_decay: _Optional[float] = None,
    stage2_lr_scheduler_type: _Optional[_Union[LearningRateScheduler, str]] = None,
    stage2_focus: _Optional[_Union[StageTwoFocus, str]] = None,
    checkpoint_save_mode: _Union[CheckpointSaveMode, str] = CheckpointSaveMode.MINIMAL,
    checkpoint_path: _Optional[_Union[str, _Sequence[_Optional[str]]]] = None,
    show_ignore_checks_warning: bool = True,
    show_advanced_init_info: bool = False,
) -> _Optional[TrainOutput]:
    """
    :param input_path: Full path to input file
    :param output_path: Full path to output file
    :param lr_decay: ReduceLROnPlateau factor for WaveNet training.
    :param threshold_esr: Stop training if ESR is better than this. Ignore if `None`.
    :param checkpoint_path: Optional .ckpt file, or an A2 two-checkpoint sequence, used to initialize additional training.
    :param fast_dev_run: One-step training, used for tests.
    """

    if seed is not None:
        _torch.manual_seed(seed)

    architecture = Architecture(architecture)
    if _is_a2_architecture(architecture):
        model_type = "PackedWaveNet"
    elif _is_causal_conv_lstm_architecture(architecture):
        model_type = "CausalConvLSTM"
    elif model_type == "WaveNet" and _is_lstm_only_architecture(architecture):
        model_type = "LSTM"
    stage_mode = TrainingStageMode(stage_mode)
    lr_scheduler_type = LearningRateScheduler(lr_scheduler_type)
    checkpoint_save_mode = CheckpointSaveMode(checkpoint_save_mode)
    stage2_lr = lr if stage2_lr is None else stage2_lr
    stage2_lr_decay = lr_decay if stage2_lr_decay is None else stage2_lr_decay
    stage2_lr_scheduler_type = (
        LearningRateScheduler.REDUCE_ON_PLATEAU
        if stage2_lr_scheduler_type is None
        else LearningRateScheduler(stage2_lr_scheduler_type)
    )
    stage2_focus = None if stage2_focus in (None, "") else StageTwoFocus(stage2_focus)
    checkpoint_paths_by_submodel = None

    def validate_checkpoint_file(checkpoint_file: _Path):
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_file}")
        if checkpoint_file.suffix.lower() != ".ckpt":
            raise ValueError(f"Checkpoint file must end in .ckpt: {checkpoint_file}")

    if isinstance(checkpoint_path, (list, tuple)):
        if not _is_a2_architecture(architecture):
            raise ValueError("Checkpoint pairs are only supported for A2 architectures.")
        checkpoint_paths_by_submodel = tuple(
            None if path in (None, "") else str(_Path(path)) for path in checkpoint_path
        )
        if (
            len(checkpoint_paths_by_submodel) != 2
            or any(path is None for path in checkpoint_paths_by_submodel)
        ):
            raise ValueError("A2 resume requires exactly two checkpoint files.")
        for checkpoint_file in checkpoint_paths_by_submodel:
            validate_checkpoint_file(_Path(checkpoint_file))
        checkpoint_path = None
    else:
        checkpoint_path = None if checkpoint_path in (None, "") else str(_Path(checkpoint_path))
        if checkpoint_path is not None:
            validate_checkpoint_file(_Path(checkpoint_path))

    sample_rate_validation = _check_audio_sample_rates(input_path, output_path)
    if not sample_rate_validation.passed:
        raise ValueError(
            "Different sample rates detected for input "
            f"({sample_rate_validation.input}) and output "
            f"({sample_rate_validation.output}) audio!"
        )
    length_validation = _check_audio_lengths(input_path, output_path)
    if not length_validation.passed:
        raise ValueError(
            "Your recording differs in length from the input file by "
            f"{length_validation.delta_seconds:.2f} seconds. Check your reamp "
            "in your DAW and ensure that they are the same length."
        )

    input_version, strong_match = _detect_input_version(
        input_path, verbose=show_advanced_init_info
    )

    user_latency = latency
    latency_analysis = _analyze_latency(
        user_latency,
        input_version,
        input_path,
        output_path,
        silent=silent,
        verbose=show_advanced_init_info,
    )
    final_latency = _get_final_latency(latency_analysis)

    data_check_output = _check_data(
        input_path,
        output_path,
        input_version,
        final_latency,
        silent,
        show_replicate_info=True,
    )

    if data_check_output is not None:
        if data_check_output.passed:
            print("-Checks passed")
        else:
            print("Failed checks!")
            if ignore_checks:
                if show_ignore_checks_warning:
                    if local and not silent:
                        _nasty_checks_modal()
                    else:
                        _print_nasty_checks_warning()
            elif not local:
                print(
                    "(To disable this check, run AT YOUR OWN RISK with "
                    "`ignore_checks=True`.)"
                )
            if not ignore_checks:
                print("Exiting core training...")
                return TrainOutput(
                    model=None,
                    metadata=_metadata.TrainingMetadata(
                        settings=_metadata.Settings(ignore_checks=ignore_checks),
                        data=_metadata.Data(
                            latency=latency_analysis, checks=data_check_output
                        ),
                        validation_esr=None,
                    ),
                )

    settings_metadata = _metadata.Settings(ignore_checks=ignore_checks)
    data_metadata = _metadata.Data(latency=latency_analysis, checks=data_check_output)
    interactive_controller = _InteractiveConsoleController.create(enabled=local)

    final_stage_output: _Optional[_StageRunOutput] = None
    training_aborted = False
    try:
        interactive_controller.start()

        if stage_mode == TrainingStageMode.REFINEMENT_ONLY:
            if checkpoint_path is None and checkpoint_paths_by_submodel is None:
                raise ValueError("Refinement Only requires a source .ckpt checkpoint.")
            if stage2_epochs <= 0:
                raise ValueError("Refinement Only requires refinement epochs greater than 0.")
            if stage2_focus is None:
                stage2_focus = StageTwoFocus.LOW
            print(
                "Starting refinement-only training from checkpoint "
                f"with {stage2_focus.value} focus: "
                f"{checkpoint_path or checkpoint_paths_by_submodel}"
            )
            data_config, model_config, learning_config = _get_configs(
                input_version,
                input_path,
                output_path,
                final_latency,
                stage2_epochs,
                model_type,
                architecture,
                ny,
                stage2_lr,
                stage2_lr_decay,
                batch_size,
                fit_mrstft,
                stage2_lr_scheduler_type,
                checkpoint_path=checkpoint_path,
                checkpoint_paths_by_submodel=checkpoint_paths_by_submodel,
                stage_two_focus=stage2_focus,
            )
            final_stage_output = _run_training_stage(
                stage_name=f"stage_2_{stage2_focus.value.lower()}",
                input_path=input_path,
                output_path=output_path,
                data_config=data_config,
                model_config=model_config,
                learning_config=learning_config,
                train_path=train_path,
                modelname=modelname,
                architecture=architecture,
                precision=precision,
                ny=ny,
                batch_size=batch_size,
                stage_epochs=stage2_epochs,
                stage_lr=stage2_lr,
                stage_lr_decay=stage2_lr_decay,
                threshold_esr=threshold_esr,
                user_metadata=user_metadata,
                settings_metadata=settings_metadata,
                data_metadata=data_metadata,
                interactive_controller=interactive_controller,
                fast_dev_run=fast_dev_run,
                show_advanced_init_info=show_advanced_init_info,
                checkpoint_save_mode=checkpoint_save_mode,
            )
        else:
            data_config, model_config, learning_config = _get_configs(
                input_version,
                input_path,
                output_path,
                final_latency,
                epochs,
                model_type,
                architecture,
                ny,
                lr,
                lr_decay,
                batch_size,
                fit_mrstft,
                lr_scheduler_type,
                checkpoint_path=checkpoint_path,
                checkpoint_paths_by_submodel=checkpoint_paths_by_submodel,
            )
            if checkpoint_path is not None:
                print(f"Starting stage 1 from checkpoint: {checkpoint_path}")
            elif checkpoint_paths_by_submodel is not None:
                print(f"Starting stage 1 from A2 checkpoint pair: {checkpoint_paths_by_submodel}")
            final_stage_output = _run_training_stage(
                stage_name="stage_1",
                input_path=input_path,
                output_path=output_path,
                data_config=data_config,
                model_config=model_config,
                learning_config=learning_config,
                train_path=train_path,
                modelname=modelname,
                architecture=architecture,
                precision=precision,
                ny=ny,
                batch_size=batch_size,
                stage_epochs=epochs,
                stage_lr=lr,
                stage_lr_decay=lr_decay,
                threshold_esr=threshold_esr,
                user_metadata=user_metadata,
                settings_metadata=settings_metadata,
                data_metadata=data_metadata,
                interactive_controller=interactive_controller,
                fast_dev_run=fast_dev_run,
                show_advanced_init_info=show_advanced_init_info,
                checkpoint_save_mode=checkpoint_save_mode,
            )

            do_stage_two = (
                stage_mode == TrainingStageMode.TWO_STAGE
                and stage2_epochs > 0
                and stage2_focus is not None
            )
            if do_stage_two:
                if final_stage_output.best_checkpoint == "":
                    print(
                        "Stage 1 did not produce a best checkpoint. "
                        "Skipping stage 2 fine-tuning."
                    )
                else:
                    stage1_best_checkpoint = final_stage_output.best_checkpoint
                    stage1_packed_best_checkpoint_pair = (
                        _get_complete_a2_packed_best_checkpoint_pair(
                            final_stage_output.packed_best_checkpoints
                        )
                        if _is_a2_architecture(architecture)
                        else None
                    )
                    _teardown_dataloaders(
                        final_stage_output.train_dataloader,
                        final_stage_output.val_dataloader,
                    )
                    final_stage_output = None
                    if stage1_packed_best_checkpoint_pair is not None:
                        print(
                            "Starting stage 2 fine-tuning from the best stage 1 A2 "
                            "submodel checkpoints with "
                            f"{stage2_focus.value} focus:"
                        )
                        for submodel_index, checkpoint_path in enumerate(
                            stage1_packed_best_checkpoint_pair
                        ):
                            print(f"  submodel {submodel_index}: {checkpoint_path}")
                    else:
                        print(
                            "Starting stage 2 fine-tuning from the best stage 1 checkpoint "
                            f"with {stage2_focus.value} focus."
                        )
                    stage2_data_config, stage2_model_config, stage2_learning_config = (
                        _get_configs(
                            input_version,
                            input_path,
                            output_path,
                            final_latency,
                            stage2_epochs,
                            model_type,
                            architecture,
                            ny,
                            stage2_lr,
                            stage2_lr_decay,
                            batch_size,
                            fit_mrstft,
                            stage2_lr_scheduler_type,
                            checkpoint_path=(
                                None
                                if stage1_packed_best_checkpoint_pair is not None
                                else stage1_best_checkpoint
                            ),
                            checkpoint_paths_by_submodel=(
                                stage1_packed_best_checkpoint_pair
                            ),
                            stage_two_focus=stage2_focus,
                        )
                    )
                    final_stage_output = _run_training_stage(
                        stage_name=f"stage_2_{stage2_focus.value.lower()}",
                        input_path=input_path,
                        output_path=output_path,
                        data_config=stage2_data_config,
                        model_config=stage2_model_config,
                        learning_config=stage2_learning_config,
                        train_path=train_path,
                        modelname=modelname,
                        architecture=architecture,
                        precision=precision,
                        ny=ny,
                        batch_size=batch_size,
                        stage_epochs=stage2_epochs,
                        stage_lr=stage2_lr,
                        stage_lr_decay=stage2_lr_decay,
                        threshold_esr=threshold_esr,
                        user_metadata=user_metadata,
                        settings_metadata=settings_metadata,
                        data_metadata=data_metadata,
                        interactive_controller=interactive_controller,
                        fast_dev_run=fast_dev_run,
                        show_advanced_init_info=show_advanced_init_info,
                        checkpoint_save_mode=checkpoint_save_mode,
                    )
    except _UserAbortRequested:
        training_aborted = True
        print("\nTraining aborted by user. Returning to the trainer UI.")
    finally:
        interactive_controller.stop()

    if training_aborted:
        if final_stage_output is not None:
            _teardown_dataloaders(
                final_stage_output.train_dataloader, final_stage_output.val_dataloader
            )
        return TrainOutput(
            model=None,
            metadata=_metadata.TrainingMetadata(
                settings=settings_metadata,
                data=data_metadata,
                validation_esr=None,
                logged_metrics={"aborted": 1.0},
            ),
            aborted=True,
        )

    if final_stage_output is None:
        raise RuntimeError("Training did not produce any stage output.")

    packed_validation_esrs = []
    validation_esr = _plot(
        final_stage_output.model,
        final_stage_output.val_dataloader.dataset,
        filepath=(
            train_path + "/" + final_stage_output.plot_basename if save_plot else None
        ),
        silent=silent,
        packed_esrs=packed_validation_esrs,
        **_window_kwargs(input_version),
    )

    _teardown_dataloaders(
        final_stage_output.train_dataloader, final_stage_output.val_dataloader
    )

    logged_metrics = dict(final_stage_output.logged_metrics)
    logged_metrics["final_plotted_validation_esr"] = validation_esr
    for submodel_index, submodel_esr in enumerate(packed_validation_esrs):
        logged_metrics[
            f"final_plotted_validation_esr_packed_{submodel_index}"
        ] = submodel_esr

    return TrainOutput(
        model=final_stage_output.model,
        metadata=_metadata.TrainingMetadata(
            settings=settings_metadata,
            data=data_metadata,
            validation_esr=validation_esr,
            logged_metrics=logged_metrics,
        ),
        aborted=False,
    )


class DataInputValidation(_BaseModel):
    passed: bool


def validate_input(input_path) -> DataInputValidation:
    """:return: Could it be validated?"""
    try:
        _detect_input_version(input_path, verbose=False)
        # succeeded...
        return DataInputValidation(passed=True)
    except _InputValidationError as e:
        print(f"Input validation failed!\n\n{e}")
        return DataInputValidation(passed=False)


class _PyTorchDataSplitValidation(_BaseModel):
    """:param msg: On exception, catch and assign. Otherwise None"""
    passed: bool
    msg: _Optional[str]


class _PyTorchDataValidation(_BaseModel):
    passed: bool
    train: _PyTorchDataSplitValidation  # cf Split.TRAIN
    validation: _PyTorchDataSplitValidation  # Split.VALIDATION


class _SampleRateValidation(_BaseModel):
    passed: bool
    input: int
    output: int


class _LengthValidation(_BaseModel):
    passed: bool
    delta_seconds: float


class DataValidationOutput(_BaseModel):
    passed: bool
    passed_critical: bool
    sample_rate: _SampleRateValidation
    length: _LengthValidation
    input_version: str
    latency: _metadata.Latency
    checks: _metadata.DataChecks
    pytorch: _PyTorchDataValidation


def _check_audio_sample_rates(
    input_path: _Path,
    output_path: _Path,
) -> _SampleRateValidation:
    _, x_info = _wav_to_np(input_path, info=True)
    _, y_info = _wav_to_np(output_path, info=True)

    return _SampleRateValidation(
        passed=x_info.rate == y_info.rate,
        input=x_info.rate,
        output=y_info.rate,
    )


def _check_audio_lengths(
    input_path: _Path,
    output_path: _Path,
    max_under_seconds: _Optional[float] = 0.0,
    max_over_seconds: _Optional[float] = 1.0,
) -> _LengthValidation:
    """
    Check that the input and output have the right lengths compared to each
    other.

    :param input_path: Path to input audio
    :param output_path: Path to output audio
    :param max_under_seconds: If not None, the maximum amount by which the
        output can be shorter than the input. Should be non-negative i.e. a
        value of 1.0 means that the output can't be more than a second shorter
        than the input.
    :param max_over_seconds: If not None, the maximum amount by which the
        output can be longer than the input. Should be non-negative i.e. a
        value of 1.0 means that the output can't be more than a second longer
        than the input.
    """
    x, x_info = _wav_to_np(input_path, info=True)
    y, y_info = _wav_to_np(output_path, info=True)

    length_input = len(x) / x_info.rate
    length_output = len(y) / y_info.rate
    delta_seconds = length_output - length_input

    passed = True
    if max_under_seconds is not None and delta_seconds < -max_under_seconds:
        passed = False
    if max_over_seconds is not None and delta_seconds > max_over_seconds:
        passed = False

    return _LengthValidation(passed=passed, delta_seconds=delta_seconds)


def validate_data(
    input_path: _Path,
    output_path: _Path,
    user_latency: _Optional[int],
    num_output_samples_per_datum: int = _NY_DEFAULT,
):
    """
    Just do the checks to make sure that the data are ok.

    * Version identification
    * Latency calibration
    * Other checks
    """
    print("Validating data...")
    passed = True  # Until proven otherwise
    passed_critical = True  # These can't be ignored

    sample_rate_validation = _check_audio_sample_rates(input_path, output_path)
    passed = passed and sample_rate_validation.passed
    passed_critical = passed_critical and sample_rate_validation.passed

    length_validation = _check_audio_lengths(input_path, output_path)
    passed = passed and length_validation.passed
    passed_critical = passed_critical and length_validation.passed

    # Data version ID
    input_version, strong_match = _detect_input_version(input_path, verbose=False)

    # Latency analysis
    latency_analysis = _analyze_latency(
        user_latency,
        input_version,
        input_path,
        output_path,
        silent=True,
        verbose=False,
    )
    if latency_analysis.manual is None and any(
        val for val in latency_analysis.calibration.warnings.model_dump().values()
    ):
        passed = False
    final_latency = _get_final_latency(latency_analysis)

    # Other data checks based on input file version
    data_checks = _check_data(
        input_path,
        output_path,
        input_version,
        latency_analysis.calibration.recommended,
        silent=True,
        show_replicate_info=False,
    )
    passed = passed and data_checks.passed

    # Finally, try to make the PyTorch Dataset objects and note any failures:
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=num_output_samples_per_datum,
        latency=final_latency,
    )
    # HACK this should depend on the model that's going to be used, but I think it will
    # be unlikely to make a difference. Still, would be nice to fix.
    data_config["common"]["nx"] = 4096

    pytorch_data_split_validation_dict: _Dict[str, _PyTorchDataSplitValidation] = {}
    for split in _Split:
        try:
            ds = _init_dataset(data_config, split)
            ds.teardown()
            pytorch_data_split_validation_dict[split.value] = (
                _PyTorchDataSplitValidation(passed=True, msg=None)
            )
        except _DataError as e:
            pytorch_data_split_validation_dict[split.value] = (
                _PyTorchDataSplitValidation(passed=False, msg=str(e))
            )
    pytorch_data_validation = _PyTorchDataValidation(
        passed=all(v.passed for v in pytorch_data_split_validation_dict.values()),
        **pytorch_data_split_validation_dict,
    )
    passed = passed and pytorch_data_validation.passed
    passed_critical = passed_critical and pytorch_data_validation.passed

    return DataValidationOutput(
        passed=passed,
        passed_critical=passed_critical,
        sample_rate=sample_rate_validation,
        length=length_validation,
        input_version=str(input_version),
        latency=latency_analysis,
        checks=data_checks,
        pytorch=pytorch_data_validation,
    )
