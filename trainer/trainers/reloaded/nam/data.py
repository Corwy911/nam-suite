# File: data.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Functions and classes for working with audio data with NAM
"""

import abc as _abc
import logging as _logging
import math as _math
import wave as _wave
from collections import namedtuple as _namedtuple
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import (
    Any as _Any,
    Callable as _Callable,
    List as _List,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import librosa as _librosa
import numpy as _np
import torch as _torch
from torch.utils.data import Dataset as _Dataset
from soundfile import SoundFile
from tqdm import tqdm as _tqdm

from ._core import (
    InitializableFromConfig as _InitializableFromConfig,
    WithTeardown as _WithTeardown,
)
from ._handshake import HandshakeError as _HandshakeError
from .hooks import ExportModelDictPostHook as _ExportModelDictPostHook
from .util import init as _init

logger = _logging.getLogger(__name__)

_REQUIRED_CHANNELS = 1  # Mono
WavInfo = _namedtuple("WavInfo", ("bit_depth", "rate"))


class Split(_Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class DataError(Exception):
    """Parent class for all special Exceptions raised by NAM datasets"""
    pass


class AudioShapeMismatchError(ValueError, DataError):
    """
    Exception when the shapes (number of samples, number of channels) of two audio files
    don't match but were supposed to.
    Note that in the audio context we use the term "samples", but in `soundfile` (which returns
    the audio file's contents as a NumPy array) the term "frames" may be used to mean the same thing.
    """
    def __init__(self, shape_expected, shape_actual, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_expected = shape_expected
        self._shape_actual = shape_actual

    @property
    def shape_expected(self):
        return self._shape_expected

    @property
    def shape_actual(self):
        return self._shape_actual


# This was needed in the original NAM code due to the use of `wavio`.
#
# Since `SoundFile.read()` returns a NumPy array - and also because the `soundfile` library allows the
# use of 32-bit floating-point inputs (`wavio` is limited to integer formats) - some of the logic in
# the original NAM code is no longer needed.
# TODO: Refactor the entire codebase, including what's right below this comment.
def wav_to_np(
    filename: _Union[str, _Path],
    require_match: _Optional[_Union[str, _Path]] = None,
    required_shape: _Optional[_Tuple[int, ...]] = None,
    required_samplerate: _Optional[int] = None,
    required_bit_depth: _Optional[str] = None,
    required_wavinfo: _Optional[WavInfo] = None,
    rate: _Optional[int] = None,
    preroll: _Optional[int] = None,
    info: bool = False,
) -> _Union[_np.ndarray, _Tuple[_np.ndarray, WavInfo]]:
    """
    :param filename: Where to load from
    :param require_match: If not `None`, assert that the data you get matches the shape
        and other characteristics of another audio file at the provided location.
    :param required_shape: If not `None`, assert that the audio loaded is of shape
        `(num_samples, num_channels)`.
    :param required_samplerate: Expected sample rate. `None` allows for anything.
    :param required_bit_depth: Expected bit depth. `None` allows for anything.
    :param required_wavinfo: Backward-compatible combined bit depth / sample rate info.
    :param rate: Backward-compatible alias for `required_samplerate`.
    :param preroll: Drop this number of samples from the beginning of the audio data.
    :param info: If `True`, also return the WAV info of this file.
    """
    if required_wavinfo is not None:
        if required_samplerate is None:
            required_samplerate = required_wavinfo.rate
        if required_bit_depth is None:
            required_bit_depth = required_wavinfo.bit_depth
    if rate is not None:
        if required_samplerate is not None and required_samplerate != rate:
            raise ValueError(
                f"Conflicting sample rate requirements {required_samplerate} versus {rate}"
            )
        required_samplerate = rate

    def main(
        filename: _Union[str, _Path],
        require_match: _Optional[_Union[str, _Path]] = None,
        required_shape: _Optional[_Tuple[int, ...]] = None,
        required_samplerate: _Optional[int] = None,
        required_bit_depth: _Optional[str] = None,
        preroll: _Optional[int] = None,
        info: bool = False,
    ):
        with SoundFile(str(filename)) as dry_input_file:
            assert dry_input_file.channels == _REQUIRED_CHANNELS, "Mono"
            if required_samplerate is not None and dry_input_file.samplerate != required_samplerate:
                raise RuntimeError(
                    f"Explicitly expected sample rate of {required_samplerate}, but found {dry_input_file.samplerate} in "
                    f"file {filename}!"
                )

            if require_match is not None:
                assert required_shape is None
                assert required_bit_depth is None
                assert required_samplerate is None
                with SoundFile(str(require_match)) as file_to_match:
                    required_shape = (len(file_to_match),)
                    required_bit_depth = file_to_match.subtype
                    required_samplerate = file_to_match.samplerate
            if required_samplerate is not None and dry_input_file.samplerate != required_samplerate:
                raise ValueError(
                    f"Mismatched rates {dry_input_file.samplerate} versus {required_samplerate}"
                )
            if required_bit_depth is not None and dry_input_file.subtype != required_bit_depth:
                raise ValueError(
                    f"Mismatched bit depth {dry_input_file.subtype} versus {required_bit_depth}"
                )
            dry_input_file.seek(0)
            arr = dry_input_file.read(dtype="float32", always_2d=False)
            if preroll is not None:
                arr = arr[preroll:]
            if required_shape is not None:
                expected_shape = required_shape
                if len(expected_shape) == 2 and expected_shape[1] == 1:
                    expected_shape = (expected_shape[0],)
                if arr.shape != expected_shape:
                    raise AudioShapeMismatchError(
                        expected_shape,
                        arr.shape,
                        f"Mismatched shapes. Expected {expected_shape}, but this is "
                        f"{arr.shape}!",
                    )
            wav_info = WavInfo(dry_input_file.subtype, dry_input_file.samplerate)
        return arr if not info else (arr, wav_info)

    # I'm leaving the `librosa` fallback as-is for now.
    # TODO: Refactor this to not rely on `wavio`, and use `soundfile` instead.
    def librosa_fallback(
        filename: _Union[str, _Path],
        rate: _Optional[int] = None,
        require_match: _Optional[_Union[str, _Path]] = None,
        required_shape: _Optional[_Tuple[int, ...]] = None,
        required_wavinfo: _Optional[WavInfo] = None,
        preroll: _Optional[int] = None,
        info: bool = False,
    ):

        x_wav, float_sample_rate = _librosa.load(str(filename), sr=None, mono=False)
        sample_rate = int(float_sample_rate)
        if _np.abs(sample_rate - float_sample_rate) > 0.0001:
            raise RuntimeError(
                f"Encountered unsupported non-integer sample rate {float_sample_rate} in file {filename}!"
            )
        # Librosa returns a 1-dimensional array if mono. instead of (N,1)
        x_sampwidth = None
        if x_wav.ndim > 1:
            raise NotImplementedError("Multi-channel audio not supported")
        # Can probably get rid of this
        x_wav = x_wav[:, None]
        if rate is not None and sample_rate != rate:
            raise RuntimeError(
                f"Explicitly expected sample rate of {rate}, but found {sample_rate} in "
                f"file {filename}!"
            )

        if require_match is not None:
            assert required_shape is None
            assert required_wavinfo is None
            y_wav, y_sample_rate = _librosa.load(str(require_match), sr=None)
            required_shape = y_wav.shape
            # HACK sample width
            y_sampwidth = 3
            required_wavinfo = WavInfo(y_sampwidth, y_sample_rate)
        if required_wavinfo is not None:
            if sample_rate != required_wavinfo.rate:
                raise ValueError(
                    f"Mismatched rates {sample_rate} versus {required_wavinfo.rate}"
                )
        arr_premono = x_wav[preroll:]
        if required_shape is not None:
            if arr_premono.shape != required_shape:
                raise AudioShapeMismatchError(
                    required_shape,  # Expected
                    arr_premono.shape,  # Actual
                    f"Mismatched shapes. Expected {required_shape}, but this is "
                    f"{arr_premono.shape}!",
                )
            # sampwidth fine--we're just casting to 32-bit float anyways
        arr = arr_premono[:, 0]
        return arr if not info else (arr, WavInfo(x_sampwidth, sample_rate))

    required_wavinfo = (
        None
        if required_samplerate is None and required_bit_depth is None
        else WavInfo(required_bit_depth, required_samplerate)
    )

    try:
        return main(
            filename=filename,
            require_match=require_match,
            required_shape=required_shape,
            required_samplerate=required_samplerate,
            required_bit_depth=required_bit_depth,
            preroll=preroll,
            info=info,
         )
    except _wave.Error:
        return librosa_fallback(
            filename=filename,
            rate=required_samplerate,
            require_match=require_match,
            required_shape=required_shape,
            required_wavinfo=required_wavinfo,
            preroll=preroll,
            info=info,
        )


def wav_to_tensor(
    *args, info: bool = False, **kwargs
) -> _Union[_torch.Tensor, _Tuple[_torch.Tensor, _Tuple[int, str]]]:
    """:param info: If `True`, also return a Tuple with the sample rate and bit depth of this file."""
    np_array_and_maybe_samplerate_and_bit_depth = wav_to_np(*args, info=info, **kwargs)
    if info:
        numpy_array, info = np_array_and_maybe_samplerate_and_bit_depth
        # NOTE:
        # A two-dimensional NumPy (frames x channels) array is returned.
        # If the SoundFile has only one channel, a one-dimensional array is returned.
        # Use `always_2d=True` to return a two-dimensional array anyway.

        # E.g.:
        # _torch.Tensor(numpy_array), 'FLOAT'
        return _torch.Tensor(numpy_array), info
    else:
        numpy_array = np_array_and_maybe_samplerate_and_bit_depth
        return _torch.Tensor(numpy_array)


def tensor_to_wav(x: _torch.Tensor, *args, **kwargs):
    """Side effect: writes an audio file to storage."""
    # Move the data from the accelerator (typically a CUDA GPU) to CPU:
    # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
    #
    # `tensor.detach().cpu()` is preferred over `tensor.cpu().detach()` to avoid unnecessary
    # autograd edge construction.
    np_to_wav(x.detach().cpu().numpy(), *args, **kwargs)


def np_to_wav(
    numpy_array: _np.ndarray,
    filename: _Union[str, _Path],
    samplerate: int = 48_000,
    bit_depth: str = 'FLOAT',
    **kwargs,
):
    """See `soundfile.available_formats()` and `soundfile.available_subtypes()` for options."""
    max_value = numpy_array.max()
    min_value = numpy_array.min()
    if min_value < -1.0 or max_value > 1.0:
        print(f"""
              NumPy array which will be written to {filename} will be hard clipped!
              numpy_array.min() < -1.0: {min_value}
              numpy_array.max() > 1.0: {max_value}
              """)
    # Anything above 1.0 is reduced to 1.0. Anything below -1 is reduced to -1.
    # TODO: Consider whether or not this introduces unwanted digital hard clipping.
    print(f"""
          Exporting {str(filename)} with {_REQUIRED_CHANNELS} channels,
          {samplerate} sample rate, and {bit_depth} bit depth...
          """)
    # 'w' for writing (truncates file) or 'x' for writing (raises an error if file already exists).
    #
    # pprint.pprint(soundfile.available_subtypes('WAV'))
    #   {...
    #   'DOUBLE': '64 bit float',
    #   'FLOAT': '32 bit float',
    #   ...
    #   'PCM_16': 'Signed 16 bit PCM',
    #   'PCM_24': 'Signed 24 bit PCM',
    #   'PCM_32': 'Signed 32 bit PCM',
    #   ...}
    with SoundFile(str(filename), 'x', samplerate, _REQUIRED_CHANNELS, bit_depth, 'FLOAT') as f:
        # Any value in the Numpy Array with a value outside of the range [-1, 1] is set to -1 or 1, respectively
        # NOTE: https://janhendrikewers.uk/exploring_faster_np_clip
        #   `np.core.umath.maximum(np.core.umath.minimum(X, VMAX), VMIN)` is roughly 4x faster than `np.clip`.
        # TODO: Would it be better to normalize instead of clip?
        # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
        #   peak_value = np.max(np.abs((numpy_array))
        #   normalized_to_0dBFS_peak = numpy_array *= (1 / peak_value)
        f.write(_np.clip(numpy_array, a_min=-1.0, a_max=1.0))


class DatasetModelHandshakeError(_HandshakeError):
    """
    Raised if a handshake fails from dataset to model
    """
    pass


class AbstractDataset(_Dataset, _abc.ABC, _WithTeardown):
    @_abc.abstractmethod
    def __getitem__(self, idx: int):
        """
        Get input and output audio segment for training / evaluation.
        :return:
        """
        pass

    def handshake(self, model: "nam.models.base.BaseNet"):  # noqa: F821
        """
        Perform a handshake with the model to ensure that it's compatible.
        Raise a DatasetModelHandshakeError if the handshake fails.

        :param model: The model to handshake with.
        """
        from nam.models.base import BaseNet

        if not isinstance(model, BaseNet):
            raise DatasetModelHandshakeError(f"Model is not a NAM: {type(model)}")


class XYError(ValueError, DataError):
    """
    Exceptions related to invalid x and y provided for data sets
    """

    pass


class StartStopError(ValueError, DataError):
    """
    Exceptions related to invalid start and stop arguments
    """

    pass


class StartError(StartStopError):
    pass


class StopError(StartStopError):
    pass


# In seconds. Can't be 0.5 or else v1.wav is invalid! Oops!
_DEFAULT_REQUIRE_INPUT_PRE_SILENCE = 0.4


def _samples_to_time(samples, samplerate):
    seconds = samples // samplerate
    remainder = samples % samplerate
    hours, minutes = 0, 0
    seconds_per_hour = 3600
    while seconds >= seconds_per_hour:
        hours += 1
        seconds -= seconds_per_hour
    seconds_per_minute = 60
    while seconds >= seconds_per_minute:
        minutes += 1
        seconds -= seconds_per_minute
    return f"{hours}:{minutes:02d}:{seconds:02d} and {remainder} samples"


class Dataset(AbstractDataset, _InitializableFromConfig):
    """
    Take a pair of matched audio files and serve input + output pairs.
    """

    class _ScaleOutputHook(_ExportModelDictPostHook):
        """
        Rescale exported model output to undo training-only data normalization.
        """

        def __init__(self, scale: float):
            self._scale = scale

        @property
        def scale(self) -> float:
            return self._scale

        def apply(self, model_dict: dict):
            strategy = {
                "WaveNet": self._apply_wavenet,
                "SlimmableContainer": self._apply_slimmable_container,
            }
            architecture = model_dict["architecture"]
            if architecture not in strategy:
                raise ValueError(f"Unsupported architecture: {architecture}")
            strategy[architecture](model_dict)
            return model_dict

        def _apply_wavenet(self, model_dict: dict):
            model_dict["config"]["head_scale"] *= self._scale
            model_dict["weights"][-1] *= self._scale
            self._adjust_metadata_loudness(model_dict)

        def _apply_slimmable_container(self, model_dict: dict):
            for submodel_config in model_dict["config"]["submodels"]:
                self.apply(submodel_config["model"])
            self._adjust_metadata_loudness(model_dict)

        def _adjust_metadata_loudness(self, model_dict: dict) -> None:
            metadata = model_dict.get("metadata")
            if not isinstance(metadata, dict) or "loudness" not in metadata:
                return
            metadata["loudness"] += 20.0 * _math.log10(self._scale)

    def __init__(
        self,
        dry_audio_tensor: _torch.Tensor,
        wet_audio_tensor: _torch.Tensor,
        nx: int,
        ny: _Optional[int],
        start: _Optional[int] = None,
        stop: _Optional[int] = None,
        start_samples: _Optional[int] = None,
        stop_samples: _Optional[int] = None,
        start_seconds: _Optional[_Union[int, float]] = None,
        stop_seconds: _Optional[_Union[int, float]] = None,
        delay: _Optional[_Union[int, float]] = None,
        y_scale: float = 1.0,
        x_path: _Optional[_Union[str, _Path]] = None,
        y_path: _Optional[_Union[str, _Path]] = None,
        input_gain: float = 0.0,
        sample_rate: _Optional[float] = None,
        require_input_pre_silence: _Optional[float] = _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
    ):
        """
        :param dry_audio_tensor: 'x' The input signal.
            But also sometimes the 'wet' audio file converted to a Tensor (???)
            When used to load input/output (wet/dry) audio file pairs for training:
                x = input.WAV
                y = output_from_my_tube_amp.WAV
            When used to load predictions from the model and targets from the validation dataset:
                x = validation dataset
                y = model's predictions
            A 1D array.
        :param wet_audio_tensor: 'y' The associated output from the model.
            But also sometimes the 'wet' audio file converted to a Tensor (???)
            When used to load input/output (wet/dry) audio file pairs for training:
                x = input.WAV
                y = output_from_my_tube_amp.WAV
            When used to load predictions from the model and targets from the validation dataset:
                x = validation dataset
                y = model's predictions
            A 1D array.
        :param nx: The number of samples required as input for the model. For example,
            for a ConvNet, this would be the receptive field.
        :param ny: How many samples to provide as the output array for a single "datum".
            It's usually more computationally-efficient to provide a larger `ny` than 1
            so that the forward pass can process more audio all at once. However, this
            shouldn't be too large or else you won't be able to provide a large batch
            size (where each input-output pair could be something substantially
            different and improve batch diversity).
        :param start: [DEPRECATED; use start_samples instead.] In samples; clip x and y
            at this point. Negative values are taken from the end of the audio.
        :param stop: [DEPRECATED; use stop_samples instead.] In samples; clip x and y at
            this point. Negative values are taken from the end of the audio.
        :param start_samples: Clip x and y at this point. Negative values are taken from
            the end of the audio.
        :param stop: Clip x and y at this point. Negative values are taken from the end
            of the audio.
        :param start_seconds: Clip x and y at this point. Negative values are taken from
            the end of the audio. Requires providing `sample_rate`.
        :param stop_seconds: Clip x and y at this point. Negative values are taken from
            the end of the audio. Requires providing `sample_rate`.
        :param delay: In samples. Positive means we get rid of the start of x, end of y
            (i.e. we are correcting for an alignment error in which y is delayed behind
            x). Only integer delays are supported.
        :param y_scale: Multiplies the output signal by a factor (e.g. if the data are
            too quiet).
        :param input_gain: In dB. If the input signal wasn't fed to the amp at unity
            gain, you can indicate the gain here. The dataset will multiply the raw
            audio file by the specified gain so that the true input signal amplitude
            experienced by the signal chain will be provided as input to the model. If
            you are using a reamping setup, you can estimate this by reamping a
            completely dry signal (i.e. connecting the interface output directly back
            into the input with which the guitar was originally recorded.)
        :param sample_rate: Sample rate for the data
        :param require_input_pre_silence: If provided, require that this much time (in
            seconds) preceding the start of the data set (`start`) have a silent input.
            If it's not, then raise an exception because the output due to it will leak
            into the data set that we're trying to use. If `None`, don't assert.
        """
        self._validate_x_y(dry_audio_tensor, wet_audio_tensor)
        self._sample_rate = sample_rate
        start, stop = self._validate_start_stop(
            dry_audio_tensor,
            wet_audio_tensor,
            start,
            stop,
            start_samples,
            stop_samples,
            start_seconds,
            stop_seconds,
            self.sample_rate,
        )
        if require_input_pre_silence is not None:
            self._validate_preceding_silence(
                dry_audio_tensor, start, require_input_pre_silence, self.sample_rate
            )
        # Clip start and and end of both audio file data tensors
        dry_audio_tensor, wet_audio_tensor = [z[start:stop] for z in (dry_audio_tensor, wet_audio_tensor)]
        if delay is not None and delay != 0:
            dry_audio_tensor, wet_audio_tensor = self._apply_delay(dry_audio_tensor, wet_audio_tensor, delay)
        # Decibel scale in terms of Voltage gain/loss is 20 log 10
        x_scale = 10.0 ** (input_gain / 20.0)
        dry_audio_tensor = dry_audio_tensor * x_scale
        wet_audio_tensor = wet_audio_tensor * y_scale
        self._x_path = x_path
        self._y_path = y_path
        self._validate_inputs_after_processing(dry_audio_tensor, wet_audio_tensor, nx, ny)
        self._x = dry_audio_tensor
        self._y = wet_audio_tensor
        self._nx = nx
        # If the two files are, for example, both 48000 samples long:
        #   If nx = 4096 and ny = None:
        #     ny = 48000 - 4096 + 1
        #
        # If the model's receptive field is 4096 and `NY` is set by the training script/GUI:
        #   nx = 4096
        #   ny = 8192
        #     Ergo ny = 8192
        self._ny = ny if ny is not None else len(dry_audio_tensor) - nx + 1
        self._y_scale = None

    def __getitem__(self, idx: int) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Return two Tensors containing data to be used for training or compared for loss or plotting, etc.

        dry_audio_tensor: 'x' The input signal.
            But also sometimes the 'wet' audio file converted to a Tensor (???)
            A 1D array.

        wet_audio_tensor: 'y' The associated output from the model.
            But also sometimes the 'wet' audio file converted to a Tensor (???)
            A 1D array.

        When used to load input/output (wet/dry) audio file pairs for training:
            x = input.WAV
            y = output_from_my_tube_amp.WAV
        When used to load predictions from the model and targets from the validation dataset:
            x = validation dataset
            y = model's predictions

        :return:
            Input (NX + NY - 1,)
            Output (NY,)
        """
        if idx >= len(self):
            raise IndexError(f"Attempted to access datum {idx}, but len is {len(self)}")
        # TODO: Replace all of these obtuse and overly terse variable names with more descriptive names.
        # i = Dataset item index * number of samples `ny`
        i = idx * self._ny
        # j = i + (self._nx - 1)
        j = i + self.y_offset
        # [0] = self.x[(Dataset item index * number of samples `ny`) : ((Dataset item index * number of samples `ny`) + (self._ny - 1))]
        # [1] = self.y[((Dataset item index * number of samples `ny`) + (self._nx - 1)) : (((Dataset item index * number of samples `ny`) + (self._nx - 1)) + self._ny)]
        return self.x[i : i + self._nx + self._ny - 1], self.y[j : j + self._ny]

    def __len__(self) -> int:
        n = len(self._x)
        # If ny were 1
        single_pairs = n - self._nx + 1
        # Number of slices of size NY in the input dry_audio_tensor
        # Floor division rounds down. This correlates with the setting `drop_last = True`.
        return single_pairs // self._ny

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._sample_rate

    @property
    def x(self) -> _torch.Tensor:
        """
        The input audio data

        :return: (N,)
        """
        return self._x

    @property
    def dry_audio_tensor(self) -> _torch.Tensor:
        return self._x

    @property
    def y(self) -> _torch.Tensor:
        """
        The output audio data

        :return: (N,)
        """
        return self._y

    @property
    def wet_audio_tensor(self) -> _torch.Tensor:
        return self._y

    @property
    def y_offset(self) -> int:
        return self._nx - 1

    def handshake(self, model: "nam.models.base.BaseNet"):  # noqa: F821
        super().handshake(model)
        if self._y_scale is not None:
            hook = self._ScaleOutputHook(scale=1.0 / self._y_scale)
            add_hook = True
            for other in model.export_model_dict_post_hooks:
                if isinstance(other, self._ScaleOutputHook):
                    if other.scale != hook.scale:
                        raise ValueError(
                            "Model already has a scale output hook with a different scale"
                        )
                    add_hook = False
            if add_hook:
                model.export_model_dict_post_hooks.append(hook)

    def scale_output(self, gain: float):
        gain = float(gain)
        if not _np.isfinite(gain) or gain == 0.0:
            raise ValueError(
                f"Output scale gain must be finite and non-zero; got {gain}"
            )
        self._y *= gain
        self._y_scale = gain if self._y_scale is None else self._y_scale * gain

    @classmethod
    def parse_config(cls, config):
        """
        :param config:
            Must contain:
                x_path (path-like)
                y_path (path-like)
            May contain:
                sample_rate (int)
                y_preroll (int)
                allow_unequal_lengths (bool)
            Must NOT contain:
                x (torch.Tensor) - loaded from x_path
                y (torch.Tensor) - loaded from y_path
            Everything else is passed on to __init__
        """
        config = _deepcopy(config)
        sample_rate = config.pop("sample_rate", None)
        x, x_wavinfo = wav_to_tensor(config.pop("x_path"), info=True, rate=sample_rate)
        sample_rate = x_wavinfo.rate
        if config.pop("allow_unequal_lengths", False):
            y = wav_to_tensor(
                config.pop("y_path"),
                rate=sample_rate,
                preroll=config.pop("y_preroll", None),
                required_wavinfo=x_wavinfo,
            )
            # Truncate to the shorter of the two
            if len(x) == 0:
                raise DataError("Input is zero-length!")
            if len(y) == 0:
                raise DataError("Output is zero-length!")
            n = min(len(x), len(y))
            if n < len(x):
                print(f"Truncating input to {_sample_to_time(n, sample_rate)}")
            if n < len(y):
                print(f"Truncating output to {_sample_to_time(n, sample_rate)}")
            x, y = [z[:n] for z in (x, y)]
        else:
            try:
                y = wav_to_tensor(
                    config.pop("y_path"),
                    rate=sample_rate,
                    preroll=config.pop("y_preroll", None),
                    required_shape=(len(x), 1),
                    required_wavinfo=x_wavinfo,
                )
            except AudioShapeMismatchError as e:
                # Really verbose message since users see this.
                x_samples, x_channels = e.shape_expected
                y_samples, y_channels = e.shape_actual
                msg = "Your audio files aren't the same shape as each other!"
                if x_channels != y_channels:
                    channels_to_stereo_mono = {1: "mono", 2: "stereo"}
                    msg += f"\n * The input is {channels_to_stereo_mono[x_channels]}, but the output is {channels_to_stereo_mono[y_channels]}!"
                if x_samples != y_samples:
                    msg += f"\n * The input is {_sample_to_time(x_samples, sample_rate)} long"
                    msg += f"\n * The output is {_sample_to_time(y_samples, sample_rate)} long"
                    msg += f"\n\nOriginal exception:\n{e}"
                raise DataError(msg)
        return {
            "dry_audio_tensor": x,
            "wet_audio_tensor": y,
            "sample_rate": sample_rate,
            **config,
        }

    @classmethod
    def _apply_delay(
        cls,
        dry_audio_tensor: _torch.Tensor,
        wet_audio_tensor: _torch.Tensor,
        delay: _Union[int, float],
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        # Check for floats that could be treated like ints (simpler algorithm)
        if isinstance(delay, float) and int(delay) == delay:
            delay = int(delay)
        if isinstance(delay, int):
            return cls._apply_delay_int(dry_audio_tensor, wet_audio_tensor, delay)
        else:
            raise TypeError(type(delay))

    @classmethod
    def _apply_delay_int(
        cls, dry_audio_tensor: _torch.Tensor, wet_audio_tensor: _torch.Tensor, delay: int
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        if delay > 0:
            dry_audio_tensor = dry_audio_tensor[:-delay]
            wet_audio_tensor = wet_audio_tensor[delay:]
        elif delay < 0:
            dry_audio_tensor = dry_audio_tensor[-delay:]
            wet_audio_tensor = wet_audio_tensor[:delay]
        return dry_audio_tensor, wet_audio_tensor

    @classmethod
    def _validate_start_stop(
        cls,
        dry_audio_tensor: _torch.Tensor,
        wet_audio_tensor: _torch.Tensor,
        start: _Optional[int] = None,
        stop: _Optional[int] = None,
        start_samples: _Optional[int] = None,
        stop_samples: _Optional[int] = None,
        start_seconds: _Optional[_Union[int, float]] = None,
        stop_seconds: _Optional[_Union[int, float]] = None,
        sample_rate: _Optional[int] = None,
    ) -> _Tuple[_Optional[int], _Optional[int]]:
        """
        Parse the requested start and stop trim points.

        These may be valid indices in Python, but probably point to invalid usage, so
        we will raise an exception if something fishy is going on (e.g. starting after
        the end of the file, etc)

        :return: parsed start/stop (if valid).
        """

        def parse_start_stop(s, samples, seconds, rate):
            # Assumes validated inputs
            if s is not None:
                return s
            if samples is not None:
                return samples
            if seconds is not None:
                return int(seconds * rate)
            # else
            return None

        # Resolve different ways of asking for start/stop...
        if start is not None:
            logger.warning("Using `start` is deprecated; use `start_samples` instead.")
        if start is not None:
            logger.warning("Using `stop` is deprecated; use `start_samples` instead.")
        if (
            int(start is not None)
            + int(start_samples is not None)
            + int(start_seconds is not None)
            >= 2
        ):
            raise ValueError(
                "More than one start provided. Use only one of `start`, `start_samples`, or `start_seconds`!"
            )
        if (
            int(stop is not None)
            + int(stop_samples is not None)
            + int(stop_seconds is not None)
            >= 2
        ):
            raise ValueError(
                "More than one stop provided. Use only one of `stop`, `stop_samples`, or `stop_seconds`!"
            )
        if start_seconds is not None and sample_rate is None:
            raise ValueError(
                "Provided `start_seconds` without sample rate; cannot resolve into samples!"
            )
        if stop_seconds is not None and sample_rate is None:
            raise ValueError(
                "Provided `stop_seconds` without sample rate; cannot resolve into samples!"
            )

        # By this point, we should have a valid, unambiguous way of asking.
        start = parse_start_stop(start, start_samples, start_seconds, sample_rate)
        stop = parse_start_stop(stop, stop_samples, stop_seconds, sample_rate)
        # And only use start/stop from this point.

        # We could do this whole thing with `if len(x[start: stop]==0`, but being more
        # explicit makes the error messages better for users.
        if start is None and stop is None:
            return start, stop
        if len(dry_audio_tensor) != len(wet_audio_tensor):
            raise ValueError(
                f"Input and output are different length. Input has {len(dry_audio_tensor)} samples, "
                f"and output has {len(wet_audio_tensor)}"
            )
        n = len(dry_audio_tensor)
        if start is not None:
            # Start after the files' end?
            if start >= n:
                raise StartError(
                    f"Arrays are only {n} samples long, but start was provided as {start}, "
                    "which is beyond the end of the array!"
                )
            # Start before the files' beginning?
            if start < -n:
                raise StartError(
                    f"Arrays are only {n} samples long, but start was provided as {start}, "
                    "which is before the beginning of the array!"
                )
        if stop is not None:
            # Stop after the files' end?
            if stop > n:
                raise StopError(
                    f"Arrays are only {n} samples long, but stop was provided as {stop}, "
                    "which is beyond the end of the array!"
                )
            # Start before the files' beginning?
            if stop <= -n:
                raise StopError(
                    f"Arrays are only {n} samples long, but stop was provided as {stop}, "
                    "which is before the beginning of the array!"
                )
        # Just in case...
        if len(dry_audio_tensor[start:stop]) == 0:
            raise StartStopError(
                f"Array length {n} with start={start} and stop={stop} would get "
                "rid of all of the data!"
            )
        return start, stop

    @classmethod
    def _validate_x_y(self, dry_audio_tensor, wet_audio_tensor):
        if len(dry_audio_tensor) != len(wet_audio_tensor):
            raise XYError(
                f"Input and output aren't the same lengths! ({len(dry_audio_tensor)} vs {len(wet_audio_tensor)})"
            )
        # TODO channels
        n = len(dry_audio_tensor)
        if n == 0:
            raise XYError("Input and output are empty!")

    def _validate_inputs_after_processing(self, dry_audio_tensor, wet_audio_tensor, nx, ny):
        assert dry_audio_tensor.ndim == 1
        assert wet_audio_tensor.ndim == 1
        assert len(dry_audio_tensor) == len(wet_audio_tensor)
        if nx > len(dry_audio_tensor):
            raise RuntimeError(  # TODO XYError?
                f"Input of length {len(dry_audio_tensor)}, but receptive field is {nx}."
            )
        if ny is not None:
            assert ny <= len(wet_audio_tensor) - nx + 1
        if _torch.abs(wet_audio_tensor).max() >= 1.0:
            msg = "Wet audio Tensor (or output from the model, whatever `y` is) is clipping! _torch.abs(wet_audio_tensor).max() >= 1.0 "
            if self._y_path is not None:
                msg += f"Source is {self._y_path}"
            raise ValueError(msg)

    @classmethod
    def _validate_preceding_silence(
        cls,
        audio_tensor: _torch.Tensor,
        start: _Optional[int],
        silent_seconds: float,
        sample_rate: _Optional[float],
    ):
        """
        Make sure that the input is silent before the starting index.
        If it's not, then the output from that non-silent input will leak into the data
        set and couldn't be predicted!

        This assumes that silence is indeed required. If it's not, then don't call this!

        See: Issue #252

        :param audio_tensor: Input `_torch.Tensor`
        :param start: Starting index (where the data starts)
        :param silent_samples: How many are expected to be silent
        """
        if sample_rate is None:
            raise ValueError(
                f"Pre-silence was required for {silent_seconds} seconds, but no sample "
                "rate was provided!"
            )
        silent_samples = int(silent_seconds * sample_rate)
        if start is None:
            return
        raw_check_start = start - silent_samples
        check_start = max(raw_check_start, 0) if start >= 0 else min(raw_check_start, 0)
        check_end = start
        if not _torch.all(audio_tensor[check_start:check_end] == 0.0):
            raise XYError(
                f"Input audio Tensor isn't silent for at least {silent_samples} samples "
                f"before the starting index of {start}. Responses to this non-silent input may "
                "leak into the dataset!"
            )


class ConcatDatasetValidationError(ValueError):
    """Error raised when a ConcatDataset fails validation"""
    pass


class ConcatDataset(AbstractDataset, _InitializableFromConfig):
    def __init__(self, datasets: _Sequence[Dataset], flatten=True):
        if flatten:
            datasets = self._flatten_datasets(datasets)
        self._validate_datasets(datasets)
        self._datasets = datasets
        self._lookup = self._make_lookup()

    def __getitem__(self, idx: int) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        i, j = self._lookup[idx]
        return self.datasets[i][j]

    def __len__(self) -> int:
        """How many sub-datasets are in this dataset"""
        return sum(len(d) for d in self._datasets)

    @property
    def datasets(self):
        return self._datasets

    @property
    def nx(self) -> int:
        # Validated at initialization
        return self.datasets[0].nx

    @property
    def ny(self) -> int:
        # Validated at initialization
        return self.datasets[0].ny

    @property
    def sample_rate(self) -> _Optional[float]:
        # This is validated to be consistent across datasets during initialization
        return self.datasets[0].sample_rate

    @classmethod
    def parse_config(cls, config):
        init = _dataset_init_registry[config.get("type", "dataset")]
        return {
            "datasets": tuple(
                init(c) for c in _tqdm(config["dataset_configs"], desc="Loading data")
            )
        }

    def _flatten_datasets(self, datasets):
        """
        If any dataset is a ConcatDataset, pull it out
        """
        flattened = []
        for d in datasets:
            if isinstance(d, ConcatDataset):
                flattened.extend(d.datasets)
            else:
                flattened.append(d)
        return flattened

    def _make_lookup(self):
        """
        For faster __getitem__
        """
        lookup = {}
        offset = 0
        j = 0  # Dataset index
        for i in range(len(self)):
            if offset == len(self.datasets[j]):
                offset -= len(self.datasets[j])
                j += 1
            lookup[i] = (j, offset)
            offset += 1
        # Assert that we got to the last data set
        if j != len(self.datasets) - 1:
            raise RuntimeError(
                f"During lookup population, didn't get to the last dataset (index "
                f"{len(self.datasets)-1}). Instead index ended at {j}."
            )
        if offset != len(self.datasets[-1]):
            raise RuntimeError(
                "During lookup population, didn't end at the index of the last datum "
                f"in the last dataset. Expected index {len(self.datasets[-1])}, got "
                f"{offset} instead."
            )
        return lookup

    @classmethod
    def _validate_datasets(cls, datasets: _Sequence[Dataset]):
        # Ensure that a couple attrs are consistent across the sub-datasets.
        Reference = _namedtuple("Reference", ("index", "val"))
        references = {name: None for name in ("nx", "ny", "sample_rate")}
        for i, d in enumerate(datasets):
            for name in references.keys():
                this_val = getattr(d, name)
                if references[name] is None:
                    references[name] = Reference(i, this_val)

                if this_val != references[name].val:
                    raise ConcatDatasetValidationError(
                        f"Mismatch between {name} of datasets {references[name].index} "
                        f"({references[name].val}) and {i} ({this_val})"
                    )


class JointDatasetHook(_abc.ABC):
    @_abc.abstractmethod
    def apply(
        self, dataset_train: AbstractDataset, dataset_validation: AbstractDataset
    ):
        pass


class JointDatasetValidationError(RuntimeError):
    pass


class _AssertSameSampleRate(JointDatasetHook):
    def apply(
        self, dataset_train: AbstractDataset, dataset_validation: AbstractDataset
    ):
        train_sample_rate = getattr(dataset_train, "sample_rate", None)
        validation_sample_rate = getattr(dataset_validation, "sample_rate", None)
        if train_sample_rate != validation_sample_rate:
            raise JointDatasetValidationError(
                "Train and validation data loaders have different data set sample "
                f"rates: {train_sample_rate}, {validation_sample_rate}"
            )


def get_joint_dataset_hooks(hook_configs: _List[dict]) -> _List[JointDatasetHook]:
    hooks = [_AssertSameSampleRate()]
    for hook_config in hook_configs:
        hook = _init(
            hook_config["name"],
            *hook_config.get("args", []),
            **hook_config.get("kwargs", {}),
        )
        hooks.append(hook)
    return hooks


def apply_joint_dataset_hooks(
    dataset_train: AbstractDataset,
    dataset_validation: AbstractDataset,
    hooks: _Sequence[JointDatasetHook],
):
    for hook in hooks:
        hook.apply(dataset_train=dataset_train, dataset_validation=dataset_validation)


class NormalizeJointDatasetOutput(JointDatasetHook):
    def __init__(self, level_rms_dbfs: float):
        self._level_rms_dbfs = level_rms_dbfs

    def apply(
        self, dataset_train: AbstractDataset, dataset_validation: AbstractDataset
    ):
        train_datasets = list(_iter_base_datasets(dataset_train, label="Train"))
        validation_datasets = list(
            _iter_base_datasets(dataset_validation, label="Validation")
        )
        train_sum_squares = sum(
            _torch.sum(_torch.square(dataset.y)).item() for dataset in train_datasets
        )
        train_numel = sum(dataset.y.numel() for dataset in train_datasets)
        if train_numel == 0:
            raise JointDatasetValidationError(
                "Train dataset is empty; cannot normalize"
            )
        if train_sum_squares == 0.0:
            raise JointDatasetValidationError(
                "Train dataset is all zeroes; cannot normalize"
            )
        train_rms = _np.sqrt(train_sum_squares / train_numel)
        scale_factor = 10 ** (self._level_rms_dbfs / 20) / train_rms
        if not _np.isfinite(scale_factor) or scale_factor == 0.0:
            raise RuntimeError(
                "Scale factor is invalid. Your data must have an `inf` or `nan` in it."
            )
        for dataset in train_datasets + validation_datasets:
            dataset.scale_output(gain=scale_factor)


def normalize_joint_dataset_output(*args, **kwargs) -> NormalizeJointDatasetOutput:
    return NormalizeJointDatasetOutput(*args, **kwargs)


def _iter_base_datasets(dataset: AbstractDataset, label: str):
    if isinstance(dataset, Dataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_base_datasets(child, label=label)
    else:
        raise JointDatasetValidationError(
            f"{label} dataset is not a NAM dataset: {type(dataset)}"
        )


_dataset_init_registry = {"dataset": Dataset.init_from_config}


def register_dataset_initializer(
    name: str, constructor: _Callable[[_Any], AbstractDataset], overwrite=False
):
    """
    If you have other dataset types, you can register their initializer by name using
    this.

    For example, the basic NAM is registered by default under the name "default", but if
    it weren't, you could register it like this:

    >>> from nam import data
    >>> data.register_dataset_initializer("parametric", MyParametricDataset.init_from_config)

    :param name: The name that'll be used in the config to ask for the dataset type
    :param constructor: The constructor that'll be fed the config.
    """
    if name in _dataset_init_registry and not overwrite:
        raise KeyError(
            f"A constructor for dataset name '{name}' is already registered!"
        )
    _dataset_init_registry[name] = constructor


def init_dataset(config, split: Split) -> AbstractDataset:
    name = config.get("type", "dataset")
    base_config = config[split.value]
    common = config.get("common", {})
    if isinstance(base_config, dict):
        init = _dataset_init_registry[name]
        return init({**common, **base_config})
    elif isinstance(base_config, list):
        return ConcatDataset.init_from_config(
            {
                "type": name,
                "dataset_configs": [{**common, **c} for c in base_config],
            }
        )
