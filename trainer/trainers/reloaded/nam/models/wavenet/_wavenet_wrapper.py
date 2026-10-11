# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet public wrapper implementation.
"""

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np
import torch as _torch

from .._abc import ImportsWeights as _ImportsWeights
from ..base import BaseNet as _BaseNet
from ._wavenet import WaveNet as _WaveNet

_LEGACY_MODEL_VERSION = "0.5.4"
_FILM_PARAM_KEYS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


def _activation_name_for_legacy_export(activation: _Any) -> _Optional[str]:
    if isinstance(activation, str):
        return activation
    if isinstance(activation, dict):
        if set(activation.keys()) == {"type"}:
            return activation["type"]
    return None


def _legacy_compatible_layer_config(layer_config: _Dict[str, _Any]) -> _Optional[_Dict]:
    channels = layer_config.get("channels")
    if layer_config.get("bottleneck", channels) != channels:
        return None

    head1x1 = layer_config.get("head1x1", {"active": False})
    if head1x1 and head1x1.get("active", False):
        return None

    layer1x1 = layer_config.get("layer1x1", {"active": True, "groups": 1})
    if not layer1x1.get("active", True) or layer1x1.get("groups", 1) != 1:
        return None

    if layer_config.get("groups_input", 1) != 1:
        return None
    if layer_config.get("groups_input_mixin", 1) != 1:
        return None

    if any(layer_config.get(key, {}).get("active", False) for key in _FILM_PARAM_KEYS):
        return None

    slimmable = layer_config.get("slimmable")
    if slimmable is not None and (
        not isinstance(slimmable, dict) or slimmable.get("method") is not None
    ):
        return None

    dilations = layer_config.get("dilations", [])
    num_layers = len(dilations)
    activations = layer_config.get("activation", "Tanh")
    if not isinstance(activations, _Sequence) or isinstance(activations, (str, dict)):
        activations = [activations] * num_layers
    gating_modes = layer_config.get("gating_mode", ["none"] * num_layers)
    if isinstance(gating_modes, str):
        gating_modes = [gating_modes] * num_layers
    secondary_activations = layer_config.get(
        "secondary_activation", [None] * num_layers
    )
    if not isinstance(secondary_activations, _Sequence) or isinstance(
        secondary_activations, (str, dict)
    ):
        secondary_activations = [secondary_activations] * num_layers

    if any(mode not in (None, "none") for mode in gating_modes):
        return None
    if any(secondary is not None for secondary in secondary_activations):
        return None

    activation_names = [_activation_name_for_legacy_export(a) for a in activations]
    if any(name is None for name in activation_names):
        return None
    if len(set(activation_names)) != 1:
        return None

    kernel_sizes = layer_config.get("kernel_sizes", layer_config.get("kernel_size"))
    if isinstance(kernel_sizes, int):
        kernel_size = kernel_sizes
    elif isinstance(kernel_sizes, _Sequence) and len(kernel_sizes) > 0:
        if len(set(kernel_sizes)) != 1:
            return None
        kernel_size = kernel_sizes[0]
    else:
        return None

    head_config = layer_config.get("head")
    if head_config is None:
        head_size = layer_config.get("head_size")
        head_bias = layer_config.get("head_bias")
    else:
        head_size = head_config.get("out_channels")
        head_bias = head_config.get("bias", True)
        if head_config.get("kernel_size", 1) != 1:
            return None
    if head_size is None:
        return None

    return {
        "input_size": layer_config["input_size"],
        "condition_size": layer_config["condition_size"],
        "head_size": head_size,
        "channels": channels,
        "kernel_size": kernel_size,
        "dilations": dilations,
        "head_bias": head_bias,
        "activation": activation_names[0],
        "gated": False,
    }


def _legacy_compatible_export_dict(model_dict: _Dict[str, _Any]) -> _Optional[_Dict]:
    config = model_dict.get("config", {})
    if "condition_dsp" in config:
        return None

    layers = config.get("layers")
    if not isinstance(layers, list):
        return None

    legacy_layers = []
    for layer_config in layers:
        legacy_layer = _legacy_compatible_layer_config(layer_config)
        if legacy_layer is None:
            return None
        legacy_layers.append(legacy_layer)

    legacy_dict = _deepcopy(model_dict)
    legacy_dict["version"] = _LEGACY_MODEL_VERSION
    legacy_dict["config"] = {
        "layers": legacy_layers,
        "head": config.get("head"),
        "head_scale": config.get("head_scale", 1.0),
    }
    return legacy_dict


class WaveNet(_BaseNet, _ImportsWeights):
    def __init__(
        self, wavenet: _WaveNet, sample_rate: _Optional[float] = None, **kwargs
    ):
        super().__init__(sample_rate=sample_rate)
        self._net = wavenet

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        wavenet = _WaveNet.init_from_config(config)

        return {"sample_rate": sample_rate, "wavenet": wavenet}

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def import_weights(self, weights: _Sequence[float], i: int = 0) -> int:
        weights_tensor = (
            weights if isinstance(weights, _torch.Tensor) else _torch.Tensor(weights)
        )
        return self._net.import_weights(weights_tensor, i)

    def _export_config(self):
        return self._net.export_config(sample_rate=self.sample_rate)

    def _export_weights(self) -> _np.ndarray:
        return self._net.export_weights()

    def _get_export_dict(self):
        model_dict = super()._get_export_dict()
        legacy_dict = _legacy_compatible_export_dict(model_dict)
        return model_dict if legacy_dict is None else legacy_dict

    def _forward(self, x, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("WaveNet does not support kwargs")
        if x.ndim == 2:
            x = x[:, None, :]
        if self.training and self._net.is_slimmable():
            with self._net.context_adjust_to_random():
                y = self._net(x)
        else:
            y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
