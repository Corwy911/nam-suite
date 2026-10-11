# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from typing import Optional as _Optional

import torch as _torch

from .._dependencies.auraloss.freq import MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss


def apply_pre_emphasis_filter(x: _torch.Tensor, coef: float) -> _torch.Tensor:
    """
    Apply first-order pre-emphsis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


def esr(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    ESR of (a batch of) predictions & targets

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    if preds.ndim == 1 and targets.ndim == 1:
        preds, targets = preds[None], targets[None]
    if preds.ndim != 2:
        raise ValueError(
            f"Expect 2D predictions (batch_size, num_samples). Got {preds.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expect 2D targets (batch_size, num_samples). Got {targets.shape}"
        )
    return _torch.mean(
        _torch.mean(_torch.square(preds - targets), dim=1)
        / _torch.mean(_torch.square(targets), dim=1)
    )


def multi_resolution_stft_loss(
    preds: _torch.Tensor,
    targets: _torch.Tensor,
    loss_func: _Optional[_MultiResolutionSTFTLoss] = None,
    device: _Optional[_torch.device] = None,
) -> _torch.Tensor:
    """
    Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
    B: Batch size
    L: Sequence length

    :param preds: (B,L)
    :param targets: (B,L)
    :param loss_func: A pre-initialized instance of the loss function module. Providing
        this saves time.
    :param device: If provided, send the preds and targets to the provided device.
    :return: ()
    """

    def ensure_shape(z: _torch.Tensor) -> _torch.Tensor:
        """
        Required for auraloss v0.4

        :param z: (L,) or (B,L)
        :return: (B,C,L)
        """
        if z.ndim == 1:
            return z[None, None, :]
        elif z.ndim == 2:
            return z[:, None, :]
        else:
            assert z.ndim == 3, f"Expected 1D or 2D tensor. Got {z.shape}"
            return z

    loss_func = _MultiResolutionSTFTLoss() if loss_func is None else loss_func
    if device is not None:
        preds, targets = [z.to(device) for z in (preds, targets)]
    preds, targets = [ensure_shape(z) for z in (preds, targets)]
    return loss_func(preds, targets)


def mse(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    MSE loss
    """
    return _torch.nn.functional.mse_loss(preds, targets)


def mse_fft(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    Fourier loss

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    fp = _torch.fft.fft(preds)
    ft = _torch.fft.fft(targets)
    e = fp - ft
    return _torch.mean(_torch.square(e.abs()))


def _bandlimit_signal(
    x: _torch.Tensor,
    sample_rate: float,
    low_hz: _Optional[float] = None,
    high_hz: _Optional[float] = None,
) -> _torch.Tensor:
    if x.ndim == 1:
        x_batch = x[None]
    elif x.ndim == 2:
        x_batch = x
    else:
        raise ValueError(
            "Band-limited losses expect 1D or 2D tensors shaped like (samples,) "
            f"or (batch, samples). Got {x.shape}."
        )

    fx = _torch.fft.rfft(x_batch, dim=-1)
    freqs = _torch.fft.rfftfreq(x_batch.shape[-1], d=1.0 / sample_rate).to(x_batch.device)
    band_mask = _torch.ones_like(freqs, dtype=_torch.bool)
    if low_hz is not None:
        band_mask = band_mask & (freqs >= low_hz)
    if high_hz is not None:
        band_mask = band_mask & (freqs < high_hz)
    if not _torch.any(band_mask):
        raise ValueError(
            "The requested frequency band is empty for the current sample rate "
            f"and sequence length: low_hz={low_hz}, high_hz={high_hz}, "
            f"sample_rate={sample_rate}, length={x_batch.shape[-1]}"
        )

    filtered = _torch.zeros_like(fx)
    filtered[..., band_mask] = fx[..., band_mask]
    y = _torch.fft.irfft(filtered, n=x_batch.shape[-1], dim=-1)
    return y[0] if x.ndim == 1 else y


def band_mse(
    sample_rate: float,
    low_hz: _Optional[float] = None,
    high_hz: _Optional[float] = None,
):
    """
    Factory for an MSE loss computed on a band-limited reconstruction.

    This is gentler than a band-only ESR objective and works well for
    fine-tuning toward low, mid, or high frequency regions without letting
    the rest of the spectrum collapse.
    """

    if low_hz is None and high_hz is None:
        raise ValueError("At least one of low_hz or high_hz must be provided.")

    def _loss(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
        preds_band = _bandlimit_signal(
            preds, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz
        )
        targets_band = _bandlimit_signal(
            targets, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz
        )
        return mse(preds_band, targets_band)

    return _loss


def band_esr(
    sample_rate: float,
    low_hz: _Optional[float] = None,
    high_hz: _Optional[float] = None,
):
    """
    Factory for an ESR-like loss computed only over a frequency band.

    Useful for fine-tuning a model toward low, mid, or high frequency content.
    """

    if low_hz is None and high_hz is None:
        raise ValueError("At least one of low_hz or high_hz must be provided.")

    def _loss(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
        if preds.ndim == 1 and targets.ndim == 1:
            preds_batch, targets_batch = preds[None], targets[None]
        else:
            preds_batch, targets_batch = preds, targets
        if preds_batch.ndim != 2 or targets_batch.ndim != 2:
            raise ValueError(
                "band_esr expects 1D or 2D tensors shaped like (samples,) or "
                f"(batch, samples). Got {preds.shape} and {targets.shape}."
            )

        fp = _torch.fft.rfft(preds_batch, dim=-1)
        ft = _torch.fft.rfft(targets_batch, dim=-1)
        freqs = _torch.fft.rfftfreq(
            targets_batch.shape[-1], d=1.0 / sample_rate
        ).to(targets_batch.device)

        band_mask = _torch.ones_like(freqs, dtype=_torch.bool)
        if low_hz is not None:
            band_mask = band_mask & (freqs >= low_hz)
        if high_hz is not None:
            band_mask = band_mask & (freqs < high_hz)
        if not _torch.any(band_mask):
            raise ValueError(
                "The requested frequency band is empty for the current sample rate "
                f"and sequence length: low_hz={low_hz}, high_hz={high_hz}, "
                f"sample_rate={sample_rate}, length={targets_batch.shape[-1]}"
            )

        error = fp[..., band_mask] - ft[..., band_mask]
        denominator = _torch.mean(_torch.square(ft[..., band_mask].abs()), dim=-1)
        denominator = _torch.clamp(denominator, min=1.0e-12)
        numerator = _torch.mean(_torch.square(error.abs()), dim=-1)
        return _torch.mean(numerator / denominator)

    return _loss
