"""Output generators for the capture engine: what gets sent to the interface, block by block.

A generator is called as gen(pos, n, indata) from the audio callback and returns a float32 array (n, n_out); `finished` becomes
True once the signal (with its fade-out) is over. `stop()` asks for a fade-out (used by the Stop button and by the watchdog), so
that no signal ever ends with a click into the gear. `indata` is the input block recorded at the same time (None for backends /
call sites that do not offer one); every generator here ignores it.
"""

import numpy as np

from . import calibration as cal

FADE_IN = 2_400      # 50 ms
FADE_OUT = 2_400


class Generator:
    finished = False

    def __init__(self, n_out: int, col: int = 0, duration=None, fade_in: int = FADE_IN, fade_out: int = FADE_OUT):
        self.n_out, self.col = n_out, col
        self.duration = duration          # frames, None = until stopped
        self.fade_in, self.fade_out = fade_in, fade_out
        self._stopping = False
        self._stop_pos = None

    def signal(self, idx: np.ndarray) -> np.ndarray:        # pragma: no cover - overridden
        raise NotImplementedError

    def stop(self) -> None:
        self._stopping = True

    def __call__(self, pos: int, n: int, indata=None) -> np.ndarray:
        out = np.zeros((n, self.n_out), dtype=np.float32)
        if self.finished:
            return out
        idx = pos + np.arange(n)
        gain = np.ones(n)
        if self.fade_in:
            gain = np.sin(np.clip(idx / self.fade_in, 0.0, 1.0) * np.pi / 2.0) ** 2
        if self.duration is not None and not self._stopping and pos + n > self.duration - self.fade_out:
            self._stopping = True
            self._stop_pos = max(self.duration - self.fade_out, 0)      # fade out to end exactly at the duration
        if self._stopping:
            if self._stop_pos is None:
                self._stop_pos = pos
            ramp = 1.0 - (idx - self._stop_pos) / max(self.fade_out, 1)
            gain = gain * np.clip(ramp, 0.0, 1.0)
            if idx[-1] - self._stop_pos >= self.fade_out:
                self.finished = True
        if self.duration is not None and idx[-1] >= self.duration - 1:
            self.finished = True
        out[:, self.col] = self.signal(idx) * gain
        return out


class ToneGen(Generator):
    """A sine, fixed length (`seconds`) or open ended. `set_level` may be called from another thread while it plays: the amplitude
    then glides to the new value (at most SLEW of full scale per sample: a whole 60 dB range in 100 ms) so that turning the level
    knob never clicks. The phase comes from the absolute position, so it stays continuous."""

    SLEW = 1.0 / 4_800

    def __init__(self, level_dbfs: float, n_out: int, col: int = 0, freq: float = cal.TONE_HZ, seconds=None, rate: int = cal.RATE):
        super().__init__(n_out, col, None if seconds is None else int(round(seconds * rate)))
        self.amp, self.freq, self.rate = cal.from_db(level_dbfs), freq, rate
        self.target = self.amp

    def set_level(self, level_dbfs: float) -> None:
        self.target = cal.from_db(level_dbfs)

    def signal(self, idx):
        n = len(idx)
        delta = self.target - self.amp
        if delta:
            step = float(np.clip(delta, -self.SLEW * n, self.SLEW * n))
            amp = self.amp + step * (np.arange(1, n + 1) / n)
            self.amp += step
        else:
            amp = self.amp
        return amp * np.sin(2.0 * np.pi * self.freq * idx / self.rate)


class CaptureGen(Generator):
    """What a capture plays: the warm-up blips, the input, its tail of silence, then a few blips more (see alignment.WarmupWatch and
    timing_check for why). The warm-up has no fixed length: blips every half second until `watch.start` is set (by the watch, fed with
    the recorded blocks), then silence until the input begins there. Without a watch the input starts at `start` (0: at once).

    Everything is scaled by `gain_db`, the blips too: they stand for the input's own blips, which get the same trim."""

    def __init__(self, samples: np.ndarray, n_out: int, col: int = 0, gain_db: float = 0.0, tail: int = 0, watch=None, start: int = 0):
        super().__init__(n_out, col, duration=None, fade_in=0)
        from . import alignment as al
        self.al = al
        self.samples = np.asarray(samples, dtype=np.float32).ravel()
        self.gain = cal.from_db(gain_db)
        self.tail, self.watch = tail, watch
        self.start = None if watch is not None else int(start)
        self.tail_blips: list = []
        self.end = None
        if self.start is not None:
            self._place()

    def _place(self) -> None:
        file_end = self.start + len(self.samples) + self.tail
        self.tail_blips = [file_end + self.al.TAIL_FIRST + k * self.al.BLIP_EVERY for k in range(self.al.TAIL_BLIPS)]
        self.end = self.tail_blips[-1] + self.al.RESP_SPAN + 8_192 if self.tail_blips else file_end

    def signal(self, idx):
        out = np.zeros(len(idx))
        if self.start is None:
            if self.watch.start is not None:          # decided while the previous block was recorded
                self.start = self.watch.start
                self._place()
            else:
                for p in self.watch.blips:
                    if idx[0] <= p <= idx[-1]:
                        out[p - idx[0]] = self.al.BLIP_VALUE
                        self.watch.played.append(p)
                return out * self.gain
        rel = idx - self.start
        ok = (rel >= 0) & (rel < len(self.samples))
        out[ok] = self.samples[rel[ok]]
        for p in self.tail_blips:
            if idx[0] <= p <= idx[-1]:
                out[p - idx[0]] = self.al.BLIP_VALUE
        return out * self.gain

    def __call__(self, pos, n, indata=None):
        block = super().__call__(pos, n, indata)
        if self.end is not None and pos + n >= self.end:
            self.finished = True
        return block


class ClickGen(Generator):
    """Single-sample clicks at known positions, to time the round trip of the interface alone."""

    def __init__(self, positions, amp: float, n_out: int, col: int = 0, tail: int = 24_000):
        super().__init__(n_out, col, duration=None, fade_in=0)
        self.positions = np.asarray(sorted(positions), dtype=np.int64)
        self.amp = amp
        self.end = int(self.positions[-1]) + tail

    def signal(self, idx):
        return np.where(np.isin(idx, self.positions), self.amp, 0.0)

    def __call__(self, pos, n, indata=None):
        block = super().__call__(pos, n, indata)
        if pos + n >= self.end:
            self.finished = True
        return block


class Silence(Generator):
    """Nothing out; used to meter the inputs."""

    def __init__(self, n_out: int):
        super().__init__(n_out, 0, duration=None, fade_in=0)

    def signal(self, idx):
        return np.zeros(len(idx))
