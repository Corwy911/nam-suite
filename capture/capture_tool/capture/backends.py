"""Audio backends of the capture engine: real interfaces through sounddevice / PortAudio (ASIO on Windows, Core Audio on macOS)
and a simulated interface.

Both offer the same tiny interface:
    list_devices()                       -> [{id, name, in_channels, out_channels, rate, kind}]
    open(device_id, in_ch, out_ch, rate, blocksize)
    info()                               -> {rate, blocksize, latency_in, latency_out}   (frames)
    run(gen, frames, stop, on_block)     -> (recording | None, stats)
    close()
`gen` is an output generator (see generators.py): gen(pos, n) -> float32 array (n, n_out); gen.finished ends the run.
On Windows only ASIO is offered for real hardware: it is the one driver model with exact channel selection, a fixed sample rate,
and a round trip that does not change from run to run, which is what a calibrated capture needs. On macOS Core Audio has all
of that (the driver of the interface fixes the rate and the buffer), provided the device already runs at the requested rate: the
rate is never changed from here, and a device that would need sample-rate conversion is refused.
"""

import math
import os
import re
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import calibration as cal

if sys.platform == "win32":
    os.environ.setdefault("SD_ENABLE_ASIO", "1")  # must be set before sounddevice is imported (it picks the ASIO-enabled PortAudio)
STALL_SECONDS = 2.0                                # no audio callback for this long = a stalled driver


class AudioError(Exception):
    """A failure the page can show: `key` is an i18n key suffix (cap.err.<key>), `detail` the technical text."""

    def __init__(self, key: str, detail: str = ""):
        super().__init__(f"{key}: {detail}" if detail else key)
        self.key = key
        self.detail = detail


# ------------------------------------------------------------------------------ real interfaces (PortAudio)

class PortAudioBackend:
    """One duplex stream on one device. The subclasses say which host API they list (`host_api`) and how channels are selected
    (`_settings`); opening, the probe, the stall watch and the run loop are the same for all of them."""

    kind = ""
    host_api = ""

    def __init__(self):
        import sounddevice as sd

        self.sd = sd
        self.stream = None
        self.device = None
        self.in_ch: List[int] = []
        self.out_ch: List[int] = []
        self.rate = 48_000
        self.blocksize = 0

    def _usable(self, dev) -> bool:
        return dev["max_input_channels"] > 0 or dev["max_output_channels"] > 0

    def _settings(self, dev, in_ch: List[int], out_ch: List[int]):
        """(input, output) `extra_settings` that select exactly these device channels (`dev` is the device's query_devices entry)."""
        raise NotImplementedError

    def _check_rate(self, dev, rate: int) -> None:
        """Hook: refuse early when the device is not already running at `rate`."""

    def buffer_range(self, index: int) -> Optional[dict]:
        """{min, max, preferred, granularity} (frames) if this host API can report real hardware limits, else None
        (the page then falls back to showing every value in its own static list, as before). Only ASIO can answer
        this - see AsioBackend's override."""
        return None

    def resolve_blocksize(self, index: int, requested: int) -> int:
        """The buffer (frames) that will really be asked of the driver. Here: as requested (0 = let PortAudio choose); AsioBackend pins it."""
        return requested

    def _suggested_latency(self):
        """The `latency` handed to PortAudio. "low" is a word, not a size: what it turns into is the host API's own business."""
        return "low"

    def list_devices(self) -> list:
        if self.device is None:
            # PortAudio lists the devices once, when it starts: restart it so that an interface switched on (or
            # plugged in, or - on Windows - whose ASIO driver only just finished loading) after the engine started
            # shows up without restarting the whole app. Never while a device is open: a stream must not lose its
            # device index. Used to be CoreAudioBackend-only; AsioBackend needs exactly the same refresh (Windows:
            # "Cerca schede" did nothing until the app itself was restarted, found by Roman turning an interface on
            # after opening the app), so it now lives here for every real backend instead of one platform's alone.
            try:
                self.sd._terminate()
                self.sd._initialize()
            except Exception:
                pass
        sd = self.sd
        out = []
        try:
            apis = sd.query_hostapis()
            for i, d in enumerate(sd.query_devices()):
                if apis[d["hostapi"]]["name"] == self.host_api and self._usable(d):
                    entry = {"id": f"{self.kind}:{i}", "name": d["name"], "in_channels": d["max_input_channels"],
                             "out_channels": d["max_output_channels"], "rate": d["default_samplerate"], "kind": self.kind}
                    buf = self.buffer_range(i)
                    if buf is not None:
                        entry["buffer_range"] = buf
                    out.append(entry)
        except Exception as e:
            raise AudioError("devices_failed", f"{type(e).__name__}: {e}")
        return out

    def open(self, device_id: str, in_ch: List[int], out_ch: List[int], rate: int = 48_000, blocksize: int = 0, probe: bool = True) -> None:
        """`probe` False: no test stream. Every stream is a driver session that sets the interface's ONE buffer size; when the interface is about to be
        handed over to the native engine (live test, player, tuner) the probe would be one more session, for nothing."""
        self.close()
        sd = self.sd
        index = int(device_id.split(":", 1)[1])
        try:
            dev = sd.query_devices(index)
            if not in_ch or not out_ch:
                raise AudioError("no_channels")
            if max(in_ch) >= dev["max_input_channels"] or max(out_ch) >= dev["max_output_channels"]:
                raise AudioError("bad_channels")
            self._check_rate(dev, rate)
            ain, aout = self._settings(dev, in_ch, out_ch)
            sd.check_input_settings(device=index, channels=len(in_ch), samplerate=rate, extra_settings=ain)
            sd.check_output_settings(device=index, channels=len(out_ch), samplerate=rate, extra_settings=aout)
        except AudioError:
            raise
        except Exception as e:
            raise AudioError("rate_refused", f"{rate} Hz: {type(e).__name__}: {e}")
        blocksize = self.resolve_blocksize(index, blocksize)
        self.device, self.in_ch, self.out_ch, self.rate, self.blocksize = index, list(in_ch), list(out_ch), rate, blocksize
        self.device_name = dev["name"]
        self._extra = (ain, aout)
        print(f"[driver] open {self.device_name}: buffer {blocksize} ({'probe' if probe else 'no probe'}), in {self.in_ch}, out {self.out_ch}", file=sys.stderr, flush=True)
        if not probe:
            self._latency = (0, 0)
            return
        try:
            self._probe(rate, blocksize)
        except BaseException:
            self.device = None                     # not open: a device list may be refreshed again
            raise

    def _probe(self, rate: int, blocksize: int) -> None:
        # A short probe stream tells the real latency now (and fails early if the driver is busy or the rate is wrong).
        count = {"n": 0}

        def probe_callback(indata, outdata, frames, t, status):
            count["n"] += 1
            outdata.fill(0)

        try:
            probe = self._make_stream(callback=probe_callback)
            with probe:
                time.sleep(0.3)
                before = count["n"]
                time.sleep(0.4)
                self._latency = tuple(int(round(v * rate)) for v in probe.latency)
                self.blocksize = probe.blocksize or blocksize
                running = count["n"] - before
        except Exception as e:
            raise AudioError("open_failed", f"{type(e).__name__}: {e}")
        if running < 3:
            # Some drivers deliver their first buffers and then nothing: no sample clock (card absent, external clock selected,
            # driver panel in a bad state). Better to know now than to hang in the middle of a take.
            raise AudioError("stalled", f"{count['n']} callbacks in total, {running} in the last 0.4 s")

    def _make_stream(self, callback, finished_callback=None):
        sd = self.sd
        print(f"[driver] stream with buffer {self.blocksize or 'auto'}", file=sys.stderr, flush=True)
        return sd.Stream(device=(self.device, self.device), samplerate=self.rate, channels=(len(self.in_ch), len(self.out_ch)),
                         dtype="float32", blocksize=self.blocksize or 0, latency=self._suggested_latency(), extra_settings=self._extra,
                         callback=callback, finished_callback=finished_callback)

    def info(self) -> dict:
        lin, lout = getattr(self, "_latency", (0, 0))
        return {"rate": self.rate, "blocksize": self.blocksize, "latency_in": lin, "latency_out": lout,
                "device_name": getattr(self, "device_name", "")}

    def run(self, gen, frames: Optional[int], stop: threading.Event, on_block: Optional[Callable] = None):
        n_in = len(self.in_ch)
        rec = np.zeros((frames, n_in), dtype=np.float32) if frames else None
        state = {"pos": 0, "xruns": 0, "last": time.monotonic()}
        done = threading.Event()
        sd = self.sd

        def callback(indata, outdata, nframes, time_info, status):
            state["last"] = time.monotonic()
            if status:
                state["xruns"] += 1
            pos = state["pos"]
            if stop.is_set():
                gen.stop()
            outdata[:] = gen(pos, nframes, indata)
            if rec is not None:
                n = min(nframes, frames - pos)
                if n > 0:
                    rec[pos:pos + n] = indata[:n]
            if on_block is not None:
                on_block(indata, pos)
            state["pos"] = pos + nframes
            if gen.finished or (frames is not None and state["pos"] >= frames):
                raise sd.CallbackStop

        stalled = False
        try:
            stream = self._make_stream(callback, finished_callback=done.set)
            with stream:
                while not done.wait(0.05):
                    if time.monotonic() - state["last"] > STALL_SECONDS:
                        stalled = True                       # the driver stopped calling us: leave instead of hanging
                        stream.abort()
                        break
        except AudioError:
            raise
        except Exception as e:
            raise AudioError("stream_failed", f"{type(e).__name__}: {e}")
        if stalled:
            raise AudioError("stalled", f"no audio for {STALL_SECONDS} s after {state['pos']} frames")
        return rec, {"xruns": state["xruns"], "frames": state["pos"]}

    def close(self) -> None:
        self.stream = None
        self.device = None


class AsioBackend(PortAudioBackend):
    kind = "asio"
    host_api = "ASIO"

    def _settings(self, dev, in_ch, out_ch):
        sd = self.sd
        return sd.AsioSettings(channel_selectors=list(in_ch)), sd.AsioSettings(channel_selectors=list(out_ch))

    def _driver_range(self, index: int) -> Optional[tuple]:
        """(min, max, preferred, granularity) in frames, exactly as the driver declares them, or None."""
        try:
            import ctypes
            dll = ctypes.CDLL(self.sd._libname)
            fn = dll.PaAsio_GetAvailableBufferSizes
            fn.restype = ctypes.c_int
            c_long_p = ctypes.POINTER(ctypes.c_long)
            fn.argtypes = [ctypes.c_int, c_long_p, c_long_p, c_long_p, c_long_p]
            mn, mx, pref, gran = ctypes.c_long(), ctypes.c_long(), ctypes.c_long(), ctypes.c_long()
            if fn(index, ctypes.byref(mn), ctypes.byref(mx), ctypes.byref(pref), ctypes.byref(gran)) != 0:
                return None
            return mn.value, mx.value, pref.value, gran.value
        except (OSError, AttributeError):
            return None

    def resolve_blocksize(self, index: int, requested: int) -> int:
        """Why this exists: the driver has ONE buffer size, shared by every program (it is what the interface's own control panel shows), and each
        stream we open changes it. With `latency="low"` PortAudio decided the size by itself (not the one picked on the page: "requested 8, the
        panel says 320"), and the native engine - which opens the same interface for the live test, the player and the tuner - used the driver's
        "preferred" size instead, so every hand-over between the two moved the driver's buffer. Now ONE number is worked out here, from the
        page's choice (0 = "Automatico" = leave the size the driver has now: its preferred one) rounded to what the driver accepts, and it is
        that number that PortAudio is told (see _suggested_latency) and that goes to the native engine afterwards (info()["blocksize"])."""
        rng = self._driver_range(index)
        if rng is None:
            return requested
        mn, mx, pref, gran = rng
        # a transient answer of a driver that is still settling (min == max, seen on the Focusrite right after a close): ask again a moment later
        for _ in range(5):
            if not (mn > 0 and mn == mx):
                break
            time.sleep(0.1)
            rng = self._driver_range(index) or rng
            mn, mx, pref, gran = rng
        size = pref if requested <= 0 else min(max(requested, mn), mx)
        if gran > 0 and size > mn:
            size = mn + ((size - mn + gran // 2) // gran) * gran           # only min + n*granularity is a valid size
        elif gran < 0 and size > mn:
            p = mn
            while p * 2 <= size and p * 2 <= mx:
                p *= 2
            size = p
        return int(size)

    def _suggested_latency(self):
        # buffering target = exactly one buffer: PortAudio's ASIO layer then uses the driver buffer we named instead of one it prefers
        return (self.blocksize / self.rate) if self.blocksize else "low"

    def buffer_range(self, index: int) -> Optional[dict]:
        # sounddevice/PortAudio's own Python API has no call for this (Stream.blocksize only ever echoes back
        # whatever was requested - confirmed by testing 8/16/32 and getting exactly 8/16/32 back regardless of what
        # the driver actually granted, which is how "buffer 8" and "buffer 16 -> shows 48" happened even though this
        # Focusrite's real ASIO minimum is 16 with a granularity of 16). PortAudio's own C library *does* expose the
        # real values via its ASIO-specific extension PaAsio_GetAvailableBufferSizes - just not through sounddevice's
        # cffi bindings. It is already loaded in-process (sd._libname, the very DLL sounddevice itself uses for
        # every call above): re-opening that same path via ctypes shares the already-initialized module (Windows
        # loads a given DLL file once per process and hands out the same instance), so this needs no separate
        # Pa_Initialize and cannot conflict with sounddevice's own use of it.
        rng = self._driver_range(index)
        if rng is None:
            return None
        mn, mx, pref, gran = rng
        # PaAsio_GetAvailableBufferSizes reports what the DRIVER declares (verified: this Focusrite Scarlett 18i20 3rd gen, driver 4.150.0.432,
        # declares min=16) but with `latency="low"` that was not what PortAudio delivered: requesting 16 or 32 gave the same real round trip as 64
        # (376 samples all three times), and 48 - not a size this driver has - a WORSE one (440). That was PortAudio picking the host buffer by
        # itself; the size is now pinned (resolve_blocksize / _suggested_latency), but the 64 floor stays until that has been measured on the
        # interface: it only stops the page from offering what was shown not to work as expected.
        return {"min": max(mn, 64), "max": mx, "preferred": pref, "granularity": gran}


class CoreAudioBackend(PortAudioBackend):
    """macOS. Only devices with both inputs and outputs are listed (one duplex stream needs one device). PortAudio is left to
    use the device as the user set it (rate and buffer size come from the interface's own driver / Audio MIDI Setup); nothing
    is changed from here, and a device running at another rate is refused instead of being resampled behind our back."""

    kind = "coreaudio"
    host_api = "Core Audio"
    VIRTUAL = re.compile(r"blackhole|teams|zoom|loopback|soundflower|aggregate|multi-output|virtual", re.I)

    def _usable(self, dev) -> bool:
        return dev["max_input_channels"] > 0 and dev["max_output_channels"] > 0

    @staticmethod
    def output_map(n_device_out: int, out_ch: List[int]) -> List[int]:
        """Core Audio's output channel map is indexed by DEVICE channel and holds the STREAM channel sent there (-1: silence), so it
        is as long as the device has outputs: stream channel k goes to device channel out_ch[k]. (The input map is the plain list of
        the device channels wanted.) Getting this wrong plays on channel 1 whatever was asked."""
        m = [-1] * n_device_out
        for k, c in enumerate(out_ch):
            m[c] = k
        return m

    def _settings(self, dev, in_ch, out_ch):
        sd = self.sd
        return (sd.CoreAudioSettings(channel_map=list(in_ch)),
                sd.CoreAudioSettings(channel_map=self.output_map(dev["max_output_channels"], out_ch)))

    def _check_rate(self, dev, rate: int) -> None:
        if abs(float(dev["default_samplerate"]) - rate) > 0.5:
            raise AudioError("rate_mismatch", f"{dev['default_samplerate']:.0f} Hz, {rate} Hz needed")

    def list_devices(self) -> list:
        # The PortAudio-restart-before-listing refresh is now in PortAudioBackend.list_devices() itself (every
        # real backend needs it, not just this one) - only the real-interfaces-first sort stays Mac-specific here.
        found = super().list_devices()
        found.sort(key=lambda d: bool(self.VIRTUAL.search(d["name"])))     # real interfaces first, software devices last
        return found


def real_backend_class():
    """The backend class for real hardware on this platform."""
    return AsioBackend if sys.platform == "win32" else CoreAudioBackend


# ---------------------------------------------------------------------- simulated interface

class SimulatedBackend:
    """A 4-in / 4-out interface with known analog levels, a fixed round-trip delay and a fake gear, for tests and the demo device.

    config: co_dbu     analog dBu at the output for a 0 dBFS sine (what the multimeter would show)
            ci_dbu     analog dBu at the input that reads 0 dBFS (same for every input in this model)
            latency    round-trip delay in frames (interface + gear)
            noise_dbfs input noise floor
            route      'loopback' (output cabled to input) or 'gear' (through the fake amp / pedal)
            gear       'head' | 'pedal': level and non-linearity of the fake gear
            speed      run faster than real time by this factor (None: as fast as possible)
            slip_at    stream position where the recording moves by `slip` samples against the playback (None: never), like the
                       driver of some USB interfaces does once after the stream starts
    """
    kind = "sim"
    ID = "sim:demo"

    def __init__(self, **config):
        self.config = {"co_dbu": 14.3, "ci_dbu": 9.2, "latency": 1056, "noise_dbfs": -85.0, "route": "loopback", "gear": "head",
                       "speed": 60.0, "slip_at": None, "slip": 1, **config}
        self.in_ch: List[int] = []
        self.out_ch: List[int] = []
        self.rate = 48_000
        self.blocksize = 128

    def list_devices(self) -> list:
        return [{"id": self.ID, "name": "Demo interface (simulated)", "in_channels": 4, "out_channels": 4, "rate": 48_000, "kind": "sim"}]

    def open(self, device_id: str, in_ch, out_ch, rate: int = 48_000, blocksize: int = 0, probe: bool = True) -> None:
        if device_id != self.ID:
            raise AudioError("no_device", device_id)
        if not in_ch or not out_ch:
            raise AudioError("no_channels")
        if max(in_ch) >= 4 or max(out_ch) >= 4:
            raise AudioError("bad_channels")
        if rate != 48_000:
            raise AudioError("rate_refused", f"{rate} Hz")
        self.in_ch, self.out_ch, self.rate = list(in_ch), list(out_ch), rate
        self.blocksize = blocksize or 128

    def info(self) -> dict:
        half = self.config["latency"] // 2
        return {"rate": self.rate, "blocksize": self.blocksize, "latency_in": half, "latency_out": self.config["latency"] - half,
                "device_name": "Demo interface (simulated)"}

    def set_config(self, **changes) -> None:
        self.config.update(changes)

    # the analog side
    def _dac_vpeak(self) -> float:
        return math.sqrt(2.0) * cal.dbu_to_vrms(self.config["co_dbu"])

    def _adc_vpeak(self) -> float:
        return math.sqrt(2.0) * cal.dbu_to_vrms(self.config["ci_dbu"])

    def _gear(self, v: np.ndarray) -> np.ndarray:
        """Volts in, volts out. A head is heard through a microphone (much lower level, soft clipping); a pedal is line level."""
        if self.config["gear"] == "pedal":
            return 2.0 * np.tanh(0.9 * v / 2.0)
        return 0.9 * np.tanh(2.5 * v / 3.0)

    def multimeter_vrms(self, sent_dbfs: float = 0.0) -> float:
        """What a multimeter would read at the output for a sine sent at `sent_dbfs`."""
        return cal.dbu_to_vrms(self.config["co_dbu"]) * cal.from_db(sent_dbfs)

    def run(self, gen, frames: Optional[int], stop: threading.Event, on_block: Optional[Callable] = None):
        cfg, n_in = self.config, len(self.in_ch)
        rng = np.random.default_rng(7)
        delay = int(cfg["latency"])
        chunk = 4_800
        rec = np.zeros((frames, n_in), dtype=np.float32) if frames else None
        pending = np.zeros(delay)                      # analog samples still "in the cable"
        pos = 0
        noise = cal.from_db(cfg["noise_dbfs"])
        last_digital = np.zeros((chunk, n_in), dtype=np.float32)   # what a passthrough gen reads: the previous block's
        while True:                                                # input (this one isn't known yet) - an honest one-block delay
            if stop.is_set():
                gen.stop()
            out = gen(pos, chunk, last_digital)
            v_out = out[:, 0].astype(np.float64) * self._dac_vpeak()
            v_in = self._gear(v_out) if cfg["route"] == "gear" else v_out
            stream = np.concatenate([pending, v_in])
            at = cfg["slip_at"]
            if at is not None and pos <= at < pos + chunk:            # the driver's step: from here on everything arrives `slip` later
                i = at - pos
                stream = np.insert(stream, i, np.repeat(stream[i], cfg["slip"])) if cfg["slip"] > 0 else np.delete(stream, range(i, i - cfg["slip"]))
            v_arrive, pending = stream[:chunk], stream[chunk:]
            digital = np.clip(v_arrive / self._adc_vpeak(), -1.0, 1.0) + rng.standard_normal(chunk) * noise
            block = np.repeat(np.clip(digital, -1.0, 1.0)[:, None], n_in, axis=1).astype(np.float32)
            if rec is not None:
                n = min(chunk, frames - pos)
                if n > 0:
                    rec[pos:pos + n] = block[:n]
            if on_block is not None:
                on_block(block, pos)
            last_digital = block
            pos += chunk
            if cfg["speed"]:
                time.sleep(chunk / self.rate / cfg["speed"])
            if gen.finished or (frames is not None and pos >= frames):
                break
        return rec, {"xruns": 0, "frames": pos}

    def close(self) -> None:
        pass


def db_peak(block: np.ndarray) -> List[float]:
    """Peak per channel of an audio block, in dBFS."""
    return [cal.to_db(float(v)) for v in np.max(np.abs(block), axis=0)]
