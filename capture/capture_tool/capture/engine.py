"""The audio engine: a separate process that owns the audio interface.

Started by session.py as `python engine.py`. Protocol: one JSON object per line.
  stdin   {"id": 1, "cmd": "open", ...}           commands from the server
  stdout  {"id": 1, "ok": true, ...}              the answer to a command (ok false: "error" = i18n key suffix, "detail")
          {"event": "state", ...}                 10 Hz while an operation runs: input peaks (dBFS), progress, xruns
          {"event": "progress", ...}              free-form step reports of an operation
          {"event": "op_done", "op": ..., "ok": ..., "result": ...}   an operation finished (also: failed / stopped)
Why a process: its own GIL (no jitter from the web server or a training), a crashing audio driver cannot take the server down, and
killing it releases the interface. The engine silences the output and exits if the server stops sending heartbeats.

Threads: ASIO drivers are COM objects with thread affinity (and PortAudio is happiest this way everywhere), so EVERYTHING that touches audio (listing, opening, streams) runs on the
main thread, one command at a time; a reader thread takes the commands from stdin, answers the light ones (heartbeat, stop, simulator
settings) at once and refuses a second operation while one runs ("busy"). Operations: tone, meter, measure, return_cal, click_probe, record.
"""

import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # nam-web, so that `capture` imports when run as a script

import numpy as np  # noqa: E402

from capture import alignment as al  # noqa: E402
from capture import backends, calibration as cal, generators as gens  # noqa: E402
from capture.backends import AudioError  # noqa: E402

HEARTBEAT_TIMEOUT = 10.0
MEASURE_SECONDS = 1.5


class Engine:
    def __init__(self):
        self._out_lock = threading.Lock()
        self.sim = backends.SimulatedBackend()
        self.real = None
        self.backend = None
        self.commands: "queue.Queue" = queue.Queue()
        self.busy = False                    # a queued command (operation, open, close, devices) has not finished yet
        self.op_name = None
        self.op_stop = threading.Event()
        self.last_hb = time.time()
        self.hb_seen = False
        self.quit = threading.Event()
        # live figures, written from the audio callback and read by the 10 Hz reporter
        self.peaks = None
        self.total = None
        self.rec_start = 0                   # a take: where its input starts in the stream (None while the warm-up lasts)
        self.pos = 0
        self.xruns = 0
        self.clip_blocks = 0
        self.level_dbfs = None
        self.live_tone = None                # the open-ended tone that set_level may change while it plays

    # ------------------------------------------------------------------ plumbing
    def emit(self, obj: dict) -> None:
        line = json.dumps(obj, allow_nan=False, default=float)
        with self._out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def on_block(self, indata, pos) -> None:
        peak = np.max(np.abs(indata), axis=0)
        self.peaks = peak if self.peaks is None else np.maximum(self.peaks, peak)
        self.pos = pos + len(indata)
        if float(peak.max()) >= cal.CLIP_LEVEL:
            self.clip_blocks += 1

    def reporter(self) -> None:
        while not self.quit.is_set():
            time.sleep(0.1)
            if self.op_name is None:
                continue
            peaks, self.peaks = self.peaks, None
            self.emit({"event": "state", "op": self.op_name, "pos": self.pos, "total": self.total, "start": self.rec_start, "xruns": self.xruns,
                       "clip": self.clip_blocks > 0, "level_dbfs": self.level_dbfs,
                       "in_peak_dbfs": None if peaks is None else [round(cal.to_db(float(p)), 2) for p in peaks]})

    def watchdog(self) -> None:
        while not self.quit.is_set():
            time.sleep(1.0)
            if self.hb_seen and time.time() - self.last_hb > HEARTBEAT_TIMEOUT:
                self.op_stop.set()
                time.sleep(0.5)
                os._exit(3)

    # ------------------------------------------------------------------ operations (run on the main thread)
    def _channels(self, out_index: int, in_index: int):
        b = self.backend
        if not (0 <= out_index < len(b.out_ch)) or not (0 <= in_index < len(b.in_ch)):
            raise AudioError("bad_channels")
        return len(b.out_ch)

    def op_tone(self, stop, level_dbfs, out_index=0, in_index=0, freq=cal.TONE_HZ):
        n_out = self._channels(out_index, in_index)
        tone = gens.ToneGen(level_dbfs, n_out, out_index, freq=freq)
        self.live_tone = tone
        try:
            self.backend.run(tone, None, stop, self.on_block)
        finally:
            self.live_tone = None
        return {}

    def op_meter(self, stop):
        self.backend.run(gens.Silence(len(self.backend.out_ch)), None, stop, self.on_block)
        return {}

    def _measure(self, stop, level_dbfs, out_index, in_index, seconds=MEASURE_SECONDS) -> dict:
        b = self.backend
        n_out = self._channels(out_index, in_index)
        self.level_dbfs = level_dbfs
        frames = int(seconds * b.rate)
        rec, stats = b.run(gens.ToneGen(level_dbfs, n_out, out_index, seconds=seconds), frames, stop, self.on_block)
        x = rec[:, in_index].astype(np.float64)
        window = x[int(0.3 * b.rate):int((seconds - 0.15) * b.rate)]     # steady part: after the fade-in and the round trip, before the fade-out
        m = cal.measure_tone(window, b.rate)
        m.update(level_dbfs=level_dbfs, xruns=stats["xruns"])
        return m

    def op_measure(self, stop, level_dbfs, out_index=0, in_index=0, seconds=MEASURE_SECONDS):
        return self._measure(stop, level_dbfs, out_index, in_index, seconds)

    def _auto_search_level(self, stop, out_index, in_index, target_dbfs=-9.0, op_name="return_cal"):
        """Finds a send level that reads near `target_dbfs` without clipping (up to 8 tries), reporting progress as it goes.
        Shared by op_return_cal's automatic mode, which needs "a sensible level on this input", not
        yet knowing Ci. Returns the last measurement; `m["level_dbfs"]` is the level it settled on (or ran out at)."""
        level = -40.0
        for step in range(8):
            m = self._measure(stop, level, out_index, in_index)
            if stop.is_set():
                return m
            self.emit({"event": "progress", "op": op_name, "step": step, "level_dbfs": level, "peak_dbfs": m["peak_dbfs"],
                       "clipped": m["clipped"]})
            settled = m["clipped"] == 0 and -12.0 <= m["peak_dbfs"] <= -3.0
            if settled:
                break
            nxt = cal.suggest_send_level(level, m["peak_dbfs"], clipped=m["clipped"] > 0, target_dbfs=target_dbfs)
            if nxt == level:                                    # cannot go lower / higher: report what we have
                break
            level = nxt
        return m

    def op_return_cal(self, stop, co, out_index=0, in_index=0, level_dbfs=None, target_dbfs=-9.0, repeats=3, fixed=False):
        """Loopback calibration of an input. Automatic: find a send level that reads near `target_dbfs` without clipping. With
        `fixed` the level chosen by the user (`level_dbfs`) is used as it is, never adjusted. Then measure `repeats` times at that
        level. Ci = Co + sent - read (the NAM guide's dBu_return = dBu_send + delta - L)."""
        if fixed:
            m = self._measure(stop, level_dbfs, out_index, in_index)
            if stop.is_set():
                return {"aborted": True}
        else:
            m = self._auto_search_level(stop, out_index, in_index, target_dbfs)
            if stop.is_set():
                return {"aborted": True}
        level = m["level_dbfs"]                                   # the level the last reading was taken at (the loop may have run out)
        readings = [m]
        for _ in range(repeats - 1):
            readings.append(self._measure(stop, level, out_index, in_index))
            if stop.is_set():
                return {"aborted": True}
        cis = [cal.input_level_dbu(co, level, r["peak_dbfs"]) for r in readings]
        worst = max(readings, key=lambda r: r["peak_dbfs"])
        return {"ci": float(np.mean(cis)), "spread_db": float(max(cis) - min(cis)), "level_dbfs": level,
                "peak_dbfs": float(np.mean([r["peak_dbfs"] for r in readings])), "thdn_db": float(np.mean([r["thdn_db"] for r in readings])),
                "clipped": int(sum(r["clipped"] for r in readings)), "judge": cal.judge_level(worst["peak_dbfs"], worst["clipped"]),
                "readings": [float(c) for c in cis]}

    def op_linearity(self, stop, co, ci, out_index=0, in_index=0):
        """The input at several levels, one reading each: with a linear input the Ci derived from every reading is the same. Levels are
        chosen from the calibration so that the readings land between -40 and -6 dBFS."""
        levels = cal.linearity_levels(co, ci)
        points = []
        for step, level in enumerate(levels):
            m = self._measure(stop, level, out_index, in_index)
            if stop.is_set():
                return {"aborted": True}
            points.append({"level_dbfs": level, "peak_dbfs": m["peak_dbfs"], "sample_peak_dbfs": m["sample_peak_dbfs"],
                           "ci_dbu": cal.input_level_dbu(co, level, m["peak_dbfs"]), "thdn_db": m["thdn_db"], "clipped": m["clipped"]})
            self.emit({"event": "progress", "op": "linearity", "step": step + 1, "total": len(levels), "level_dbfs": level,
                       "peak_dbfs": m["peak_dbfs"]})
        return {"points": points}

    def op_click_probe(self, stop, out_index=0, in_index=0, amp=0.5, clicks=4):
        """Round trip of the interface alone (output cabled to input): clicks at known places, the delay each one comes back with."""
        b = self.backend
        n_out = self._channels(out_index, in_index)
        positions = [int((0.5 + 0.5 * i) * b.rate) for i in range(clicks)]
        gen = gens.ClickGen(positions, amp, n_out, out_index)
        frames = gen.end
        rec, stats = b.run(gen, frames, stop, self.on_block)
        delay = al.click_latency(rec[:, in_index], positions)
        info = b.info()
        return {"delay": delay, "ms": None if delay is None else round(delay / b.rate * 1000, 3), "xruns": stats["xruns"], "amp": amp,
                "driver_latency": info["latency_in"] + info["latency_out"]}

    def op_record(self, stop, input_path, save_path, out_index=0, in_index=0, gain_db=0.0, extra_frames=48_000, warmup=False):
        """A take. The driver of some USB interfaces moves the recording by one sample against the playback once, at a random moment after
        the stream starts (a Focusrite 18i20: 1.6 s, 4.3 s, more than 13.8 s in) - a fixed pre-roll of silence was not enough. With `warmup`
        (a switch on the page, for interfaces that do it) the input is preceded by blips that last until that step has been seen
        (alignment.WarmupWatch); always, a few blips follow it and the timing around the input is measured (alignment.timing_check). Saved: the recording from the first sample of the input on,
        `len(input) + extra_frames` long, as before; the warm-up and the blips after it only leave their numbers in `timing`."""
        import soundfile as sf

        b = self.backend
        n_out = self._channels(out_index, in_index)
        x, sr = sf.read(input_path, dtype="float32", always_2d=False)
        if sr != b.rate or x.ndim != 1:
            raise AudioError("input_format", f"{sr} Hz, shape {x.shape}")
        watch = al.WarmupWatch() if warmup else None
        gen = gens.CaptureGen(x, n_out, out_index, gain_db, tail=extra_frames, watch=watch)
        longest = (al.WARMUP_MAX + al.WARMUP_GAP if warmup else 0) + len(x) + extra_frames + al.TAIL_FIRST \
            + al.TAIL_BLIPS * al.BLIP_EVERY + al.RESP_SPAN + 16_384
        self.level_dbfs = gain_db
        self.rec_start = gen.start
        self.total = None if gen.start is None else gen.end

        def on_block(indata, pos):
            self.on_block(indata, pos)
            if watch is not None:
                watch.feed(indata[:, in_index], pos)
            if self.total is None and gen.end is not None:
                self.rec_start, self.total = gen.start, gen.end        # the warm-up is over: the page shows the progress of the take

        rec, stats = b.run(gen, longest, stop, on_block)
        if stop.is_set() or gen.start is None:
            return {"aborted": True}
        stream = rec[:, in_index]
        mono = stream[gen.start:gen.start + len(x) + extra_frames]
        if len(mono) < len(x) + extra_frames:
            mono = np.concatenate([mono, np.zeros(len(x) + extra_frames - len(mono), dtype=mono.dtype)])
        np.save(save_path, mono)
        timing = {**(watch.summary() if watch else {}),
                  **al.timing_check(stream[:max(int(stats["frames"]), 0)], gen.start, len(x), watch.played if watch else [], gen.tail_blips)}
        return {"path": save_path, "frames": int(len(mono)), "xruns": stats["xruns"], "clip_blocks": self.clip_blocks,
                "peak_dbfs": cal.to_db(float(np.max(np.abs(mono)))), "timing": timing}

    OPS = ("tone", "meter", "measure", "return_cal", "linearity", "click_probe", "record")
    # The helpers that only listen: any new command (another operation, open, devices) stops them instead of being
    # refused, and runs right after their op_done (Roman: "se clicco su calibra il tono di prova si deve spegnere" -
    # the page's own stop-then-start has a poll-lag race, so the engine must not depend on it). Real operations
    # still refuse each other: a calibration or a take must not be killed by a stray click.
    YIELDING = ("tone", "meter")

    def run_op(self, cmd: str, params: dict) -> None:
        """Runs on the main thread. The command was answered {"started"} by the reader; the outcome is an op_done event."""
        fn = getattr(self, "op_" + cmd)
        self.op_name, self.peaks, self.pos, self.xruns, self.clip_blocks, self.total = cmd, None, 0, 0, 0, None
        self.rec_start = 0
        self.level_dbfs = params.get("level_dbfs")
        try:
            result = fn(self.op_stop, **params)
            done = {"event": "op_done", "op": cmd, "ok": True, "stopped": self.op_stop.is_set(), "result": result}
        except AudioError as e:
            done = {"event": "op_done", "op": cmd, "ok": False, "error": e.key, "detail": e.detail}
        except Exception as e:
            done = {"event": "op_done", "op": cmd, "ok": False, "error": "internal", "detail": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()[-1500:]}
        self.op_name = None
        self.busy = False                     # cleared BEFORE announcing the end: the next command may follow immediately
        self.emit(done)

    # ------------------------------------------------------------------ commands on the main thread
    def real_backend(self):
        if self.real is None:
            self.real = backends.real_backend_class()()      # imports sounddevice = initialises PortAudio on THIS thread
        return self.real

    def heavy(self, cmd: str, p: dict) -> dict:
        if cmd == "devices":
            found, asio_error = [], None
            try:
                found += self.real_backend().list_devices()
            except AudioError as e:
                asio_error = {"error": e.key, "detail": e.detail}
            except Exception as e:
                asio_error = {"error": "devices_failed", "detail": f"{type(e).__name__}: {e}"}
            found += self.sim.list_devices()
            return {"devices": found, "asio_error": asio_error, "platform": sys.platform}
        if cmd == "open":
            backend = self.sim if str(p["device"]).startswith("sim:") else self.real_backend()
            backend.open(p["device"], p["in_channels"], p["out_channels"], int(p.get("rate", cal.RATE)), int(p.get("blocksize", 0)), bool(p.get("probe", True)))
            self.backend = backend
            return backend.info()
        if cmd == "close":
            if self.backend is not None:
                self.backend.close()
            self.backend = None
            return {}
        raise AudioError("unknown_command", cmd)

    def light(self, cmd: str, p: dict):
        """Answered by the reader thread at once; None means: not a light command."""
        if cmd == "hello":
            return {"version": 1, "pid": os.getpid()}
        if cmd == "hb":
            self.last_hb, self.hb_seen = time.time(), True
            return {}
        if cmd == "stop":
            self.op_stop.set()
            return {}
        if cmd == "set_level":
            tone = self.live_tone
            if tone is None:
                raise AudioError("no_tone")
            level = max(-60.0, min(0.0, float(p["level_dbfs"])))
            tone.set_level(level)
            self.level_dbfs = level
            return {"level_dbfs": level}
        if cmd == "sim":
            self.sim.set_config(**{k: v for k, v in p.items() if k in self.sim.config})
            return {"config": self.sim.config}
        if cmd == "sim_multimeter":
            return {"vrms": self.sim.multimeter_vrms(float(p.get("level_dbfs", 0.0)))}
        return None

    def read_stdin(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                cmd = msg.pop("cmd")
                mid = msg.pop("id", None)
            except Exception:
                continue
            if cmd == "quit":
                break
            try:
                result = self.light(cmd, msg)
                if result is not None:
                    if cmd != "hb":
                        self.emit({"id": mid, "ok": True, **result})
                    continue
                if cmd not in self.OPS and cmd not in ("devices", "open", "close"):
                    raise AudioError("unknown_command", cmd)
                if cmd == "close":
                    self.op_stop.set()                                 # closing while something plays: fade it out first
                elif self.busy and self.op_name in self.YIELDING:
                    self.op_stop.set()                                 # the helper ends; this command is queued below
                                                                       # and runs as soon as its op_done is announced
                elif self.busy:
                    raise AudioError("busy", self.op_name or "")
                if cmd in self.OPS:
                    if self.backend is None:
                        raise AudioError("not_open")
                    self.op_stop = threading.Event()
                    self.emit({"id": mid, "ok": True, "started": cmd})
                self.busy = True
                self.commands.put((mid, cmd, msg))
            except AudioError as e:
                self.emit({"id": mid, "ok": False, "error": e.key, "detail": e.detail})
            except Exception as e:
                self.emit({"id": mid, "ok": False, "error": "internal", "detail": f"{type(e).__name__}: {e}"})
        # stdin closed (server gone) or quit: fade out whatever is playing, then let the main loop end
        self.op_stop.set()
        self.commands.put(None)

    def serve(self) -> None:
        threading.Thread(target=self.reporter, daemon=True).start()
        threading.Thread(target=self.watchdog, daemon=True).start()
        threading.Thread(target=self.read_stdin, daemon=True).start()
        while True:
            item = self.commands.get()
            if item is None:
                break
            mid, cmd, params = item
            if cmd in self.OPS:
                self.run_op(cmd, params)                         # clears `busy` itself, before its op_done
                continue
            try:
                response = {"id": mid, "ok": True, **self.heavy(cmd, params)}
            except AudioError as e:
                response = {"id": mid, "ok": False, "error": e.key, "detail": e.detail}
            except Exception as e:
                response = {"id": mid, "ok": False, "error": "internal", "detail": f"{type(e).__name__}: {e}"}
            self.busy = False
            self.emit(response)
        self.quit.set()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    Engine().serve()


if __name__ == "__main__":
    main()
