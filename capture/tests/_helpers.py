"""Shared bits for the capture tests: import path and a synthetic v3-like take."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "capture_tool"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from capture import alignment as al  # noqa: E402


def write_fake_input(path, n: int = al.INPUT_V3_FRAMES, seed: int = 5) -> None:
    """A stand-in for the v3 input with the same layout (validation noise at both ends, two blips at 10.5 s / 11.5 s, silence before).
    Not the real NAM file (which is not in the repository); enough for the engine and the alignment."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)
    validation = (rng.standard_normal(al.T_VALIDATE) * 0.05).astype(np.float32)
    x[:al.T_VALIDATE] = validation
    x[n - al.T_VALIDATE:] = validation
    for start in al.BLIP_LOCATIONS:
        x[start] = 0.99
    x[600_000:600_000 + 200_000] = (rng.standard_normal(200_000) * 0.1).astype(np.float32)     # some "training" signal
    sf.write(str(path), x, al.RATE, subtype="PCM_24")


def make_take(delay: int = 173, gain: float = 0.3, noise_db: float = -72.0, n: int = al.INPUT_V3_FRAMES,
              extra: int = 48_000, seed: int = 1, blips: bool = True, clip: bool = False) -> np.ndarray:
    """What the interface records while the v3 input plays: the response to the two blips 10.5 s and 11.5 s in, and the same
    validation signal at the start and at the end (so the replicate ESR is meaningful), all `delay` samples late."""
    rng = np.random.default_rng(seed)
    total = n + extra
    y = np.zeros(total)
    validation = rng.standard_normal(al.T_VALIDATE) * 0.05
    y[delay:delay + al.T_VALIDATE] = validation
    y[delay + n - al.T_VALIDATE:delay + n] = validation
    if blips:
        ring = np.exp(-np.arange(400) / 60.0) * np.cos(np.arange(400) * 0.7)
        for start in al.BLIP_LOCATIONS:
            y[delay + start:delay + start + 400] += gain * ring
    y += rng.standard_normal(total) * 10 ** (noise_db / 20)
    if clip:
        y[delay + 1_000_000:delay + 1_000_005] = 1.0        # a flat top, like a converter that hit full scale
    return y[:total]


# The real ports are 8765 (NAM Trainer) and 8767 (the native engine of NAM Capture), and the real programs may well be running on the machine
# that runs these tests: the tests must never talk to them (a test capture would land in the trainer's library, a stop would cut a session
# that somebody is listening to). Every test that reaches either of them pins its port to these first.
TRAINER_TEST_PORT = 18765
NATIVE_TEST_PORT = 18767


def pin_trainer_port():
    import capture_api.core
    import capture_api.trainer
    capture_api.trainer.TRAINER_PORT = TRAINER_TEST_PORT
    capture_api.core.NATIVE_PORT = NATIVE_TEST_PORT


class FakeNativeEngine:
    """A stand-in for packaging/ControlServer.swift on 127.0.0.1:NATIVE_TEST_PORT, as a context manager: records every POST in `.calls`
    as (path, payload). `replies` maps a path to a dict (or a function of the payload); any other POST answers {"ok": True}.
    `state` is what GET answers. Lets the relay's own logic be tested without the compiled Swift/C++ binary."""

    PORT = NATIVE_TEST_PORT

    def __init__(self, replies=None, state=None):
        import http.server
        import json
        import threading

        self.calls = []
        outer = self
        replies = replies or {}
        state = state or {"opened": True}

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.calls.append((self.path, payload))
                reply = replies.get(self.path, {"ok": True})
                self._reply(reply(payload) if callable(reply) else reply)

            def do_GET(self):
                # the relay sends a JSON body even with GET: read it, or closing with unread data resets the connection on Windows
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._reply(state)

            def log_message(self, *a):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", self.PORT), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class FakeTrainer:
    """A stand-in for the trainer's own tiny HTTP side (/nam-trainer/ping, /nam-trainer/import, /nam-trainer/import-input) on
    127.0.0.1:TRAINER_TEST_PORT: GET answers `ping_status` (the health check), POST records the whole JSON body in `.imports`
    (or `.inputs`) and answers 200. Simply don't use this context manager at all to simulate the trainer being off (a real
    connection refusal, not a fake reply)."""

    PORT = TRAINER_TEST_PORT

    def __init__(self, ping_status=200, needs_input=None, input_status=200):
        """needs_input None: an old trainer (empty answer); True/False: the new one, which says whether it wants the input.wav too."""
        import http.server
        import json
        import threading

        self.imports = []
        self.inputs = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(ping_status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/nam-trainer/import-input":
                    outer.inputs.append(payload)
                    self.send_response(input_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.imports.append(payload)
                body = b"" if needs_input is None else json.dumps({"received": "Da NAM Capture/" + payload["wav_name"], "input_rel": "", "input_needed": needs_input}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", self.PORT), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
