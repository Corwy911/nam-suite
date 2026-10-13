"""Server side of the capture engine: starts the engine process and talks to it. Independent of FastAPI."""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

ENGINE = Path(__file__).with_name("engine.py")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class EngineError(Exception):
    """An error reported by the engine (or by the transport). `key` is an i18n key suffix: cap.err.<key>."""

    def __init__(self, key: str, detail: str = ""):
        super().__init__(f"{key}: {detail}" if detail else key)
        self.key = key
        self.detail = detail


class EngineSession:
    def __init__(self, work_dir: Path, python: Optional[str] = None):
        self.work_dir = Path(work_dir)
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._log = None
        self.opened: Optional[dict] = None       # device / channels / info of the open device
        self.state: dict = {}                    # last "state" event
        self.state_at = 0.0
        self.op: Optional[str] = None            # operation running now
        self.op_result: Optional[dict] = None    # last op_done event
        self._op_done = threading.Event()
        self.progress = deque(maxlen=30)
        self.last_error: Optional[dict] = None

    # ------------------------------------------------------------------ process
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_started(self) -> None:
        with self._lock:
            if self.alive:
                return
            self.work_dir.mkdir(parents=True, exist_ok=True)
            if self._log is None or self._log.closed:
                self._log = open(self.work_dir / "capture-engine.log", "ab")
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                   "TEMP": str(self.work_dir), "TMP": str(self.work_dir)}
            if sys.platform == "win32":
                env["SD_ENABLE_ASIO"] = "1"
            self.proc = subprocess.Popen([self.python, "-u", str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self._log, env=env, text=True, encoding="utf-8", bufsize=1,
                                         creationflags=_NO_WINDOW)
            self.opened, self.op, self.state, self.op_result, self.last_error = None, None, {}, None, None
            threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="engine-reader").start()
            threading.Thread(target=self._heartbeat, args=(self.proc,), daemon=True, name="engine-heartbeat").start()
        self.call("hello", timeout=20)

    def _heartbeat(self, proc) -> None:
        while proc.poll() is None:
            try:
                proc.stdin.write('{"cmd": "hb"}\n')
                proc.stdin.flush()
            except (OSError, ValueError):
                return
            time.sleep(2.0)

    def _read(self, proc) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and msg["id"] is not None:
                waiter = self._pending.get(msg["id"])
                if waiter:
                    waiter["msg"] = msg
                    waiter["event"].set()
                continue
            kind = msg.get("event")
            if kind == "state":
                self.state, self.state_at = msg, time.time()
            elif kind == "progress":
                self.progress.append(msg)
            elif kind == "op_done":
                if msg.get("op") != self.op:
                    continue       # a helper the engine stopped for a newer command: the session is already waiting
                                   # on that newer op, and ITS op_done must be the one the waiter gets
                self.op, self.op_result = None, msg
                if not msg.get("ok"):
                    self.last_error = {"key": msg.get("error"), "detail": msg.get("detail")}
                self._op_done.set()
        # the engine is gone
        if self.proc is proc:
            self.last_error = self.last_error or {"key": "engine_exited", "detail": f"code {proc.poll()}"}
            self.opened, self.op = None, None
            self._op_done.set()
        for waiter in list(self._pending.values()):
            waiter["msg"] = {"ok": False, "error": "engine_exited", "detail": ""}
            waiter["event"].set()

    # ------------------------------------------------------------------ commands
    def call(self, cmd: str, timeout: float = 15.0, **params) -> dict:
        if not self.alive:
            raise EngineError("engine_exited")
        with self._lock:
            mid = self._next_id
            self._next_id += 1
        waiter = {"event": threading.Event(), "msg": None}
        self._pending[mid] = waiter
        try:
            try:
                self.proc.stdin.write(json.dumps({"id": mid, "cmd": cmd, **params}) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                raise EngineError("engine_exited")
            if not waiter["event"].wait(timeout):
                raise EngineError("timeout", cmd)
            msg = waiter["msg"]
        finally:
            self._pending.pop(mid, None)
        if not msg.get("ok"):
            raise EngineError(msg.get("error") or "internal", msg.get("detail") or "")
        return msg

    def open(self, device: str, in_channels, out_channels, rate: int = 48_000, blocksize: int = 0, probe: bool = True) -> dict:
        info = self.call("open", timeout=30, device=device, in_channels=list(in_channels), out_channels=list(out_channels),
                         rate=rate, blocksize=blocksize, probe=probe)
        self.opened = {"device": device, "in_channels": list(in_channels), "out_channels": list(out_channels),
                       "device_name": info.get("device_name") or device,
                       "info": {k: info[k] for k in ("rate", "blocksize", "latency_in", "latency_out") if k in info}}
        return self.opened

    def close_device(self) -> None:
        if self.alive:
            try:
                self.call("close", timeout=10)
            except EngineError:
                pass
        self.opened, self.op = None, None

    def start_op(self, cmd: str, **params) -> None:
        """Start an operation; its outcome arrives as an op_done event (see wait_op / snapshot)."""
        prev = self.op
        self._op_done.clear()
        self.op_result = None
        self.progress.clear()
        self.op = cmd                       # before the call: a very short operation may finish before call() returns
        try:
            self.call(cmd, **params)
        except EngineError:
            self.op = prev                  # the new op never started: the engine is still busy with the previous
                                            # one, whose op_done must still be attributed (not dropped as stale)
            raise

    def wait_op(self, timeout: float = 60.0) -> dict:
        if not self._op_done.wait(timeout):
            raise EngineError("timeout", "operation")
        result = self.op_result
        if result is None:
            raise EngineError("engine_exited")
        if not result.get("ok"):
            raise EngineError(result.get("error") or "internal", result.get("detail") or "")
        return result

    def stop_op(self) -> None:
        if self.alive:
            try:
                self.call("stop", timeout=5)
            except EngineError:
                pass

    def shutdown(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write('{"cmd": "quit"}\n')
                proc.stdin.flush()
                proc.wait(3)
            except Exception:
                proc.kill()
        self.opened, self.op = None, None

    def snapshot(self) -> dict:
        fresh = self.state if time.time() - self.state_at < 1.5 else {}
        return {"alive": self.alive, "opened": self.opened, "op": self.op, "state": fresh, "progress": list(self.progress),
                "result": self.op_result, "error": self.last_error}
