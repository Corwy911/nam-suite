"""Smoke tests of the trainer's web program: where it keeps its files, which accelerator it reports, the routes NAM Capture uses.

    python -m unittest discover -s trainer/tests        (run with the Python that has the trainer's packages installed)

Each check runs in a fresh Python process with NAM_DATA_DIR pointing at a temporary folder, so a test can never touch real models,
WAV files or settings, whatever the machine.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "nam-web"


def run_py(code: str, data_dir: str = "") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "NAM_DATA_DIR"}
    env.update(PYTHONPATH=str(WEB), PYTHONUTF8="1")
    if data_dir:
        env["NAM_DATA_DIR"] = data_dir
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(WEB), timeout=300)


class PathsTest(unittest.TestCase):
    def test_default_layout_is_next_to_the_program(self):
        out = run_py("import paths, json; print(json.dumps([str(paths.LIB_DIR), str(paths.WAV_DIR), str(paths.SETTINGS_FILE), str(paths.TRAINERS_DIR), paths.CUSTOM]))")
        lib, wav, settings, trainers, custom = json.loads(out.stdout)
        root = WEB.parent
        self.assertEqual((Path(lib), Path(wav), Path(settings), Path(trainers), custom),
                         (root / "NAM Generati", root / "File WAV caricati", WEB / "settings.json", root / "trainers", False))

    def test_custom_data_dir_moves_everything_that_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run_py("import paths, json; paths.prepare(); print(json.dumps([str(paths.LIB_DIR), str(paths.TMP_DIR), str(paths.SETTINGS_FILE), str(paths.TRAINERS_DIR)]))", tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            lib, tmpdir, settings, trainers = (Path(x) for x in json.loads(out.stdout))
            data = Path(tmp)
            for item in (lib, tmpdir, settings.parent, trainers):
                self.assertIn(data, item.parents, item)
            self.assertTrue(lib.is_dir() and tmpdir.is_dir())
            self.assertTrue((trainers / "official").is_dir(), "the bundled trainers are copied on first run")


class HardwareTest(unittest.TestCase):
    def info(self, hardware_json: dict):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "state").mkdir()
            (Path(tmp) / "state" / "hardware.json").write_text(json.dumps(hardware_json), encoding="utf-8")
            out = run_py("import hardware, json; print(json.dumps([hardware.variant(), hardware.gpu_info()]))", tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            return json.loads(out.stdout)

    def test_apple_gpu(self):
        self.assertEqual(self.info({"variant": "mps", "gpu_name": "Apple M9 Max"}), ["mps", {"kind": "mps", "name": "Apple M9 Max"}])

    def test_amd_and_cpu(self):
        self.assertEqual(self.info({"variant": "amd", "gpu_name": "AMD Radeon RX 9999"}), ["amd", {"kind": "amd", "name": "AMD Radeon RX 9999"}])
        variant, chip = self.info({"variant": "cpu", "cpu_name": "Test CPU"})
        self.assertEqual((variant, chip), ("cpu", {"kind": "cpu", "name": "Test CPU"}))


class PriorityTest(unittest.TestCase):
    """The training worker's boost (priority.py): checked in a child process, so the test run itself keeps its own priority."""

    CODE = "import priority, json, sys; print('RESULT' + json.dumps({'line': priority.boost_process()}))"

    def line(self, extra_env=None):
        env = {**os.environ, "PYTHONPATH": str(WEB), "PYTHONUTF8": "1", **(extra_env or {})}
        env.pop("NAM_DATA_DIR", None)
        out = subprocess.run([sys.executable, "-c", self.CODE], capture_output=True, text=True, env=env, cwd=str(WEB), timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        return json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))[6:])["line"]

    def test_the_boost_can_be_turned_off_and_says_so(self):
        self.assertIn("off", self.line({"NAM_NO_BOOST": "1"}))

    @unittest.skipUnless(sys.platform == "win32", "Windows scheduling classes")
    def test_windows_gets_high_priority_and_no_power_throttling(self):
        line = self.line()
        self.assertIn("HIGH priority", line)
        self.assertIn("no power throttling", line)
        self.assertNotIn("REALTIME", line)                                   # never: it can starve the audio drivers and the input devices

    @unittest.skipUnless(sys.platform == "win32", "Windows scheduling classes")
    def test_the_priority_class_really_changes_in_the_child(self):
        code = ("import priority, ctypes; k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p; "
                "before = k.GetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess())); priority.boost_process(); "
                "after = k.GetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess())); print('RESULT', hex(before), hex(after))")
        env = {**os.environ, "PYTHONPATH": str(WEB)}
        env.pop("NAM_NO_BOOST", None)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(WEB), timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue(out.stdout.strip().endswith("0x80"), out.stdout)     # HIGH_PRIORITY_CLASS


APP_CHECK = r'''
import base64, io, json
import numpy as np, soundfile as sf
from fastapi.testclient import TestClient
import app, paths

buf = io.BytesIO()
sf.write(buf, np.zeros(4800, dtype="float32"), 48000, subtype="PCM_24", format="WAV")
body = {"name": "t", "input_rel": "Input/input.wav", "wav_name": "t.wav", "report": {"ok": True},
        "metadata": {"input_level_dbu": 5.8, "output_level_dbu": 12.6, "gear_type": "pedal"},
        "wav_base64": base64.b64encode(buf.getvalue()).decode()}
with TestClient(app.app) as c:
    out = {"ping": c.get("/nam-trainer/ping").json()}
    r = c.post("/nam-trainer/import", json=body)
    out["import"] = [r.status_code, r.json()]
    out["bad_base64"] = c.post("/nam-trainer/import", json={"wav_base64": "!!"}).status_code
    out["wavs"] = c.get("/api/wavs").json()["files"]
    out["sidecar"] = (paths.WAV_DIR / "Da NAM Capture" / "t.json").is_file()
print("RESULT" + json.dumps(out))
'''


class AppTest(unittest.TestCase):
    def test_capture_import_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run_py(APP_CHECK, tmp)
            self.assertEqual(out.returncode, 0, out.stdout[-1500:] + out.stderr[-1500:])
            result = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))[6:])
            self.assertEqual(result["ping"], {"ok": True})
            self.assertEqual(result["import"], [200, {"received": "Da NAM Capture/t.wav", "input_rel": "Input/input.wav", "input_needed": False}])
            self.assertEqual(result["bad_base64"], 400)
            self.assertTrue(result["sidecar"])
            self.assertEqual([f["capture_levels"] for f in result["wavs"]], [{"input_level_dbu": 5.8, "output_level_dbu": 12.6}])
            self.assertFalse((WEB.parent / "File WAV caricati" / "Da NAM Capture" / "t.wav").exists(), "nothing may be written next to the program")


INPUT_CHECK = r'''
import base64, hashlib, io, json
import numpy as np, soundfile as sf
from fastapi.testclient import TestClient
import app, paths

def wav_bytes(n):
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.1, 0.1, n).astype("float32"), 48000, subtype="PCM_24", format="WAV")
    return buf.getvalue()

capture, source = wav_bytes(4800), wav_bytes(9600)
md5 = hashlib.md5(source).hexdigest()
b64 = lambda b: base64.b64encode(b).decode()
body = {"name": "Amp", "wav_name": "Amp.wav", "report": {"ok": False, "replicate_esr": 0.0132, "warnings": ["esr_high"]},
        "metadata": {"input_level_dbu": 12.09}, "input_md5": md5, "input_size": len(source), "wav_base64": b64(capture)}
out = {}
with TestClient(app.app) as c:
    out["empty_list"] = c.get("/api/captures").json()
    first = c.post("/nam-trainer/import", json=body).json()
    out["first"] = first
    out["wrong_md5"] = c.post("/nam-trainer/import-input", json={"capture_rel": first["received"], "md5": "0" * 32, "wav_base64": b64(source)}).status_code
    out["unknown_capture"] = c.post("/nam-trainer/import-input", json={"capture_rel": "nope.wav", "md5": md5, "wav_base64": b64(source)}).status_code
    out["input"] = c.post("/nam-trainer/import-input", json={"capture_rel": first["received"], "name": "input.wav", "md5": md5, "wav_base64": b64(source)}).json()
    out["sidecar_input"] = json.loads((paths.WAV_DIR / "Da NAM Capture" / "Amp.json").read_text(encoding="utf-8"))["input_rel"]
    second = c.post("/nam-trainer/import", json={**body, "name": "Amp 2", "wav_name": "Amp 2.wav"}).json()     # same input again: not asked for twice
    out["second"] = second
    out["inputs_on_disk"] = sorted(p.name for p in (paths.WAV_DIR / "Input").iterdir())
    out["old_style"] = c.post("/nam-trainer/import", json={"name": "Old", "wav_name": "Old.wav", "input_rel": "Input/x.wav", "wav_base64": b64(capture)}).json()
    out["list"] = [[i["name"], i["from_capture"], i["input_rel"], i["esr"], i["warnings"], i["levels"], i["runs"]] for i in c.get("/api/captures").json()]
    out["del_outside"] = c.delete("/api/captures", params={"rel": "../x.wav"}).status_code
    out["del_missing"] = c.delete("/api/captures", params={"rel": "Da NAM Capture/none.wav"}).status_code
    out["del_ok"] = c.delete("/api/captures", params={"rel": "Da NAM Capture/Old.wav"}).json()
    out["after_delete"] = sorted(i["name"] for i in c.get("/api/captures").json())
    out["old_files_left"] = sorted(p.name for p in (paths.WAV_DIR / "Da NAM Capture").glob("Old.*"))
    out["address"] = c.get("/api/settings").json()
    out["cache"] = c.get("/").headers.get("cache-control")
    out["reveal_remote"] = c.post("/api/captures/reveal").status_code       # TestClient's host is not this PC: refused, no window opens
    out["local_flag"] = c.get("/api/settings").json()["local_client"]
print("RESULT" + json.dumps(out))
'''


class CaptureInputTest(unittest.TestCase):
    def test_input_travels_once_and_the_list_shows_what_arrived(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run_py(INPUT_CHECK, tmp)
            self.assertEqual(out.returncode, 0, out.stdout[-1500:] + out.stderr[-1500:])
            r = json.loads(next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))[6:])
            self.assertEqual(r["empty_list"], [])
            self.assertEqual(r["first"], {"received": "Da NAM Capture/Amp.wav", "input_rel": "", "input_needed": True})
            self.assertEqual((r["wrong_md5"], r["unknown_capture"]), (400, 404))
            self.assertEqual(r["input"], {"input_rel": "Input/input.wav"})
            self.assertEqual(r["sidecar_input"], "Input/input.wav")
            self.assertEqual(r["second"]["input_rel"], "Input/input.wav")
            self.assertFalse(r["second"]["input_needed"])
            self.assertEqual(r["inputs_on_disk"], ["input.wav"], "the same input is stored once")
            self.assertEqual(r["old_style"]["input_needed"], False, "a capture app that does not send the checksum is still accepted")
            by_name = {row[0]: row for row in r["list"]}
            self.assertEqual(set(by_name), {"Amp", "Amp 2", "Old"})
            self.assertEqual(by_name["Amp"][1:], [True, "Input/input.wav", 0.0132, ["esr_high"], {"input_level_dbu": 12.09}, []])
            self.assertIn(r["del_outside"], (404, 422), "a path that escapes the library is refused")
            self.assertEqual(r["del_missing"], 404)
            self.assertEqual(r["del_ok"], {"ok": True})
            self.assertEqual(r["after_delete"], ["Amp", "Amp 2"])
            self.assertEqual(r["old_files_left"], [], "the wav and its .json go together")
            self.assertEqual(len(r["address"]["address"].split(".")) in (4, 1), True)
            self.assertEqual(r["cache"], "no-cache", "after an update the browser must not keep showing the old page")
            self.assertEqual(r["reveal_remote"], 403, "opening a folder is for a page on the trainer's own PC only")
            self.assertFalse(r["local_flag"])
            if r["address"]["addresses"]:
                self.assertIn(r["address"]["address"], r["address"]["addresses"], "one address, taken from the PC's own list")


if __name__ == "__main__":
    unittest.main()
