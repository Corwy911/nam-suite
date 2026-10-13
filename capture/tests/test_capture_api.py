"""The capture API end to end with the simulated interface: calibrations, a full take, saving to the library.
The app is imported for its helpers; its folders are swapped for temporary ones so that nothing is written to the real library."""

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from _helpers import FakeNativeEngine, FakeTrainer, al, pin_trainer_port, write_fake_input


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import app as appmod
        import capture_api
        import settings
        from capture import store

        pin_trainer_port()
        cls.tmp = Path(tempfile.mkdtemp(prefix="nam-api-"))
        cls.wav_dir, cls.tmp_dir = cls.tmp / "wav", cls.tmp / "tmp"
        cls.model_dir, cls.ir_dir = cls.tmp / "Modelli NAM", cls.tmp / "IR"
        cls.usage_file = cls.tmp / "live_usage.json"
        cls.wav_dir.mkdir(); cls.tmp_dir.mkdir(); cls.model_dir.mkdir(); cls.ir_dir.mkdir()
        cls.saved = (appmod.WAV_DIR, appmod.TMP_DIR, store.CALIBRATION_FILE)
        appmod.WAV_DIR, appmod.TMP_DIR = cls.wav_dir, cls.tmp_dir
        store.CALIBRATION_FILE = cls.tmp / "calibration.json"
        cls.settings = settings
        cls.saved_settings_file = settings.SETTINGS_FILE
        settings.SETTINGS_FILE = cls.tmp / "settings.json"      # never touch the real one
        capture_api.core._session = None
        capture_api.core.st.take = None
        capture_api.configure(wav_dir=cls.wav_dir, tmp_dir=cls.tmp_dir, model_dir=cls.model_dir, ir_dir=cls.ir_dir,
                              usage_file=cls.usage_file, safe_name=appmod.safe_name, safe_subfolder=appmod.safe_subfolder,
                              unique_file=appmod.unique_file, wav_rel=appmod.wav_rel, wav_from_ref=appmod.wav_from_ref,
                              store_wav=appmod.store_wav, save_upload=appmod.save_upload)
        cls.appmod, cls.capture_api, cls.store = appmod, capture_api, store
        cls.local = TestClient(appmod.app, client=("127.0.0.1", 50000))
        cls.remote = TestClient(appmod.app)                     # host "testclient": not this PC
        # a stand-in for the v3 input, recognised by the md5 we give the alignment module
        fake = cls.wav_dir / "Test" / "input.wav"
        fake.parent.mkdir()
        write_fake_input(fake)
        cls.saved_md5 = al.INPUT_V3_MD5
        al.INPUT_V3_MD5 = hashlib.md5(fake.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        import shutil
        al.INPUT_V3_MD5 = cls.saved_md5
        cls.capture_api.shutdown()
        cls.appmod.WAV_DIR, cls.appmod.TMP_DIR, cls.store.CALIBRATION_FILE = cls.saved
        cls.settings.SETTINGS_FILE = cls.saved_settings_file
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def post(self, path, json=None, expect=200):
        r = self.local.post("/api/capture" + path, json=json or {})
        self.assertEqual(r.status_code, expect, r.text)
        return r.json()

    def get(self, path, expect=200):
        r = self.local.get("/api/capture" + path)
        self.assertEqual(r.status_code, expect, r.text)
        return r.json()

    def wait_take(self, status, timeout=180):
        end = time.time() + timeout
        while time.time() < end:
            take = self.get("/state")["take"]
            if take and take["status"] in (status, "error", "aborted"):
                return take
            time.sleep(0.3)
        self.fail("take did not finish")

    # ------------------------------------------------------------------------------------ tests
    def test_01_only_this_pc(self):
        for method, path in (("get", "/api/capture/state"), ("get", "/api/capture/devices"), ("post", "/api/capture/tone/start"),
                             ("post", "/api/capture/record/start"), ("get", "/api/capture/input")):
            r = getattr(self.remote, method)(path)
            self.assertEqual(r.status_code, 403, path)

    def test_02_full_flow_with_the_demo_interface(self):
        devices = self.get("/devices")["devices"]
        self.assertIn("sim:demo", [d["id"] for d in devices])
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/demo/config", {"speed": 0.0, "co_dbu": 14.3, "ci_dbu": 9.2, "latency": 1056, "route": "loopback"})
        state = self.get("/state")
        self.assertTrue(state["demo"])
        self.assertIsNone(state["calibrations"]["out"])

        # refusals before anything is calibrated
        self.post("/tone/start", {"level_dbfs": 0.0}, expect=400)                       # no confirmation
        self.post("/tone/start", {"level_dbfs": 3.0, "confirm": True}, expect=400)      # above full scale
        self.post("/calibrate/output", {"vrms": "abc"}, expect=400)
        self.post("/calibrate/input", {}, expect=400)                                   # no confirmation
        self.post("/calibrate/input", {"confirm": True}, expect=409)                    # output first
        self.post("/calibrate/input", {"confirm": True, "level_dbfs": 3.0}, expect=400)  # above full scale
        self.post("/tone/level", {"level_dbfs": -20.0}, expect=400)                     # no tone is playing
        self.post("/record/start", {"gear": "head", "input_ref": "Test/input.wav", "calibrated": True}, expect=409)

        # output level from the (simulated) multimeter, at -12 dBFS to prove the level sent is accounted for
        vrms = self.get("/demo/multimeter?level_dbfs=-12")["vrms"]
        out = self.post("/calibrate/output", {"vrms": f"{vrms:.4f}", "sent_dbfs": -12.0})["entry"]
        self.assertAlmostEqual(out["co_dbu"], 14.3, delta=0.01)
        self.assertEqual(out["device"], "Demo interface (simulated)")                     # filed under the name, never under the list index

        # the tone can be started and stopped
        self.post("/tone/start", {"level_dbfs": -40.0, "confirm": True})
        time.sleep(0.3)
        self.assertAlmostEqual(self.post("/tone/level", {"level_dbfs": -38.5})["level_dbfs"], -38.5)   # the knob, while it plays
        self.post("/tone/level", {"level_dbfs": 1.0}, expect=400)
        self.post("/stop")
        time.sleep(0.5)

        # input level by loopback + verification + interface latency
        cal_in = self.post("/calibrate/input", {"confirm": True})
        self.assertAlmostEqual(cal_in["entry"]["ci_dbu"], 9.2, delta=0.05)
        self.assertEqual(cal_in["entry"]["judge"], "ok")
        self.assertFalse(cal_in["entry"]["manual"])

        # the same input at a software level chosen by hand: the send level is accounted for, so Ci does not move
        for level in (-20.0, -14.5, -9.5):
            manual = self.post("/calibrate/input", {"confirm": True, "level_dbfs": level})
            self.assertTrue(manual["entry"]["manual"])
            self.assertEqual(manual["entry"]["level_dbfs"], level)
            self.assertAlmostEqual(manual["entry"]["ci_dbu"], 9.2, delta=0.05)
        # a level that clips the input is not stored, and the previous calibration stays
        clipped = self.post("/calibrate/input", {"confirm": True, "level_dbfs": 0.0})
        self.assertEqual(clipped["rejected"], "clip")
        self.assertIsNone(clipped["entry"])
        self.assertEqual(self.get("/state")["calibrations"]["in"]["level_dbfs"], -9.5)
        cal_in = self.post("/calibrate/input", {"confirm": True})                          # back to the automatic search
        ver = self.post("/verify/input")
        self.assertLess(abs(ver["deviation_db"]), 0.05)
        self.post("/demo/config", {"ci_dbu": 12.0})                                       # somebody turned the gain
        self.assertGreater(abs(self.post("/verify/input")["deviation_db"]), 2.0)
        self.post("/demo/config", {"ci_dbu": 9.2})
        self.assertEqual(self.post("/latency-probe")["delay"], 1056)

        # optional linearity check of the input: needs confirmation, is stored, and goes stale when the input is calibrated again
        self.post("/linearity", {}, expect=400)
        lin = self.post("/linearity", {"confirm": True})["entry"]
        self.assertEqual(lin["status"], "ok")
        self.assertLessEqual(lin["max_dev_db"], 0.1)
        self.assertGreaterEqual(len(lin["points"]), 4)
        self.assertFalse(self.get("/state")["calibrations"]["lin"]["stale"])
        self.post("/calibrate/input", {"confirm": True, "level_dbfs": -20.0})
        self.assertTrue(self.get("/state")["calibrations"]["lin"]["stale"])
        self.post("/calibrate/input", {"confirm": True})

        # level test with the gear in the loop: refused without confirmation, judged on the sample peak
        self.post("/level-test", {"send_gain_db": 0.0}, expect=400)
        self.post("/level-test", {"send_gain_db": 3.0, "confirm": True}, expect=400)
        ok = self.post("/level-test", {"send_gain_db": -12.0, "confirm": True})
        self.assertIn(ok["judge"], ("ok", "hot"))
        self.assertEqual(ok["clipped"], 0)
        self.post("/demo/config", {"ci_dbu": -6.0})                                       # a very sensitive input: 0 dBFS now clips
        hot = self.post("/level-test", {"send_gain_db": 0.0, "confirm": True})
        self.assertEqual(hot["judge"], "clip")
        self.assertLessEqual(hot["suggest_trim_db"], -6.0)
        self.post("/demo/config", {"ci_dbu": 9.2})

        # the input file must be the standard v3
        found = self.get("/input")["found"]
        self.assertEqual([f["rel"] for f in found], ["Test/input.wav"])

        # a pedal capture, calibrated: both levels go into the metadata
        self.post("/demo/config", {"route": "gear", "gear": "pedal"})
        started = self.post("/record/start", {"gear": "pedal", "name": "My Pedal - clean", "input_ref": "Test/input.wav", "calibrated": True, "send_gain_db": -3.0})
        self.assertAlmostEqual(started["metadata"]["input_level_dbu"], 14.3 - 3.0, delta=0.01)
        self.assertAlmostEqual(started["metadata"]["output_level_dbu"], 9.2, delta=0.05)
        self.assertEqual(started["metadata"]["gear_type"], "pedal")
        self.post("/record/start", {"gear": "pedal", "input_ref": "Test/input.wav"}, expect=409)      # already running
        take = self.wait_take("ready")
        self.assertEqual(take["status"], "ready", take)
        self.assertEqual(take["report"]["latency"]["delay"], 1056)
        self.assertTrue(take["report"]["ok"], take["report"]["warnings"])
        hand = self.post("/take/accept", {})
        out_path = self.wav_dir / hand["output_rel"]
        info = sf.info(str(out_path))
        self.assertEqual((info.frames, info.channels, info.samplerate, info.subtype), (al.INPUT_V3_FRAMES, 1, 48000, "PCM_24"))
        self.assertEqual(hand["input_ref"] if "input_ref" in hand else hand["input_rel"], "Test/input.wav")
        self.assertTrue(hand["output_rel"].startswith("Catture/"))
        # one folder per capture: the WAV, its sidecar, the raw recording and the whole record of how it was made
        folder = out_path.parent
        self.assertEqual(hand["folder_rel"], "Catture/" + folder.name)
        self.assertTrue((folder / "_originale" / "raw-recording.wav").is_file())
        record = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["files"]["output"], hand["output_rel"])
        self.assertEqual(record["gear"], "pedal")
        self.assertAlmostEqual(record["calibration"]["out"]["co_dbu"], 14.3, delta=0.01)
        self.assertEqual(record["input"]["ref"], "Test/input.wav")
        self.assertEqual(record["input"]["md5"], al.INPUT_V3_MD5)
        self.assertEqual(hand["name"], "My Pedal - clean")
        side = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))       # the levels travel with the WAV
        self.assertEqual(side["metadata"], hand["metadata"])
        self.assertAlmostEqual(side["metadata"]["input_level_dbu"], 14.3 - 3.0, delta=0.01)
        self.assertEqual(side["metadata"]["gear_type"], "pedal")
        self.assertIsNone(self.get("/state")["take"])
        # what is in the file is the response aligned to the input: the first blip is where the trainer expects it (+1 sample)
        y, _ = sf.read(str(out_path), dtype="float64")
        first = int(np.argmax(np.abs(y[al.BLIP_LOCATIONS[0] - 50:al.BLIP_LOCATIONS[0] + 50]))) + al.BLIP_LOCATIONS[0] - 50
        self.assertEqual(first, al.BLIP_LOCATIONS[0] + 1)

        # a head, uncalibrated capture: aligned and cut like the other, no levels in the metadata
        self.post("/demo/config", {"route": "gear", "gear": "head"})
        started = self.post("/record/start", {"gear": "head", "name": "Head", "input_ref": "Test/input.wav", "calibrated": False})
        self.assertIsNone(started["metadata"]["input_level_dbu"])
        self.assertIsNone(started["metadata"]["output_level_dbu"])
        self.wait_take("ready")
        self.post("/take/discard")
        self.assertIsNone(self.get("/state")["take"])
        head = [d for d in (self.wav_dir / "Catture").iterdir() if d.is_dir() and d.name.endswith("- Head")][0]
        self.assertEqual(json.loads((head / "capture.json").read_text(encoding="utf-8"))["status"], "discarded")
        self.assertTrue((head / "_originale" / "raw-recording.wav").is_file())      # a discarded take stays recoverable

    def test_03_wrong_input_file_is_refused(self):
        bad = self.wav_dir / "other.wav"
        sf.write(str(bad), np.zeros(48_000), 48_000, subtype="PCM_24")
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/record/start", {"gear": "head", "input_ref": "other.wav", "calibrated": False}, expect=400)

    def test_04_native_relay_is_unreachable_by_default(self):
        # with no native engine listening (this is a Terminal-run test, not the packaged app), the relay fails
        # cleanly and gives the device straight back to Python instead of leaving it owned by nobody
        (self.ir_dir / "Cab.wav").write_bytes(b"RIFF....")
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/native/live/start", {"confirm": True, "ir": "Cab.wav"}, expect=502)
        self.assertIsNotNone(self.get("/state")["engine"]["opened"])          # handed straight back, not left closed

    def test_06_native_live_files_lists_models_and_irs(self):
        (self.model_dir / "Marshall.nam").write_text("{}")
        (self.ir_dir / "4x12.wav").write_bytes(b"RIFF....")
        (self.ir_dir / "notes.txt").write_text("not an IR")   # ignored: wrong extension
        found = self.get("/native/live/files")
        names = lambda entries: [e["name"] for e in entries]
        self.assertIn("Marshall.nam", names(found["models"]))
        self.assertIn("4x12.wav", names(found["irs"]))
        self.assertNotIn("notes.txt", names(found["irs"]))
        self.assertEqual(next(e for e in found["models"] if e["name"] == "Marshall.nam")["uses"], 0)

    def test_07_native_live_start_rejects_path_traversal_and_missing_choice(self):
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/demo/config", {"co_dbu": 14.3, "ci_dbu": 9.2, "route": "loopback"})
        self.post("/calibrate/output", {"vrms": f"{self.get('/demo/multimeter?level_dbfs=0')['vrms']:.4f}", "sent_dbfs": 0.0})
        self.post("/calibrate/input", {"confirm": True})
        self.post("/native/live/start", {"confirm": True}, expect=400)   # neither model nor ir
        self.post("/native/live/start", {"confirm": True, "model": "../../etc/passwd"}, expect=400)
        self.post("/native/live/start", {"confirm": True, "model": "nope.nam"}, expect=404)   # not created above
        self.assertIsNotNone(self.get("/state")["engine"]["opened"])   # never handed the device away for a rejected request

    def test_08_native_live_start_with_a_fake_native_engine(self):
        # same stand-in technique as test_5: proves the gain math and the return-channel device handoff, not the
        # compiled Swift/C++ binary (that only gets verified live, with Roman, on the Fireface - see the plan)
        (self.model_dir / "Amp.nam").write_text("{}")
        (self.ir_dir / "Cab.wav").write_bytes(b"RIFF....")
        with FakeNativeEngine({"/open": {"ok": True, "blocksize": 32, "sample_rate": 48000}, "/model": {"ok": True, "input_level_dbu": 6.0, "output_level_dbu": 12.0}},
            state={"opened": True, "in_peak_dbfs": -10.0, "out_peak_dbfs": -8.0, "xruns": 0, "model_loaded": True, "ir_loaded": True,
                   "model_input_level_dbu": 6.0, "model_output_level_dbu": 12.0}) as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            before = self.get("/state")["engine"]["opened"]
            started = self.post("/native/live/start", {"confirm": True, "model": "Amp.nam", "ir": "Cab.wav"})
            self.assertAlmostEqual(started["pre_gain_db"], 9.2 - 6.0, delta=0.01)     # Ci - model_input_level_dbu
            self.assertAlmostEqual(started["gain_db"], 12.0 - 14.3, delta=0.01)       # model_output_level_dbu - Co
            self.assertIsNone(self.get("/state")["engine"]["opened"])
            self.assertEqual(calls[0][0], "/open")
            self.assertEqual([c[0] for c in calls if c[0] in ("/model", "/ir", "/pregain", "/gain")],
                             ["/model", "/ir", "/pregain", "/gain"])
            self.assertTrue(str(calls[1][1]["path"]).endswith("Amp.nam"))
            self.assertTrue(str(calls[2][1]["path"]).endswith("Cab.wav"))
            live_state = self.get("/native/monitor/state")   # the state proxy is generic, reused by both tools
            self.assertTrue(live_state["model_loaded"])
            found = self.get("/native/live/files")           # a saved (library) file's use gets counted
            self.assertEqual(next(e for e in found["models"] if e["name"] == "Amp.nam")["uses"], 1)
            self.assertEqual(next(e for e in found["irs"] if e["name"] == "Cab.wav")["uses"], 1)
            self.post("/native/live/stop")
            after = self.get("/state")["engine"]["opened"]
            self.assertEqual(after["device"], before["device"])   # Python has the device back
            self.assertEqual(calls[-1][0], "/stop")

    def test_09_native_live_start_ir_only_uses_return_channel_unity_gain(self):
        # "testata" mode: no model, so the fallback is the same G = Ci - Co already used by the guitar monitor -
        # applied here to whatever is open (the RETURN input), not the guitar
        (self.ir_dir / "HeadCab.wav").write_bytes(b"RIFF....")
        with FakeNativeEngine() as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            started = self.post("/native/live/start", {"confirm": True, "ir": "HeadCab.wav", "listen_out_channels": [2, 3]})
            self.assertAlmostEqual(started["pre_gain_db"], 0.0, delta=0.01)      # no model to feed
            self.assertAlmostEqual(started["gain_db"], 9.2 - 14.3, delta=0.01)   # Ci - Co, same formula as the guitar monitor
            self.assertEqual([c[0] for c in calls if c[0] in ("/model", "/ir")], ["/ir"])   # no /model call at all
            opened_call = next(c[1] for c in calls if c[0] == "/open")
            self.assertEqual((opened_call["out_channel"], opened_call["extra_out_channels"]), (2, [3]))   # heard on the monitor pair, not the reamp output
            self.post("/native/live/stop")

    def test_10_native_live_upload_import_saves_it_for_good(self):
        r = self.local.post("/api/capture/native/live/upload", data={"kind": "model", "do_import": "true"},
                            files={"file": ("My Amp.nam", b"{}", "application/octet-stream")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["imported"])
        self.assertEqual(body["ref"], "My Amp.nam")
        self.assertTrue((self.model_dir / "My Amp.nam").is_file())
        found = self.get("/native/live/files")
        self.assertIn("My Amp.nam", [e["name"] for e in found["models"]])

    def test_11_native_live_upload_without_import_is_a_one_off_and_gets_cleaned_up(self):
        # Roman: "non mi piace spostare file" - browsing without ticking "importa" must not touch Modelli NAM/IR
        r = self.local.post("/api/capture/native/live/upload", data={"kind": "ir", "do_import": "false"},
                            files={"file": ("Cab From Desktop.wav", b"RIFF....", "audio/wav")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["imported"])
        self.assertTrue(body["ref"].startswith("tmp:"))
        tmp_name = body["ref"][len("tmp:"):]
        tmp_path = self.tmp_dir / "live_uploads" / tmp_name
        self.assertTrue(tmp_path.is_file())
        self.assertFalse((self.ir_dir / "Cab From Desktop.wav").exists())   # never landed in the library

        with FakeNativeEngine() as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            self.post("/native/live/start", {"confirm": True, "ir": body["ref"]})
            self.assertTrue(tmp_path.exists())           # kept while the session runs: a later swap re-sends this ref
            found = self.get("/native/live/files")
            self.assertNotIn("tmp:" + tmp_name, [e["name"] for e in found["irs"]])   # never counted as a saved file
            self.post("/native/live/stop")
            self.assertTrue(tmp_path.exists())           # still there: a second Avvia with the same pick must work

    def test_12_native_live_upload_rejects_wrong_extension(self):
        r = self.local.post("/api/capture/native/live/upload", data={"kind": "model", "do_import": "false"},
                            files={"file": ("not-a-model.txt", b"hello", "text/plain")})
        self.assertEqual(r.status_code, 400, r.text)

    def test_13_native_live_files_delete_removes_it_and_its_usage_count(self):
        (self.model_dir / "ToDelete.nam").write_text("{}")
        self.get("/native/live/files")   # (no-op; just documents nothing needs to be open for this)
        r = self.local.delete("/api/capture/native/live/files?kind=model&name=ToDelete.nam")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["removed"])
        self.assertFalse((self.model_dir / "ToDelete.nam").exists())
        found = self.get("/native/live/files")
        self.assertNotIn("ToDelete.nam", [e["name"] for e in found["models"]])
        r = self.local.delete("/api/capture/native/live/files?kind=model&name=ToDelete.nam")   # already gone
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["removed"])

    def test_14_native_live_files_delete_rejects_path_traversal(self):
        r = self.local.delete("/api/capture/native/live/files?kind=model&name=../../etc/passwd")
        self.assertEqual(r.status_code, 400, r.text)

    def test_15_native_live_start_dual_drives_the_pedal_while_monitoring_nam_ir(self):
        # Roman: guitar -> unity gain -> real pedal -> return -> NAM -> IR -> a SEPARATE monitor output, all at
        # once, so correcting a wrong NAM choice (native_live_swap) never breaks the guitar->pedal loop
        (self.model_dir / "PedalAmp.nam").write_text("{}")
        # a second input's own calibration (channel 1 = "guitar"), distinct from channel 0's "return" cal from test_09
        self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})
        self.post("/demo/config", {"co_dbu": 14.3, "ci_dbu": 5.5, "route": "loopback"})
        self.post("/calibrate/input", {"confirm": True})
        self.post("/close")
        with FakeNativeEngine({"/open_dual": {"ok": True, "blocksize": 32, "sample_rate": 48000}, "/model": {"ok": True, "input_level_dbu": 6.0, "output_level_dbu": 12.0}}) as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [1, 0], "out_channels": [0]})   # guitar=1, return=0
            started = self.post("/native/live/start", {"confirm": True, "model": "PedalAmp.nam", "monitor_out_channels": [2, 3]})
            self.assertAlmostEqual(started["guitar_gain_db"], 5.5 - 14.3, delta=0.01)     # Ci_guitar - Co, same as the plain monitor
            self.assertAlmostEqual(started["pre_gain_db"], 9.2 - 6.0, delta=0.01)         # unaffected: still Ci_return-based (channel 0 from test_09)
            self.assertEqual(calls[0][0], "/open_dual")
            self.assertEqual(calls[0][1]["guitar_channel"], 1)
            self.assertEqual(calls[0][1]["return_channel"], 0)
            self.assertEqual(calls[0][1]["pedal_out_channel"], 0)
            self.assertEqual(calls[0][1]["monitor_out_channels"], [2, 3])
            self.assertNotIn("/open", [c[0] for c in calls])   # dual mode calls /open_dual, never plain /open
            # clearing both is still rejected here too (swap keeps the same "at least one" rule as start)
            self.post("/native/live/swap", {"confirm": True, "model": None, "ir": None}, expect=400)
            self.post("/native/live/stop")
            after = self.get("/state")["engine"]["opened"]
            self.assertEqual(after["in_channels"], [1, 0])   # Python got both inputs back, guitar first

    def test_16_native_live_start_dual_rejects_bad_monitor_channels_and_single_input(self):
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/native/live/start", {"confirm": True, "ir": "HeadCab.wav", "monitor_out_channels": [1, 2, 3]}, expect=400)
        self.post("/native/live/start", {"confirm": True, "ir": "HeadCab.wav", "monitor_out_channels": [2]}, expect=400)
        self.assertIsNotNone(self.get("/state")["engine"]["opened"])   # never touched the device for a rejected request

    def test_17_native_live_start_uncalibrated_skips_calibration_and_defaults_to_0db(self):
        # channel 0 on sim:demo IS calibrated (test_02) - calibrated=False must ignore that entirely, not just
        # "happen to be uncalibrated": always-startable 0 dB fallback
        (self.ir_dir / "Uncal.wav").write_bytes(b"RIFF....")
        with FakeNativeEngine() as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            started = self.post("/native/live/start", {"confirm": True, "ir": "Uncal.wav", "calibrated": False})
            self.assertEqual(started["pre_gain_db"], 0.0)
            self.assertEqual(started["gain_db"], 0.0)
            self.assertIsNone(started["co_dbu"])
            self.assertIsNone(started["ci_dbu"])
            self.assertEqual([c[1]["pre_gain_db"] for c in calls if c[0] == "/pregain"], [0.0])
            self.assertEqual([c[1]["gain_db"] for c in calls if c[0] == "/gain"], [-18.0])        # 0 dB + the listening volume (-18 dB by default)
            self.post("/native/live/stop")

    def test_18b_capture_and_level_test_refuse_an_interface_left_open_on_two_inputs(self):
        # what a stopped live session leaves behind is [guitar, return]: a take would silently read the FIRST one (the guitar)
        self.post("/open", {"device": "sim:demo", "in_channels": [1, 0], "out_channels": [0]})
        self.post("/level-test", {"confirm": True, "send_gain_db": -12.0}, expect=409)
        self.post("/record/start", {"gear": "head", "input_ref": "Test/input.wav", "calibrated": False}, expect=409)
        self.post("/close")
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})     # the one return input: fine again
        self.post("/level-test", {"confirm": True, "send_gain_db": -12.0})

    def test_18a_calibrated_ir_only_needs_no_return_calibration_but_the_guitar_leg_needs_its_own(self):
        (self.ir_dir / "NoRet.wav").write_bytes(b"RIFF....")
        (self.model_dir / "NeedsRet.nam").write_text("{}")
        with FakeNativeEngine({"/model": {"ok": True, "input_level_dbu": 6.0, "output_level_dbu": 12.0}}) as fake:
            # channel 5 has NO calibration at all (neither output nor input): calibrated IR-only must still start, at 0 dB
            self.post("/open", {"device": "sim:demo", "in_channels": [3], "out_channels": [3]})
            started = self.post("/native/live/start", {"confirm": True, "ir": "NoRet.wav"})
            self.assertEqual((started["pre_gain_db"], started["gain_db"]), (0.0, 0.0))
            self.post("/native/live/stop")
            # ...but a profile needs the return's calibration, and the guitar leg needs the output + guitar input ones
            self.post("/open", {"device": "sim:demo", "in_channels": [3], "out_channels": [3]})
            self.post("/native/live/start", {"confirm": True, "model": "NeedsRet.nam"}, expect=409)
            self.post("/close")
            self.post("/open", {"device": "sim:demo", "in_channels": [3, 3], "out_channels": [3]})
            self.post("/native/live/start", {"confirm": True, "ir": "NoRet.wav", "monitor_out_channels": [0, 1]}, expect=409)
            # with the output and the guitar input calibrated (channels 0) the head chain starts: IR alone, return uncalibrated
            self.post("/close")
            self.post("/open", {"device": "sim:demo", "in_channels": [0, 3], "out_channels": [0]})
            self.post("/native/live/start", {"confirm": True, "ir": "NoRet.wav", "monitor_out_channels": [0, 1]}, expect=400)   # the send's own channel is a monitor too
            started = self.post("/native/live/start", {"confirm": True, "ir": "NoRet.wav", "monitor_out_channels": [1, 2]})
            self.assertAlmostEqual(started["guitar_gain_db"], 9.2 - 14.3, delta=0.01)       # exact unity: Ci_guitar - Co
            self.assertEqual(started["gain_db"], 0.0)                                       # the uncalibrated return leg stays at 0 dB
            self.post("/native/live/stop")

    def test_18_native_live_controls_relays_manual_trims_and_needs_a_running_session(self):
        self.post("/native/live/controls", {"pre_gain_db": 1.0}, expect=409)   # nothing running yet
        (self.ir_dir / "Ctrl.wav").write_bytes(b"RIFF....")
        with FakeNativeEngine() as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            self.post("/native/live/start", {"confirm": True, "ir": "Ctrl.wav", "calibrated": False})
            calls.clear()
            self.post("/native/live/controls", {"pre_gain_db": 2.5, "gain_db": -1.0, "gate_threshold_db": -30.0,
                                                 "eq_bass_db": 3.0, "eq_treble_db": -2.0, "ir_in_db": 1.5, "ir_out_db": -3.0})
            self.assertEqual([c[0] for c in calls], ["/pregain", "/gain", "/gate", "/eq", "/ir_gain"])
            self.assertEqual(calls[0][1], {"pre_gain_db": 2.5})
            self.assertEqual(calls[1][1], {"gain_db": -19.0})                      # -1 dB + the listening volume (-18 dB by default)
            self.assertEqual(calls[2][1], {"threshold_db": -30.0})
            self.assertEqual(calls[3][1], {"bass_db": 3.0, "treble_db": -2.0})   # mid_db omitted: not sent at all
            self.assertEqual(calls[4][1], {"in_db": 1.5, "out_db": -3.0})
            self.post("/native/live/stop")

    def test_18b_the_listening_volume_rides_on_the_output_gain_and_starts_at_minus_18(self):
        (self.ir_dir / "Vol.wav").write_bytes(b"RIFF....")
        with FakeNativeEngine() as fake:
            calls = fake.calls
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            self.post("/native/live/start", {"confirm": True, "ir": "Vol.wav", "calibrated": False, "master_db": -12.0})
            self.assertEqual([c[1]["gain_db"] for c in calls if c[0] == "/gain"], [-12.0])
            calls.clear()
            self.post("/native/live/controls", {"master_db": -3.0})                    # the master alone resends the output gain
            self.assertEqual(calls, [("/gain", {"gain_db": -3.0})])
            calls.clear()
            self.post("/native/live/controls", {"gain_db": 2.0})                       # the Output knob keeps working, the master stays on top
            self.assertEqual(calls, [("/gain", {"gain_db": -1.0})])
            calls.clear()
            self.post("/native/live/controls", {"master_db": 99.0})                    # out of range: capped at +12
            self.assertEqual(calls, [("/gain", {"gain_db": 14.0})])
            self.post("/native/live/stop")
            calls.clear()
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            self.post("/native/live/start", {"confirm": True, "ir": "Vol.wav", "calibrated": False})
            self.assertEqual([c[1]["gain_db"] for c in calls if c[0] == "/gain"], [-18.0])
            self.post("/native/live/stop")

    def test_19_captures_reveal_opens_the_captures_folder(self):
        from unittest import mock
        with mock.patch.object(self.capture_api.takes.subprocess, "Popen") as popen:
            r = self.post("/captures/reveal")
        self.assertEqual(Path(r["path"]), self.wav_dir / "Catture")
        self.assertEqual(popen.call_args[0][0][-1], str(self.wav_dir / "Catture"))

    def test_20_inputs_have_their_own_folder_and_the_library_gets_tidied(self):
        fake = (self.wav_dir / "Test" / "input.wav").read_bytes()
        r = self.local.post("/api/capture/input/upload", files={"file": ("My Input.wav", fake, "audio/wav")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["rel"], "Input/My Input.wav")                 # not in Catture/
        found = [f["rel"] for f in self.get("/input")["found"]]
        self.assertEqual(found[0], "Input/My Input.wav")                          # the Input folder is listed first
        self.assertIn("Test/input.wav", found)
        # a library from before the split: a loose input and loose results (each with its sidecar) in Catture/
        catture = self.wav_dir / "Catture"
        catture.mkdir(exist_ok=True)
        (catture / "input.wav").write_bytes(fake)
        for name in ("Old Amp", "Old Pedal"):
            sf.write(str(catture / f"{name}.wav"), np.zeros(4800), 48_000, subtype="PCM_24")
            (catture / f"{name}.json").write_text("{}", encoding="utf-8")
        moved = self.capture_api.organize_library()
        self.assertEqual(moved["inputs"], ["input.wav"])
        self.assertEqual(sorted(moved["results"]), ["Old Amp", "Old Pedal"])
        self.assertTrue((self.wav_dir / "Input" / "input.wav").is_file())
        self.assertTrue((catture / "Old Amp" / "Old Amp.wav").is_file())
        self.assertTrue((catture / "Old Amp" / "Old Amp.json").is_file())
        self.assertFalse((catture / "input.wav").exists())
        self.assertEqual(self.capture_api.organize_library(), {"inputs": [], "results": []})     # idempotent

    def test_20b_the_standard_input_that_ships_with_the_app_is_seeded_only_when_the_library_has_none(self):
        from unittest import mock
        takes = self.capture_api.takes
        seed = self.wav_dir.parent / "seed-input.wav"
        seed.write_bytes((self.wav_dir / "Test" / "input.wav").read_bytes())
        with mock.patch.object(takes, "find_v3_inputs", return_value=[{"rel": "Input/x.wav", "name": "x.wav"}]):
            self.assertIsNone(takes.seed_standard_input(seed))                      # the library already has one: nothing is added
        with mock.patch.object(takes, "find_v3_inputs", return_value=[]):
            self.assertIsNone(takes.seed_standard_input(self.wav_dir.parent / "no-such-seed.wav"))
            name = takes.seed_standard_input(seed)                                  # an empty library: the shipped one is copied into Input/
            self.assertTrue((self.wav_dir / "Input" / name).is_file())
            self.assertEqual((self.wav_dir / "Input" / name).read_bytes(), seed.read_bytes())
            again = takes.seed_standard_input(seed)                                 # never over an existing file
            self.assertNotEqual(again, name)
            self.assertTrue((self.wav_dir / "Input" / name).is_file())

    def test_20c_the_shipped_standard_input_really_is_the_v3(self):
        shipped = Path(self.capture_api.__file__).resolve().parent.parent / "seed" / "input.wav"
        if not shipped.is_file():
            self.skipTest("the standard input is not part of the public source tree (suite/publish.py leaves it out); the installers carry it")
        # the md5 written out here, not al.INPUT_V3_MD5: this class swaps that one for its synthetic stand-in
        self.assertEqual(hashlib.md5(shipped.read_bytes()).hexdigest(), "36cd1af62985c2fac3e654333e36431e")

    def url(self, host):
        return f"http://{host}:{self.capture_api.trainer.TRAINER_PORT}/"

    def test_21_trainer_host_and_status_detect_it_live_on_the_network(self):
        self.post("/trainer/host", {"host": ""}, expect=400)
        self.settings.update(trainer_host="")
        self.assertEqual(self.get("/trainer/status"), {"host": None, "online": False, "local_online": False, "url": None})
        # nothing listens on the trainer's port yet: setting the host still works, just reports offline
        r = self.post("/trainer/host", {"host": "127.0.0.1"})
        self.assertEqual(r, {"host": "127.0.0.1", "online": False})
        self.assertEqual(self.get("/trainer/status"), {"host": "127.0.0.1", "online": False, "local_online": False, "url": self.url("127.0.0.1")})
        # once something answers its ping, the same address reports online
        with FakeTrainer():
            self.assertEqual(self.get("/trainer/status"), {"host": "127.0.0.1", "online": True, "local_online": True, "url": self.url("127.0.0.1")})
        self.assertEqual(self.get("/trainer/status"), {"host": "127.0.0.1", "online": False, "local_online": False, "url": self.url("127.0.0.1")})   # gone again after the fake stops
        # a wrong address while a trainer answers on this PC: offline, but the page is told that "localhost" would work
        self.settings.update(trainer_host="192.0.2.1")                                          # TEST-NET: nothing answers there
        with FakeTrainer():
            self.assertEqual(self.get("/trainer/status"), {"host": "192.0.2.1", "online": False, "local_online": True, "url": self.url("192.0.2.1")})

    def test_21b_the_address_is_cleaned_of_what_the_trainer_page_used_to_show(self):
        for typed, kept in (("192.168.1.230", "192.168.1.230"), ("http://192.168.1.230:8765", "192.168.1.230"), ("https://192.168.1.230:8765/", "192.168.1.230"),
                            ("  192.168.1.230:8765 ", "192.168.1.230"), ("my-pc.local:8765", "my-pc.local"), ("localhost", "localhost"), ("[::1]:8765", "::1")):
            self.assertEqual(self.post("/trainer/host", {"host": typed})["host"], kept, typed)
        self.assertEqual(self.settings.load()["trainer_host"], "::1")
        for bad in ("http://", "   ", "a b c", "<script>"):
            self.post("/trainer/host", {"host": bad}, expect=400)

    def test_22_trainer_scan_and_send(self):
        # a hand-made capture folder, shaped like one take_accept would leave behind
        cap_dir = self.wav_dir / "Catture" / "2026-01-01 12.00 - Fake Amp"
        cap_dir.mkdir(parents=True)
        sf.write(str(cap_dir / "Fake Amp.wav"), np.zeros(4800), 48_000, subtype="PCM_24")
        (cap_dir / "Fake Amp.json").write_text(json.dumps({"name": "Fake Amp", "input_rel": "Test/input.wav",
            "metadata": {"input_level_dbu": 1.0}}), encoding="utf-8")
        (cap_dir / "capture.json").write_text(json.dumps({"status": "accepted", "name": "Fake Amp",
            "created": "2026-01-01T12:00:00",
            "files": {"output": "Catture/2026-01-01 12.00 - Fake Amp/Fake Amp.wav"}}), encoding="utf-8")
        # scanning the parent ("Catture") finds it inside, marked exportable
        found = self.post("/trainer/scan", {"path": str(self.wav_dir / "Catture")})["captures"]
        entry = next(c for c in found if c["name"] == "Fake Amp")
        self.assertTrue(entry["exportable"])
        # scanning the capture folder itself directly also works
        direct = self.post("/trainer/scan", {"path": str(cap_dir)})["captures"]
        self.assertEqual([c["dir"] for c in direct], [str(cap_dir)])
        # sending refuses without a trainer address configured
        self.settings.update(trainer_host="")
        self.post("/trainer/send", {"dir": str(cap_dir)}, expect=409)
        # ... and refuses when nothing answers at the configured address
        self.settings.update(trainer_host="127.0.0.1")
        self.post("/trainer/send", {"dir": str(cap_dir)}, expect=503)
        with FakeTrainer() as trainer:
            r = self.post("/trainer/send", {"dir": str(cap_dir)})
            self.assertEqual(r, {"sent": "Fake Amp.wav", "input_sent": False})
            self.assertEqual(len(trainer.imports), 1)
            sent = trainer.imports[0]
            self.assertEqual(sent["name"], "Fake Amp")
            self.assertEqual(sent["wav_name"], "Fake Amp.wav")
            self.assertAlmostEqual(sent["metadata"]["input_level_dbu"], 1.0)
            import base64
            self.assertEqual(base64.b64decode(sent["wav_base64"]), (cap_dir / "Fake Amp.wav").read_bytes())
            # the same capture, referenced by its path relative to the library instead of an absolute one (the
            # button on the result screen only ever knows the relative folder) - and a traversal attempt is refused
            rel = "Catture/2026-01-01 12.00 - Fake Amp"
            r2 = self.post("/trainer/send", {"folder_rel": rel})
            self.assertEqual(r2, {"sent": "Fake Amp.wav", "input_sent": False})
            self.assertEqual(len(trainer.imports), 2)
        self.post("/trainer/send", {"folder_rel": "../../etc"}, expect=400)

    def test_22b_the_input_of_the_capture_goes_along_only_when_the_trainer_asks_for_it(self):
        cap_dir = self.wav_dir / "Catture" / "2026-01-01 12.01 - With Input"
        cap_dir.mkdir(parents=True)
        sf.write(str(cap_dir / "With Input.wav"), np.zeros(4800), 48_000, subtype="PCM_24")
        (cap_dir / "With Input.json").write_text(json.dumps({"name": "With Input", "input_rel": "Test/input.wav", "metadata": {}}), encoding="utf-8")
        (cap_dir / "capture.json").write_text(json.dumps({"status": "accepted", "name": "With Input", "input": {"ref": "Test/input.wav"},
            "files": {"output": "Catture/2026-01-01 12.01 - With Input/With Input.wav"}}), encoding="utf-8")
        self.settings.update(trainer_host="127.0.0.1")
        import base64
        import hashlib
        source = (self.wav_dir / "Test" / "input.wav").read_bytes()
        with FakeTrainer(needs_input=True) as trainer:
            self.assertEqual(self.post("/trainer/send", {"dir": str(cap_dir)}), {"sent": "With Input.wav", "input_sent": True})
            self.assertEqual(trainer.imports[0]["input_md5"], hashlib.md5(source).hexdigest())
            self.assertEqual(trainer.imports[0]["input_size"], len(source))
            self.assertEqual(len(trainer.inputs), 1)
            self.assertEqual(trainer.inputs[0]["capture_rel"], "Da NAM Capture/With Input.wav")
            self.assertEqual(trainer.inputs[0]["name"], "input.wav")
            self.assertEqual(base64.b64decode(trainer.inputs[0]["wav_base64"]), source)
        with FakeTrainer(needs_input=False) as trainer:           # the trainer already has that input: nothing more is sent
            self.assertEqual(self.post("/trainer/send", {"dir": str(cap_dir)})["input_sent"], False)
            self.assertEqual(trainer.inputs, [])
        with FakeTrainer(needs_input=True, input_status=500) as trainer:
            self.post("/trainer/send", {"dir": str(cap_dir)}, expect=503)

    def test_23_trainer_send_refuses_a_capture_without_the_trainer_files_yet(self):
        cap_dir = self.wav_dir / "Catture" / "2026-01-01 12.05 - Pending"
        cap_dir.mkdir(parents=True)
        (cap_dir / "capture.json").write_text(json.dumps({"status": "pending", "name": "Pending",
            "files": {"output": None}}), encoding="utf-8")
        self.settings.update(trainer_host="127.0.0.1")
        found = self.post("/trainer/scan", {"path": str(cap_dir)})["captures"]
        self.assertEqual(found, [{"dir": str(cap_dir), "name": "Pending", "status": "pending",
            "created": None, "exportable": False}])
        with FakeTrainer():
            self.post("/trainer/send", {"dir": str(cap_dir)}, expect=409)


if __name__ == "__main__":
    unittest.main()
