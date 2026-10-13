"""The NAM player's API (capture_api/player.py): calibrated gains of the three legs, the A/B switch, hot swap, the controls, stop,
and the way back from the trainer. The native engine and the trainer are stand-ins (a recorded list of calls / a tiny HTTP server):
what is proved here is the numbers and the order of the calls, not the compiled audio code (native/engine/tests and a real interface do that)."""

import http.server
import json
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from _helpers import TRAINER_TEST_PORT, FakeNativeEngine, pin_trainer_port

CO, CI_GUITAR, CI_RETURN = 14.3, 5.5, 9.2          # the interface's calibrated levels, dBu
MODEL_IN, MODEL_OUT = 6.0, 12.0                    # what the model's file says
MODEL_REPLY = {"ok": True, "input_level_dbu": MODEL_IN, "output_level_dbu": MODEL_OUT}


class FakeTrainerModels:
    """The trainer's two routes the player uses (/api/models, /api/models/<id>/model) and the ping, on 127.0.0.1:TRAINER_TEST_PORT."""

    MODELS = [{"id": "aaa111", "name": "Old amp", "created": 100.0, "has_model": True, "model_size": 10, "folder": "", "state": "done"},
              {"id": "bbb222", "name": "Fresh capture", "created": 200.0, "has_model": True, "model_size": 12, "folder": "", "state": "done"},
              {"id": "ccc333", "name": "Still training", "created": 300.0, "has_model": False, "state": "running"}]
    FILES = {"aaa111": b'{"version":"0.5.4","architecture":"WaveNet"}', "bbb222": b'{"version":"0.5.4","architecture":"WaveNet","x":1}',
             "ddd444": b"not a model at all"}

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, status, body, ctype="application/json"):
                self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/nam-trainer/ping":
                    return self._send(200, b'{"ok":true}')
                if self.path == "/api/models":
                    return self._send(200, json.dumps(outer.MODELS).encode())
                if self.path.startswith("/api/models/") and self.path.endswith("/model"):
                    key = self.path[len("/api/models/"):-len("/model")]
                    if key in outer.FILES:
                        return self._send(200, outer.FILES[key], "application/octet-stream")
                return self._send(404, b"{}")

            def log_message(self, *a):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", TRAINER_TEST_PORT), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown(); self._server.server_close(); self._thread.join(timeout=5)


class PlayerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import app as appmod
        import capture_api
        import settings
        from capture import store

        pin_trainer_port()
        cls.tmp = Path(tempfile.mkdtemp(prefix="nam-player-"))
        cls.wav_dir, cls.tmp_dir = cls.tmp / "wav", cls.tmp / "tmp"
        cls.model_dir, cls.ir_dir = cls.tmp / "Modelli NAM", cls.tmp / "IR"
        for d in (cls.wav_dir, cls.tmp_dir, cls.model_dir, cls.ir_dir):
            d.mkdir()
        cls.saved = (appmod.WAV_DIR, appmod.TMP_DIR, store.CALIBRATION_FILE, settings.SETTINGS_FILE)
        appmod.WAV_DIR, appmod.TMP_DIR = cls.wav_dir, cls.tmp_dir
        store.CALIBRATION_FILE = cls.tmp / "calibration.json"
        settings.SETTINGS_FILE = cls.tmp / "settings.json"
        capture_api.core._session = None
        capture_api.core.st.take = None
        capture_api.core.st.live_active = None
        capture_api.core.st.player_active = None
        capture_api.configure(wav_dir=cls.wav_dir, tmp_dir=cls.tmp_dir, model_dir=cls.model_dir, ir_dir=cls.ir_dir,
                              usage_file=cls.tmp / "live_usage.json", safe_name=appmod.safe_name, safe_subfolder=appmod.safe_subfolder,
                              unique_file=appmod.unique_file, wav_rel=appmod.wav_rel, wav_from_ref=appmod.wav_from_ref,
                              store_wav=appmod.store_wav, save_upload=appmod.save_upload)
        cls.appmod, cls.capture_api, cls.store, cls.settings = appmod, capture_api, store, settings
        cls.local = TestClient(appmod.app, client=("127.0.0.1", 50000))
        (cls.model_dir / "Amp.nam").write_text("{}")
        (cls.ir_dir / "Cab.wav").write_bytes(b"RIFF....")

    @classmethod
    def tearDownClass(cls):
        import shutil
        cls.capture_api.shutdown()
        cls.appmod.WAV_DIR, cls.appmod.TMP_DIR, cls.store.CALIBRATION_FILE, cls.settings.SETTINGS_FILE = cls.saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.capture_api.core.st.live_active = None
        self.capture_api.core.st.player_active = None

    def post(self, path, json=None, expect=200):
        r = self.local.post("/api/capture" + path, json=json or {})
        self.assertEqual(r.status_code, expect, r.text)
        return r.json()

    def get(self, path, expect=200):
        r = self.local.get("/api/capture" + path)
        self.assertEqual(r.status_code, expect, r.text)
        return r.json()

    def open_both_inputs(self, calibrate=True):
        """The demo interface opened as the player wants it: guitar = input 1, return = input 0, output 0."""
        self.post("/open", {"device": "sim:demo", "in_channels": [1, 0], "out_channels": [0]})
        name = self.get("/state")["engine"]["opened"]["device_name"]
        if calibrate:
            self.store.put("out", name, 0, 48000, co_dbu=CO)
            self.store.put("in", name, 1, 48000, ci_dbu=CI_GUITAR, co_at_time=CO)
            self.store.put("in", name, 0, 48000, ci_dbu=CI_RETURN, co_at_time=CO)
        return name

    def start_body(self, **over):
        return {"confirm": True, "model": "Amp.nam", "monitor_out_channels": [2, 3], **over}

    NATIVE_REPLIES = {"/open_player": {"ok": True, "blocksize": 32, "sample_rate": 48000}, "/model": MODEL_REPLY}

    # ----------------------------------------------------------------------------------------------------------
    def test_01_start_is_refused_until_everything_it_needs_is_there(self):
        self.post("/native/player/start", self.start_body(), expect=409)                    # nothing open yet: "not open"
        self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
        self.post("/native/player/start", self.start_body(), expect=400)                    # only one input open
        self.open_both_inputs(calibrate=False)
        self.post("/native/player/start", self.start_body(confirm=False), expect=400)
        self.post("/native/player/start", self.start_body(model=None), expect=400)
        self.post("/native/player/start", self.start_body(leg="C"), expect=400)
        self.post("/native/player/start", self.start_body(monitor_out_channels=[1, 2, 3]), expect=400)
        self.post("/native/player/start", self.start_body(monitor_out_channels=None), expect=400)
        refusal = self.local.post("/api/capture/native/player/start", json=self.start_body())   # nothing calibrated, calibrated mode
        self.assertEqual(refusal.status_code, 409)
        self.assertIn("passo 3", refusal.json()["detail"])                                  # says what is missing
        self.assertIn("passo 4", refusal.json()["detail"])
        self.post("/native/player/start", self.start_body(model="../../x.nam", calibrated=False), expect=400)
        self.post("/native/player/start", self.start_body(model="Nope.nam", calibrated=False), expect=404)
        self.assertIsNotNone(self.get("/state")["engine"]["opened"])                       # never handed the device away for a refusal

    def test_02_calibrated_start_sends_the_right_numbers_and_stop_gives_the_device_back(self):
        self.open_both_inputs()
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            calls = fake.calls
            started = self.post("/native/player/start", self.start_body(ir="Cab.wav", b_delay_ms=2.0))
            g = started["gains"]
            self.assertAlmostEqual(started["feed_gain_db"], CI_GUITAR - CO, delta=0.01)    # guitar -> the gear: the monitor's unity gain
            self.assertAlmostEqual(g["pre"], CI_GUITAR - MODEL_IN, delta=0.01)             # guitar -> model
            self.assertAlmostEqual(g["post"], MODEL_OUT - CO, delta=0.01)                  # model -> monitor
            self.assertAlmostEqual(g["amp"], CI_RETURN - CO, delta=0.01)                   # return -> monitor: the real gear's own level
            self.assertTrue(g["model_levels"])
            self.assertEqual(started["b_delay_samples"], 96)                               # 2 ms at 48 kHz
            self.assertEqual(started["cal"], {"co_dbu": CO, "ci_guitar_dbu": CI_GUITAR, "ci_return_dbu": CI_RETURN})
            self.assertIsNone(self.get("/state")["engine"]["opened"])                      # Python gave the interface to the native engine
            self.assertEqual([c[0] for c in calls],
                             ["/open_player", "/model", "/ir", "/ir/unload", "/pregain", "/gain", "/amp_gain", "/b_delay", "/leg"])
            self.assertEqual(calls[0][1]["guitar_channel"], 1)
            self.assertEqual(calls[0][1]["return_channel"], 0)
            self.assertEqual(calls[0][1]["reamp_out_channel"], 0)
            self.assertEqual(calls[0][1]["monitor_out_channels"], [2, 3])
            self.assertAlmostEqual(calls[0][1]["feed_gain_db"], CI_GUITAR - CO, delta=0.01)
            self.assertEqual(calls[2][1]["legs"], ["B"])                                    # the cabinet is on the model's leg only ...
            self.assertEqual(calls[3][1], {"legs": ["A"]})                                  # ... and never left on the gear's leg
            self.assertEqual(calls[-1][1], {"leg": "A"})                                    # starts on the real gear
            self.assertEqual(calls[5], ("/gain", {"gain_db": round(MODEL_OUT - CO, 2) - 18.0}))      # the master volume (-18 dB) is on both legs' output gain ...
            self.assertEqual(calls[6], ("/amp_gain", {"gain_db": round(CI_RETURN - CO, 2) - 18.0}))
            self.assertEqual(g["post"], round(MODEL_OUT - CO, 2))                                   # ... while the gains shown stay the calibrated ones
            # the A/B switch
            self.post("/native/player/leg", {"leg": "B"})
            self.assertEqual(calls[-1], ("/leg", {"leg": "B"}))
            self.post("/native/player/leg", {"leg": "X"}, expect=400)
            # a running player blocks a second start and the live tool
            self.post("/native/player/start", self.start_body(), expect=409)
            self.post("/native/live/start", {"confirm": True, "ir": "Cab.wav"}, expect=409)
            # stop: the native engine is stopped and Python has both inputs back, guitar first
            self.post("/native/player/stop")
            self.assertEqual(calls[-1][0], "/stop")
            self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [1, 0])
            self.post("/native/player/leg", {"leg": "A"}, expect=409)                       # nothing running any more

    def test_03_uncalibrated_start_needs_no_calibration_and_uses_0_db(self):
        self.post("/open", {"device": "sim:demo", "in_channels": [3, 2], "out_channels": [1]})     # channels nobody calibrated
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            started = self.post("/native/player/start", self.start_body(calibrated=False))
            self.assertEqual(started["gains"], {"feed": 0.0, "amp": 0.0, "pre": 0.0, "post": 0.0, "model_levels": False, "amp_matched": False})   # by hand: 0 dB, the master volume is what keeps it safe
            self.assertEqual(started["feed_gain_db"], 0.0)
            self.assertIsNone(started["cal"])
            self.assertEqual([c[1]["pre_gain_db"] for c in fake.calls if c[0] == "/pregain"], [0.0])
            self.post("/native/player/stop")

    def test_03d_the_master_volume_rides_on_both_legs_and_starts_at_minus_18(self):
        self.open_both_inputs()
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            calls = fake.calls
            self.post("/native/player/start", self.start_body(master_db=-12.0))
            self.assertEqual(self.get("/native/player/active")["active"]["master_db"], -12.0)
            del calls[:]
            self.post("/native/player/controls", {"master_db": -3.0})
            self.assertEqual(calls, [("/gain", {"gain_db": round(MODEL_OUT - CO, 2) - 3.0}), ("/amp_gain", {"gain_db": round(CI_RETURN - CO, 2) - 3.0})])
            del calls[:]
            self.post("/native/player/controls", {"gain_db": 1.0})                                  # the Output knob keeps working, the master stays on top
            self.assertEqual(calls, [("/gain", {"gain_db": -2.0})])
            del calls[:]
            self.post("/native/player/controls", {"master_db": 99.0})                               # out of range: capped
            self.assertEqual(calls[0], ("/gain", {"gain_db": 13.0}))                                # 1 dB + 12 dB
            del calls[:]
            swapped = self.post("/native/player/swap", {"confirm": True, "model": "Amp.nam"})       # a swap keeps the master
            self.assertEqual(calls[[c[0] for c in calls].index("/gain")][1]["gain_db"], round(swapped["gains"]["post"] + 12.0, 2))
            self.post("/native/player/stop")
            default = self.start_body()
            self.assertNotIn("master_db", default)                                                  # not sent: the server's own default (-18 dB) applies
            self.post("/native/player/start", default)
            self.assertEqual(self.get("/native/player/active")["active"]["master_db"], -18.0)
            self.post("/native/player/stop")

    def test_03a_the_return_input_is_optional_the_real_gears_leg_then_stays_at_0_db(self):
        self.open_both_inputs()                                                                    # output 0, guitar 1 and input 0 calibrated
        self.post("/close")
        self.post("/open", {"device": "sim:demo", "in_channels": [1, 3], "out_channels": [0]})     # return on channel 3: nobody calibrated it
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            started = self.post("/native/player/start", self.start_body())
            g = started["gains"]
            self.assertAlmostEqual(g["pre"], CI_GUITAR - MODEL_IN, delta=0.01)                     # the model's leg is fully calibrated
            self.assertAlmostEqual(g["post"], MODEL_OUT - CO, delta=0.01)
            self.assertAlmostEqual(started["feed_gain_db"], CI_GUITAR - CO, delta=0.01)
            self.assertEqual((g["amp"], g["amp_matched"]), (0.0, False))                            # the real gear's leg cannot be matched
            self.assertIsNone(started["cal"]["ci_return_dbu"])
            self.post("/native/player/stop")
        self.post("/close")
        self.post("/open", {"device": "sim:demo", "in_channels": [3, 3], "out_channels": [0]})      # the guitar input must be calibrated
        self.post("/native/player/start", self.start_body(), expect=409)

    def test_03b_the_profiles_levels_can_be_bypassed_even_when_calibrated(self):
        self.open_both_inputs()
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            started = self.post("/native/player/start", self.start_body(use_model_levels=False))
            g = started["gains"]
            self.assertEqual((g["pre"], g["post"], g["model_levels"]), (0.0, 0.0, False))        # the model's gains are manual ...
            self.assertAlmostEqual(g["amp"], CI_RETURN - CO, delta=0.01)                            # ... the real gear's leg and the send stay calibrated
            self.assertAlmostEqual(started["feed_gain_db"], CI_GUITAR - CO, delta=0.01)
            self.assertEqual(self.get("/native/player/active")["active"]["use_model_levels"], False)
            swapped = self.post("/native/player/swap", {"confirm": True, "model": "Amp.nam"})      # a swap keeps the choice
            self.assertEqual((swapped["gains"]["pre"], swapped["gains"]["post"]), (0.0, 0.0))
            self.post("/native/player/stop")

    def test_03c_the_send_and_the_monitor_cannot_share_an_output_channel(self):
        self.post("/open", {"device": "sim:demo", "in_channels": [1, 0], "out_channels": [2]})
        self.post("/native/player/start", self.start_body(calibrated=False, monitor_out_channels=[2, 3]), expect=400)
        self.post("/native/player/start", self.start_body(calibrated=False, monitor_out_channels=[0, 2]), expect=400)
        self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [1, 0])           # refused before the device was handed over

    def test_04_a_model_without_levels_is_played_at_0_db_and_says_so(self):
        self.open_both_inputs()
        with FakeNativeEngine({**self.NATIVE_REPLIES, "/model": {"ok": True, "input_level_dbu": None, "output_level_dbu": None}}) as fake:
            started = self.post("/native/player/start", self.start_body())
            self.assertFalse(started["gains"]["model_levels"])
            self.assertEqual((started["gains"]["pre"], started["gains"]["post"]), (0.0, 0.0))      # manual: in and out at 0 dB
            self.assertAlmostEqual(started["gains"]["amp"], CI_RETURN - CO, delta=0.01)    # the gear's leg does not depend on the model
            self.post("/native/player/stop")

    def test_04b_a_head_profile_with_only_the_input_level_still_gets_its_calibrated_input(self):
        """The real case that was heard: a head capture writes input_level_dbu (the send) and no output level (the return is a mic/DI). The model's
        input used to fall to 0 dB - the guitar (CI_GUITAR - MODEL_IN) dB too hot in the model against the real head. Now the input comes from the
        profile and the output is matched to the calibrated return, i.e. to leg A."""
        self.open_both_inputs()
        with FakeNativeEngine({**self.NATIVE_REPLIES, "/model": {"ok": True, "input_level_dbu": MODEL_IN, "output_level_dbu": None}}) as fake:
            started = self.post("/native/player/start", self.start_body())
            g = started["gains"]
            self.assertAlmostEqual(g["pre"], CI_GUITAR - MODEL_IN, delta=0.01)             # from the profile, not 0 dB
            self.assertAlmostEqual(g["post"], CI_RETURN - CO, delta=0.01)                  # = leg A: both legs leave on the same level
            self.assertEqual(g["post"], g["amp"])
            self.assertTrue(g["pre_from_profile"] and g["post_assumed"])
            self.assertFalse(g["model_levels"])                                            # the profile is still not complete
            self.post("/native/player/stop")

    def test_04c_the_same_profile_without_a_calibrated_return_keeps_the_output_by_hand(self):
        g = self.capture_api.player._gains({"co": CO, "ci_guitar": CI_GUITAR, "ci_return": None}, MODEL_IN, None)
        self.assertAlmostEqual(g["pre"], CI_GUITAR - MODEL_IN, delta=0.01)
        self.assertEqual(g["post"], 0.0)                                                    # nothing to match it to: by hand
        self.assertTrue(g["pre_from_profile"]); self.assertFalse(g["post_assumed"]); self.assertFalse(g["amp_matched"])

    def test_05_swap_recomputes_the_model_gains_and_moves_the_cabinet(self):
        self.open_both_inputs()
        (self.model_dir / "Other.nam").write_text("{}")
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            calls = fake.calls
            self.post("/native/player/swap", self.start_body(), expect=409)                # not running yet
            self.post("/native/player/start", self.start_body())
            del calls[:]
            swapped = self.post("/native/player/swap", {"confirm": True, "model": "Other.nam", "ir": "Cab.wav", "ir_on_gear": True})
            self.assertAlmostEqual(swapped["gains"]["pre"], CI_GUITAR - MODEL_IN, delta=0.01)
            self.assertEqual([c[0] for c in calls], ["/model", "/ir", "/pregain", "/gain", "/amp_gain"])
            self.assertEqual(calls[1][1]["legs"], ["B", "A"])                               # this time the gear's leg has the cabinet too
            self.assertTrue(str(calls[0][1]["path"]).endswith("Other.nam"))
            self.post("/native/player/swap", {"confirm": True, "model": None}, expect=400)  # a player always has a model
            self.post("/native/player/stop")

    def test_06_controls_relay_the_knobs_and_the_players_own(self):
        self.open_both_inputs()
        with FakeNativeEngine(self.NATIVE_REPLIES) as fake:
            calls = fake.calls
            self.post("/native/player/controls", {"b_trim_db": 1.5}, expect=409)           # not running
            self.post("/native/player/start", self.start_body())
            del calls[:]
            self.post("/native/player/controls", {"pre_gain_db": 2.0, "gate_threshold_db": -40.0, "eq_bass_db": 3.0})
            self.assertEqual(calls, [("/pregain", {"pre_gain_db": 2.0}), ("/gate", {"threshold_db": -40.0}), ("/eq", {"bass_db": 3.0})])
            del calls[:]
            self.post("/native/player/controls", {"b_trim_db": -1.5, "b_delay_ms": 3.0, "amp_gain_db": -2.0})
            self.assertEqual(calls, [("/b_trim", {"trim_db": -1.5}), ("/b_delay", {"samples": 144}), ("/amp_gain", {"gain_db": -20.0})])        # -2 dB + the master volume (-18 dB by default)
            del calls[:]
            self.post("/native/player/controls", {"b_delay_ms": 5000.0})                    # more than the engine can hold: capped
            self.assertEqual(calls[0][1]["samples"], 9120)                                  # 190 ms
            self.post("/native/player/stop")

    def test_06b_an_a2_model_offers_its_lite_version(self):
        # official trainer 0.13: a SlimmableContainer with a Lite and a full WaveNet; the engine says how many sizes, the player starts on the full one
        self.open_both_inputs()
        replies = {**self.NATIVE_REPLIES, "/model": {**MODEL_REPLY, "sizes": 2}, "/model_size": lambda body: {"ok": True, "index": body["index"], "sizes": 2}}
        with FakeNativeEngine(replies) as fake:
            r = self.post("/native/player/start", self.start_body())
            self.assertEqual((r["model_sizes"], r["model_size"]), (2, 1))
            del fake.calls[:]
            self.post("/native/player/controls", {"model_size": 0})
            self.assertEqual(fake.calls, [("/model_size", {"index": 0})])
            active = self.local.get("/api/capture/native/player/active").json()["active"]
            self.assertEqual((active["model_sizes"], active["model_size"]), (2, 0))
            self.post("/native/player/stop")
        with FakeNativeEngine(self.NATIVE_REPLIES):                                      # an ordinary model: one size, no switch on the page
            r = self.post("/native/player/start", self.start_body())
            self.assertEqual((r["model_sizes"], r["model_size"]), (1, 0))
            self.post("/native/player/stop")

    # ----------------------------------------------------------------------------------------------------------
    # "Test Pedali": a pedal's model in front of the amp's
    PEDAL_IN, PEDAL_OUT = 3.0, 8.0
    PEDAL_REPLIES = {"/open_player": {"ok": True, "blocksize": 32, "sample_rate": 48000}, "/model": {**MODEL_REPLY, "sizes": 2},
                     "/pedal_model": {"ok": True, "input_level_dbu": 3.0, "output_level_dbu": 8.0, "sizes": 2}}

    def test_06c_a_pedal_test_loads_the_amp_twice_and_levels_the_pedals_output_into_it(self):
        (self.model_dir / "Pedal.nam").write_text("{}")
        self.open_both_inputs()
        with FakeNativeEngine(self.PEDAL_REPLIES) as fake:
            calls = fake.calls
            r = self.post("/native/player/start", self.start_body(pedal="Pedal.nam", ir="Cab.wav"))
            g = r["gains"]
            self.assertAlmostEqual(g["pedal_pre"], CI_GUITAR - self.PEDAL_IN, delta=0.01)       # guitar -> the pedal's input level
            self.assertAlmostEqual(g["pre"], self.PEDAL_OUT - MODEL_IN, delta=0.01)             # the pedal's output level is what the amp is fed with
            self.assertAlmostEqual(g["amp_pre_a"], CI_RETURN - MODEL_IN, delta=0.01)            # the real pedal's return -> the amp's input level
            self.assertAlmostEqual(g["post"], MODEL_OUT - CO, delta=0.01)
            self.assertEqual(g["amp"], g["post"])                                               # both legs leave the amp stage at the same level
            self.assertTrue(g["model_levels"] and g["amp_matched"])
            self.assertEqual(r["pedal"], {"ref": "Pedal.nam", "input_level_dbu": 3.0, "output_level_dbu": 8.0, "sizes": 2, "size": 1})
            by = {}
            for path, body in calls:
                by.setdefault(path, []).append(body)
            self.assertEqual(by["/model"][0]["legs"], ["B", "A"])                                # a second copy of the amp for the real pedal's return
            self.assertEqual(by["/ir"][0]["legs"], ["B", "A"])                                   # the cabinet is part of the amp stage: both legs
            self.assertEqual(by["/pedal_model"][0]["path"], str((self.model_dir / "Pedal.nam").resolve()))      # resolved: macOS temp folders sit behind a /var -> /private/var link
            self.assertEqual(by["/pedal_controls"], [{"pre_db": round(CI_GUITAR - self.PEDAL_IN, 2), "post_db": 0.0}])
            self.assertEqual(by["/amp_pregain"], [{"gain_db": round(CI_RETURN - MODEL_IN, 2)}])
            self.assertEqual(by["/gain"], by["/amp_gain"])                                       # same output gain (master -18 dB added) on both
            # the amp's Input knob moves both legs, A keeping its own calibrated offset; Output moves both
            del calls[:]
            self.post("/native/player/controls", {"pre_gain_db": 5.0})
            offset = round(CI_RETURN - MODEL_IN, 2) - round(self.PEDAL_OUT - MODEL_IN, 2)
            self.assertEqual(calls, [("/pregain", {"pre_gain_db": 5.0}), ("/amp_pregain", {"gain_db": 5.0 + offset})])
            del calls[:]
            self.post("/native/player/controls", {"gain_db": -4.0})
            self.assertEqual(calls, [("/gain", {"gain_db": -22.0}), ("/amp_gain", {"gain_db": -22.0})])
            del calls[:]
            self.post("/native/player/controls", {"pedal_pre_gain_db": 1.0, "pedal_gain_db": -2.0})
            self.assertEqual(calls, [("/pedal_controls", {"pre_db": 1.0, "post_db": -2.0})])
            del calls[:]
            self.post("/native/player/controls", {"pedal_size": 0})
            self.assertEqual(calls, [("/pedal_controls", {"size": 0})])
            active = self.get("/native/player/active")["active"]
            self.assertEqual((active["pedal_ref"], active["pedal"]["size"]), ("Pedal.nam", 0))
            # without the pedal again: it goes, with the amp's copy on the real gear's leg and that leg's input gain
            del calls[:]
            back = self.post("/native/player/swap", self.start_body())
            self.assertIsNone(back["pedal"])
            paths = [c[0] for c in calls]
            self.assertIn("/pedal_model/unload", paths)
            self.assertIn(("/model/unload", {"legs": ["A"]}), calls)
            self.assertIn(("/amp_pregain", {"gain_db": 0.0}), calls)
            self.post("/native/player/stop")

    def test_06d_a_calibrated_pedal_test_needs_the_pedals_output_level(self):
        (self.model_dir / "Pedal.nam").write_text("{}")
        self.open_both_inputs()
        replies = {**self.PEDAL_REPLIES, "/pedal_model": {"ok": True, "input_level_dbu": 3.0, "output_level_dbu": None, "sizes": 1}}
        with FakeNativeEngine(replies):
            r = self.local.post("/api/capture/native/player/start", json=self.start_body(pedal="Pedal.nam"))
            self.assertEqual(r.status_code, 409, r.text)
            self.assertIn("pedale", r.json()["detail"].lower())
        self.open_both_inputs()
        with FakeNativeEngine(replies):                                                          # by hand (levels ignored) it plays, everything at 0 dB
            r = self.post("/native/player/start", self.start_body(pedal="Pedal.nam", use_model_levels=False))
            self.assertEqual((r["gains"]["pedal_pre"], r["gains"]["pre"], r["gains"]["amp_pre_a"]), (0.0, 0.0, 0.0))
            self.post("/native/player/stop")

    def test_06e_the_pedal_alone_without_an_amp_model_switches_the_amp_stage_off(self):
        (self.model_dir / "Pedal.nam").write_text("{}")
        self.open_both_inputs()
        with FakeNativeEngine(self.PEDAL_REPLIES) as fake:
            calls = fake.calls
            self.post("/native/player/start", self.start_body(model=None), expect=400)                    # neither an amp nor a pedal: nothing to play
            r = self.post("/native/player/start", self.start_body(model=None, pedal="Pedal.nam"))
            g = r["gains"]
            self.assertFalse(g["shared_amp"])
            self.assertAlmostEqual(g["pedal_pre"], CI_GUITAR - self.PEDAL_IN, delta=0.01)
            self.assertAlmostEqual(g["post"], self.PEDAL_OUT - CO, delta=0.01)                           # leg B ends on the pedal's own output level
            self.assertAlmostEqual(g["amp"], CI_RETURN - CO, delta=0.01)                                  # leg A is a bare return again
            self.assertEqual((g["pre"], g["amp_pre_a"]), (0.0, 0.0))
            paths = [c[0] for c in calls]
            self.assertNotIn("/model", paths)
            self.assertIn(("/model/unload", {"legs": ["B", "A"]}), calls)
            self.assertEqual(r["model_input_level_dbu"], None)
            del calls[:]
            self.post("/native/player/controls", {"pre_gain_db": 3.0, "gain_db": -4.0})                   # no shared amp stage: A is left alone
            self.assertEqual(calls, [("/pregain", {"pre_gain_db": 3.0}), ("/gain", {"gain_db": -22.0})])
            # an amp model back in: the pedal test with the amp, then the amp alone, then the pedal alone again
            both = self.post("/native/player/swap", self.start_body(pedal="Pedal.nam"))
            self.assertTrue(both["gains"]["shared_amp"])
            alone = self.post("/native/player/swap", self.start_body())
            self.assertIsNone(alone["pedal"])
            del calls[:]
            self.post("/native/player/swap", self.start_body(model=None, pedal="Pedal.nam"))
            self.assertIn(("/model/unload", {"legs": ["B", "A"]}), calls)
            self.post("/native/player/swap", self.start_body(model=None), expect=400)
            self.post("/native/player/stop")

    def test_07_a_failure_on_the_native_side_gives_the_device_back(self):
        self.open_both_inputs()
        r = self.local.post("/api/capture/native/player/start", json=self.start_body())      # no native engine listening at all
        self.assertEqual(r.status_code, 502, r.text)
        opened = self.get("/state")["engine"]["opened"]
        self.assertEqual(opened["in_channels"], [1, 0])                                      # Python has both inputs back, as before
        self.assertIsNone(self.capture_api.core.st.player_active)

    def test_08_models_come_back_from_the_trainer(self):
        offline = self.get("/trainer/models")
        self.assertEqual((offline["online"], offline["models"]), (False, []))                # no trainer configured: an empty list, not an error
        self.post("/trainer/host", {"host": "127.0.0.1"})
        self.assertFalse(self.get("/trainer/models")["online"])                              # configured but nobody there
        with FakeTrainerModels():
            listing = self.get("/trainer/models")
            self.assertTrue(listing["online"])
            self.assertEqual([m["id"] for m in listing["models"]], ["bbb222", "aaa111"])     # newest first, the one still training is left out
            one_off = self.post("/trainer/models/fetch", {"id": "bbb222", "name": "Fresh capture"})
            self.assertFalse(one_off["imported"])
            self.assertTrue(one_off["ref"].startswith("tmp:"))
            self.assertTrue((self.tmp_dir / "live_uploads" / one_off["ref"][4:]).read_bytes().startswith(b"{"))
            kept = self.post("/trainer/models/fetch", {"id": "aaa111", "name": "Old amp", "do_import": True})
            self.assertTrue(kept["imported"])
            self.assertTrue((self.model_dir / kept["ref"]).is_file())
            self.post("/trainer/models/fetch", {"id": "zzz", "name": "x"}, expect=404)       # the trainer no longer has it
            self.post("/trainer/models/fetch", {"id": "ddd444", "name": "x"}, expect=502)    # not a model file
            self.post("/trainer/models/fetch", {"id": "../../etc/passwd", "name": "x"}, expect=400)


    def test_09_tuner_runs_alone_on_the_guitar_channel_and_gives_the_device_back(self):
        st = self.capture_api.core.st
        st.tuner_active = None
        self.post("/close")
        self.post("/native/tuner/start", expect=409)                                          # nothing open yet
        self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})  # the guitar input of step 1, first
        with FakeNativeEngine({"/open": {"ok": True}}) as fake:
            self.assertEqual(self.post("/native/tuner/start"), {"guitar_ch": 1})
            self.assertEqual(fake.calls[0][0], "/open")
            body = fake.calls[0][1]
            self.assertEqual((body["in_channel"], body["out_channel"], body["extra_out_channels"]), (1, 0, []))
            self.assertLessEqual(body["gain_db"], -100.0)                                      # nothing reaches the pedal or the amp
            self.assertIsNone(self.get("/state")["engine"]["opened"])                          # Python gave the interface to the native engine
            self.assertEqual(self.get("/native/tuner/active")["active"], {"guitar_ch": 1})
            self.assertEqual(self.post("/native/tuner/start"), {"guitar_ch": 1})               # starting twice is harmless
            self.assertEqual([c[0] for c in fake.calls], ["/open"])
            # the page opening the device itself (or closing it) takes the interface back first
            self.post("/open", {"device": "sim:demo", "in_channels": [0], "out_channels": [0]})
            self.assertEqual([c[0] for c in fake.calls], ["/open", "/stop"])
            self.assertIsNone(self.get("/native/tuner/active")["active"])
            self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [0])
            # stop gives the device back as it was
            self.post("/close")
            self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})
            self.post("/native/tuner/start")
            self.post("/native/tuner/stop")
            self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [1])
            self.assertIsNone(self.get("/native/tuner/active")["active"])
            self.post("/native/tuner/stop")                                                    # nothing running: not an error
            # a live session or the player in the way: refused
            st.player_active = {"x": 1}
            self.post("/native/tuner/start", expect=409)
            st.player_active = None
        # no native engine at all: the tuner fails cleanly and Python keeps its channels
        self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})
        self.post("/native/tuner/start", expect=502)
        self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [1])
        self.assertIsNone(self.capture_api.core.st.tuner_active)

    def test_09b_every_driver_session_is_one_the_page_asked_for(self):
        """Each open of the interface sets the driver's one buffer size and, between sessions, the driver falls back to its own: a stop followed by the page's
        own restore must not open the interface twice, and a hand-over must not make a test stream first."""
        st = self.capture_api.core.st
        st.tuner_active = None
        self.post("/close")
        opened_with = []
        real_open = self.capture_api.core.session().open

        def spy(*a, **kw):
            opened_with.append(kw if kw else a)
            return real_open(*a, **kw)

        with mock.patch.object(self.capture_api.core.session(), "open", side_effect=spy):
            self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0], "probe": False})
            with FakeNativeEngine({"/open": {"ok": True}}):
                self.post("/native/tuner/start")
                n = len(opened_with)
                self.post("/native/tuner/stop?restore=0")                                   # the page restores by itself: the server does not open it too
                self.assertEqual(len(opened_with), n)
                self.assertIsNone(self.get("/state")["engine"]["opened"])
                self.assertIsNone(st.native_prev)
                self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})
                self.post("/native/tuner/start")
                n = len(opened_with)
                self.post("/native/tuner/stop")                                              # the default still hands the device back
                self.assertEqual(len(opened_with), n + 1)
                self.assertEqual(self.get("/state")["engine"]["opened"]["in_channels"], [1])
                # opening or closing the device while the tuner runs: it lets go without a Python session in between
                self.post("/native/tuner/start")
                n = len(opened_with)
                self.post("/close")
                self.assertEqual(len(opened_with), n)
                self.assertIsNone(st.tuner_active)

    def test_10_live_start_takes_the_interface_back_from_the_tuner(self):
        st = self.capture_api.core.st
        self.post("/open", {"device": "sim:demo", "in_channels": [1], "out_channels": [0]})
        with FakeNativeEngine({"/open": {"ok": True}}) as fake:
            self.post("/native/tuner/start")
            self.post("/native/live/start", {"confirm": True, "ir": "Cab.wav", "calibrated": False}, expect=200)
            self.assertEqual(fake.calls[1][0], "/stop")                                      # the tuner let go first
            self.assertIsNone(st.tuner_active)
            self.assertIsNotNone(st.live_active)
            self.post("/native/live/stop")


if __name__ == "__main__":
    unittest.main()
