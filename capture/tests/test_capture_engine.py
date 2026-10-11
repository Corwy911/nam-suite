"""The engine process with the simulated interface, driven through the same session class the server uses."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _helpers import al, write_fake_input
from capture import calibration as cal
from capture.session import EngineError, EngineSession


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nam-engine-"))
        cls.s = EngineSession(cls.tmp)
        cls.s.ensure_started()
        cls.s.call("sim", speed=0)                       # as fast as possible
        cls.s.open("sim:demo", [0], [0])

    @classmethod
    def tearDownClass(cls):
        cls.s.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_op(self, cmd, timeout=120, **params):
        self.s.start_op(cmd, **params)
        return self.s.wait_op(timeout)["result"]

    def test_devices_lists_the_demo_interface(self):
        devices = self.s.call("devices")["devices"]
        self.assertIn("sim:demo", [d["id"] for d in devices])

    def test_bad_channels_and_rate_are_refused(self):
        with self.assertRaises(EngineError) as e:
            self.s.open("sim:demo", [7], [0])
        self.assertEqual(e.exception.key, "bad_channels")
        with self.assertRaises(EngineError) as e:
            self.s.open("sim:demo", [0], [0], rate=44_100)
        self.assertEqual(e.exception.key, "rate_refused")
        self.s.open("sim:demo", [0], [0])

    def test_multimeter_reading_gives_the_output_level(self):
        self.s.call("sim", co_dbu=14.3)
        for sent in (0.0, -12.0):
            v = self.s.call("sim_multimeter", level_dbfs=sent)["vrms"]
            self.assertAlmostEqual(cal.output_level_dbu(v, sent), 14.3, places=6)

    def test_measure_reads_the_expected_level_in_loopback(self):
        self.s.call("sim", co_dbu=14.3, ci_dbu=9.2, route="loopback")
        r = self.run_op("measure", level_dbfs=-12.0)
        self.assertAlmostEqual(r["peak_dbfs"], 14.3 - 12.0 - 9.2, delta=0.05)
        self.assertEqual(r["clipped"], 0)

    def test_return_calibration_recovers_the_input_level(self):
        self.s.call("sim", co_dbu=14.3, ci_dbu=9.2, route="loopback")
        r = self.run_op("return_cal", co=14.3)
        self.assertAlmostEqual(r["ci"], 9.2, delta=0.05)
        self.assertLess(r["spread_db"], 0.05)
        self.assertEqual(r["judge"], "ok")
        self.assertLessEqual(r["level_dbfs"], -0.5)

    def test_return_calibration_with_a_hot_input_backs_off(self):
        # an input that reads full scale at +2 dBu: -40 dBFS sent is already loud, the search must settle without clipping
        self.s.call("sim", co_dbu=20.0, ci_dbu=2.0, route="loopback")
        r = self.run_op("return_cal", co=20.0)
        self.assertAlmostEqual(r["ci"], 2.0, delta=0.05)
        self.assertEqual(r["clipped"], 0)

    def test_click_probe_finds_the_round_trip(self):
        self.s.call("sim", latency=1056, route="loopback")
        r = self.run_op("click_probe")
        self.assertEqual(r["delay"], 1056)

    def test_take_end_to_end(self):
        self.s.call("sim", co_dbu=14.3, ci_dbu=9.2, latency=1056, route="gear", gear="head")
        src = self.tmp / "input.wav"
        write_fake_input(src)
        raw = self.tmp / "take.npy"
        r = self.run_op("record", timeout=300, input_path=str(src), save_path=str(raw), extra_frames=48_000)
        take = np.load(raw)
        self.assertEqual(len(take), al.INPUT_V3_FRAMES + 48_000)
        aligned, rep = al.analyze_take(take)
        self.assertEqual(len(aligned), al.INPUT_V3_FRAMES)
        self.assertEqual(rep["latency"]["delay"], 1056)
        self.assertEqual(rep["residual"]["recommended"], 0)
        self.assertLess(rep["replicate_esr"], 0.01)
        self.assertTrue(rep["ok"], rep["warnings"])
        self.assertEqual(r["xruns"], 0)
        # cut at delay-1, so the response of the first blip lands one sample after the input's blip (residual delay 1 = what the
        # official trainer's safety factor expects; it then applies 0)
        first = int(np.argmax(np.abs(aligned[al.BLIP_LOCATIONS[0] - 50:al.BLIP_LOCATIONS[0] + 50]))) + al.BLIP_LOCATIONS[0] - 50
        self.assertEqual(first, al.BLIP_LOCATIONS[0] + 1)

    def _take_with_step(self, slip_at, warmup=True):
        self.s.call("sim", co_dbu=14.3, ci_dbu=9.2, latency=1056, route="gear", gear="head", slip_at=slip_at)
        try:
            src = self.tmp / "input.wav"
            write_fake_input(src)
            raw = self.tmp / "take-step.npy"
            r = self.run_op("record", timeout=300, input_path=str(src), save_path=str(raw), extra_frames=48_000, warmup=warmup)
        finally:
            self.s.call("sim", slip_at=None)
        take = np.load(raw)
        self.assertEqual(len(take), al.INPUT_V3_FRAMES + 48_000)
        return r["timing"], al.analyze_take(take, timing=r["timing"])[1]

    def test_a_driver_step_during_the_warmup_is_waited_for_and_the_take_is_clean(self):
        timing, rep = self._take_with_step(130_000)          # 2.7 s into the stream, like the Focusrite
        self.assertEqual(timing["reason"], "step_seen")
        self.assertEqual(timing["steps"][0][1], 1)
        self.assertGreater(timing["warmup_s"], 130_000 / 48_000)
        self.assertLess(timing["warmup_s"], 10)
        self.assertTrue(timing["checked"])
        self.assertFalse(timing["moved_in_training"])
        self.assertTrue(rep["ok"], rep["warnings"])
        self.assertNotIn("esr_repaired", rep["warnings"])     # nothing to mend: the step happened before the input
        self.assertLess(rep["replicate_esr"], 0.01)

    def test_a_driver_step_inside_the_take_is_caught(self):
        timing, rep = self._take_with_step(al.WARMUP_MAX + al.WARMUP_GAP + 3_000_000)    # no step in the warm-up: it waits 30 s; then one 60 s into the input
        self.assertEqual(timing["reason"], "max")
        self.assertTrue(timing["moved_in_training"])
        self.assertAlmostEqual(timing["after_input"], 1.0, delta=0.05)
        self.assertIn("timing_step", rep["warnings"])
        self.assertFalse(rep["ok"])

    def test_without_warmup_the_check_after_the_input_still_runs(self):
        # the warm-up is a switch, off by default: a stable interface starts at once. A step on the first replicate is realigned,
        # one inside the training part is still caught by the blips after the input
        timing, rep = self._take_with_step(100_000, warmup=False)                 # 2.1 s into the input: first replicate
        self.assertIsNone(timing.get("reason"))
        self.assertIsNone(timing["before_input"])
        self.assertFalse(timing["moved_in_training"])
        self.assertIn("esr_repaired", rep["warnings"])
        self.assertTrue(rep["ok"], rep["warnings"])
        timing, rep = self._take_with_step(3_000_000, warmup=False)               # 62 s into the input
        self.assertTrue(timing["moved_in_training"])
        self.assertIn("timing_step", rep["warnings"])

    def test_stop_fades_out_and_ends_the_operation(self):
        self.s.call("sim", speed=60, route="loopback")
        self.s.start_op("tone", level_dbfs=-20.0)
        import time
        time.sleep(0.4)
        self.s.stop_op()
        result = self.s.wait_op(10)
        self.assertTrue(result["stopped"])
        self.s.call("sim", speed=0)

    def test_second_operation_while_busy_is_refused(self):
        self.s.call("sim", speed=60, route="loopback")
        self.s.start_op("tone", level_dbfs=-30.0)
        with self.assertRaises(EngineError) as e:
            self.s.start_op("meter")
        self.assertEqual(e.exception.key, "busy")
        self.s.stop_op()
        self.s.wait_op(10)
        self.s.call("sim", speed=0)


if __name__ == "__main__":
    unittest.main()
