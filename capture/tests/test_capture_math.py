import math
import unittest
from pathlib import Path

import numpy as np

import _helpers  # noqa: F401  (sets sys.path)
from capture import calibration as cal


class ConversionTests(unittest.TestCase):
    def test_guide_examples(self):
        # NAM calibration guide: 6.40 V RMS = 18.3 dBu ; 18.3 + (-6.50) - (-0.5) = 12.3 dBu
        self.assertAlmostEqual(cal.vrms_to_dbu(6.40), 18.3, places=1)
        self.assertAlmostEqual(cal.input_level_dbu(18.3, -6.5, -0.5), 12.3, places=6)

    def test_dbu_roundtrip_and_reference(self):
        self.assertAlmostEqual(cal.vrms_to_dbu(0.7746), 0.0, places=9)
        for v in (0.001, 0.25, 1.0, 6.4, 20.0):
            self.assertAlmostEqual(cal.dbu_to_vrms(cal.vrms_to_dbu(v)), v, places=9)
        with self.assertRaises(ValueError):
            cal.vrms_to_dbu(0)

    def test_output_level_accounts_for_the_level_sent(self):
        # the same interface read at 0 dBFS or at -20 dBFS gives the same Co
        co = 12.0
        v_full = cal.dbu_to_vrms(co)
        v_low = cal.dbu_to_vrms(co - 20.0)
        self.assertAlmostEqual(cal.output_level_dbu(v_full, 0.0), co, places=9)
        self.assertAlmostEqual(cal.output_level_dbu(v_low, -20.0), co, places=9)

    def test_unity_gain(self):
        co, ci = 18.3, 10.0        # hi-z reads 0 dBFS at +10 dBu, the DAC gives +18.3 dBu at full scale
        g = cal.unity_gain_db(ci, co)
        self.assertAlmostEqual(g, -8.3, places=9)
        x = 4.0                     # a guitar peak of 4 dBu arrives at the amp as 4 dBu
        self.assertAlmostEqual((x - ci) + g + co, x, places=9)

    def test_loopback_recovers_input_level(self):
        co, ci_true = 18.3, 9.4
        sent = -12.0
        read = (co + sent) - ci_true                    # what an input with full-scale ci_true would read
        self.assertAlmostEqual(cal.input_level_dbu(co, sent, read), ci_true, places=9)


class NamGainTests(unittest.TestCase):
    """The gains around a NAM model, checked against what a model IS: its input is calibrated so that 0 dBFS = input_level_dbu
    and its output so that 0 dBFS = output_level_dbu (trainer metadata). So the physical test is a capture of a plain cable."""

    def test_a_transparent_capture_plays_back_at_unity_gain(self):
        co_cap, ci_cap = 11.1, 3.8             # the interface when the capture was made
        model_in, model_out = co_cap, ci_cap   # what the capture writes into the .nam
        # a cable gives back at the input exactly what the output sent, so the model learned y = x + (model_in - model_out) dBFS
        model_gain = model_in - model_out
        co, ci_guitar = 12.3, 5.2              # another day, another interface setting
        pre = cal.nam_input_gain_db(model_in, ci_guitar)
        post = cal.nam_output_gain_db(model_out, co)
        for x_dbu in (-20.0, -6.0, 0.0, 4.0):  # the guitar's analog level; the chain must give the same level back
            digital = (x_dbu - ci_guitar) + pre + model_gain + post
            self.assertAlmostEqual(digital + co, x_dbu, places=9)

    def test_the_two_legs_of_the_player_end_on_the_same_level(self):
        g = cal.player_gains(ci_guitar=3.77, ci_return=5.5, co=11.09, model_in=11.09, model_out=5.5)
        self.assertAlmostEqual(g["feed"], 3.77 - 11.09, places=9)       # the guitar to the gear: the monitor's unity gain
        # a gear that is exactly a cable (return = what the reamp sent) and its transparent model: same level on both legs
        x = 0.0                                                          # guitar level, dBu
        gear_out_dbu = x                                                 # the cable: what it gets, it returns
        leg_a = (gear_out_dbu - 5.5) + g["amp"] + 11.09                  # return input -> monitor
        leg_b = (x - 3.77) + g["pre"] + (11.09 - 5.5) + g["post"] + 11.09
        self.assertAlmostEqual(leg_a, x, places=9)
        self.assertAlmostEqual(leg_b, x, places=9)

    def test_missing_numbers_give_none_not_zero(self):
        g = cal.player_gains(ci_guitar=3.77, ci_return=None, co=11.09)
        self.assertIsNone(g["amp"]); self.assertIsNone(g["pre"]); self.assertIsNone(g["post"])
        self.assertAlmostEqual(g["feed"], 3.77 - 11.09, places=9)


class ParseTests(unittest.TestCase):
    def test_parse_vrms(self):
        self.assertEqual(cal.parse_vrms("6.40"), 6.40)
        self.assertEqual(cal.parse_vrms("6,40 V"), 6.40)
        self.assertAlmostEqual(cal.parse_vrms("6400 mV"), 6.4)
        self.assertIsNone(cal.parse_vrms("abc"))
        self.assertIsNone(cal.parse_vrms("0"))
        self.assertIsNone(cal.parse_vrms("5000"))


class ToneTests(unittest.TestCase):
    def test_sine_is_exactly_1000_cycles_with_zero_phase(self):
        s = cal.sine()
        self.assertEqual(len(s), 48_000)
        self.assertEqual(s[0], 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(s))), 1.0, places=9)
        self.assertAlmostEqual(float(s[12]), 1.0, places=9)   # quarter of a 48-sample cycle

    def test_sine_matches_the_guide_file_when_available(self):
        path = Path(r"C:\Users\Roman\Downloads\sine_1k.wav")
        if not path.exists():
            self.skipTest("sine_1k.wav not on this PC")
        import soundfile as sf
        ref, sr = sf.read(str(path), dtype="float64")
        self.assertEqual(sr, 48_000)
        self.assertLess(float(np.max(np.abs(cal.sine() - ref))), 3 / 2 ** 23)   # a couple of 24-bit LSB

    def test_measure_tone_levels(self):
        for peak in (0.0, -6.0, -20.0, -47.3):
            m = cal.measure_tone(cal.sine(1.5, peak))
            self.assertAlmostEqual(m["peak_dbfs"], peak, places=3)
            self.assertAlmostEqual(m["rms_dbfs"], peak - 3.0103, places=3)
            self.assertEqual(m["clipped"], 0)

    def test_measure_tone_ignores_dc_and_reports_noise(self):
        rng = np.random.default_rng(3)
        x = cal.sine(2.0, -10.0) + 0.02 + rng.standard_normal(96_000) * 10 ** (-60 / 20)
        m = cal.measure_tone(x)
        self.assertAlmostEqual(m["peak_dbfs"], -10.0, places=1)
        self.assertAlmostEqual(m["dc"], 0.02, places=3)
        self.assertAlmostEqual(m["residual_dbfs"], -60.0, delta=0.5)

    def test_measure_tone_flags_clipping(self):
        x = np.clip(cal.sine(1.0, 6.0), -1.0, 1.0)
        m = cal.measure_tone(x)
        self.assertGreater(m["clipped"], 0)
        self.assertEqual(cal.judge_level(m["sample_peak_dbfs"], m["clipped"]), "clip")

    def test_measure_tone_needs_a_cycle(self):
        with self.assertRaises(ValueError):
            cal.measure_tone(np.zeros(10))


class SearchTests(unittest.TestCase):
    def test_suggest_send_level(self):
        # aim the reading at -9 dBFS: it read -30 at a send of -40, so send 21 dB more
        self.assertAlmostEqual(cal.suggest_send_level(-40.0, -30.0), -19.0)
        self.assertEqual(cal.suggest_send_level(-10.0, 0.0, clipped=True), -22.0)      # clipped: back off 12 dB
        self.assertEqual(cal.suggest_send_level(-50.0, -120.0), -30.0)                 # nothing arrives: try 20 dB louder
        self.assertEqual(cal.suggest_send_level(-1.0, -40.0), -0.5)                    # never above the ceiling
        self.assertEqual(cal.suggest_send_level(-58.0, -5.0), -60.0)                   # nor below the floor

    def test_judge_level(self):
        self.assertEqual(cal.judge_level(-0.1), "clip")
        self.assertEqual(cal.judge_level(-2.0), "hot")
        self.assertEqual(cal.judge_level(-12.0), "ok")
        self.assertEqual(cal.judge_level(-40.0), "low")
        self.assertEqual(cal.judge_level(-80.0), "silent")


if __name__ == "__main__":
    unittest.main()


class LinearityMathTests(unittest.TestCase):
    def points(self, cis, reads=(-40.0, -30.0, -20.0, -12.0, -6.0), clipped=0):
        return [{"level_dbfs": r - 3.0, "peak_dbfs": r, "ci_dbu": c, "clipped": clipped} for r, c in zip(reads, cis)]

    def test_levels_land_the_readings_on_the_targets(self):
        levels = cal.linearity_levels(11.09, 3.77)
        self.assertEqual(levels, [-47.5, -37.5, -27.5, -19.5, -13.5])
        for level, target in zip(levels, (-40.0, -30.0, -20.0, -12.0, -6.0)):
            self.assertAlmostEqual(11.09 + level - 3.77, target, delta=0.26)

    def test_levels_outside_the_dac_range_are_dropped(self):
        self.assertEqual(cal.linearity_levels(0.0, 30.0, targets=(-6.0,)), [])          # would need more than 0 dBFS
        self.assertEqual(cal.linearity_levels(30.0, 0.0, targets=(-40.0,)), [])         # would need less than -60 dBFS

    def test_a_linear_input_is_ok(self):
        v = cal.linearity_verdict(self.points([3.77, 3.76, 3.77, 3.78, 3.77]))
        self.assertEqual(v["status"], "ok")
        self.assertLessEqual(v["max_dev_db"], 0.05)
        self.assertEqual(v["span_db"], 34.0)

    def test_a_drifting_input_is_flagged(self):
        self.assertEqual(cal.linearity_verdict(self.points([3.77, 3.8, 3.9, 4.0, 3.77]))["status"], "warn")
        self.assertEqual(cal.linearity_verdict(self.points([3.77, 3.8, 3.9, 4.5, 5.6]))["status"], "bad")

    def test_clipping_is_never_ok(self):
        self.assertEqual(cal.linearity_verdict(self.points([3.77] * 5, clipped=1))["status"], "bad")

    def test_noise_limited_readings_are_left_out(self):
        pts = self.points([3.77, 3.77, 3.77, 3.77, 3.77], reads=(-70.0, -60.0, -30.0, -20.0, -6.0))
        pts[0]["ci_dbu"] = 9.0                      # a reading at the noise floor: nonsense, but not held against the input
        v = cal.linearity_verdict(pts)
        self.assertEqual(v["status"], "ok")
        self.assertIsNone(v["deviations"][0])

    def test_too_few_usable_readings_is_not_a_pass(self):
        self.assertEqual(cal.linearity_verdict(self.points([3.77, 3.77], reads=(-30.0, -20.0)))["status"], "bad")

    def test_suggested_trim_brings_the_peak_to_minus_six(self):
        self.assertEqual(cal.suggest_trim_db(-2.0), -4.0)
        self.assertEqual(cal.suggest_trim_db(-2.7), -3.5)        # 3.3 dB over: rounded up to the next half dB
        self.assertEqual(cal.suggest_trim_db(0.0), -6.0)          # clipped: the real peak is unknown, assume the worst known
