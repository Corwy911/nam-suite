"""The tone generator: a level change while it plays must glide (no click) and the phase must stay continuous."""

import unittest

import numpy as np

import _helpers  # noqa: F401  (puts nam-web on sys.path)
from capture import calibration as cal
from capture.generators import ToneGen


def play(gen, blocks, n):
    return np.concatenate([gen(i * n, n)[:, 0] for i in range(blocks)])


class ToneGenLevelTests(unittest.TestCase):
    def test_a_steady_tone_is_the_plain_sine(self):
        gen = ToneGen(-6.0, 1, 0)
        x = play(gen, 40, 128)
        expected = cal.from_db(-6.0) * np.sin(2 * np.pi * 1000 * np.arange(len(x)) / 48000)
        n = 2400                                            # past the 50 ms fade-in
        self.assertLess(np.max(np.abs(x[n:] - expected[n:])), 1e-6)

    def test_half_a_db_step_reaches_its_level_and_does_not_click(self):
        gen = ToneGen(-20.0, 1, 0)
        a = play(gen, 40, 128)
        gen.set_level(-19.5)
        b = np.concatenate([gen((40 + i) * 128, 128)[:, 0] for i in range(40)])
        x = np.concatenate([a, b])
        self.assertAlmostEqual(cal.to_db(np.max(np.abs(b[-480:]))), -19.5, delta=0.02)
        # no sample-to-sample jump beyond what a 1 kHz sine of this size does anyway
        self.assertLess(np.max(np.abs(np.diff(x))), 2 * np.pi * 1000 / 48000 * cal.from_db(-19.5) * 1.05)

    def test_a_large_step_glides_over_about_a_hundred_milliseconds(self):
        gen = ToneGen(-60.0, 1, 0)
        play(gen, 40, 128)
        gen.set_level(0.0)
        x = np.concatenate([gen((40 + i) * 128, 128)[:, 0] for i in range(60)])
        self.assertLess(np.max(np.abs(x[:480])), 0.15)     # the first 10 ms are still far from full scale
        self.assertAlmostEqual(float(np.max(np.abs(x[-480:]))), 1.0, delta=0.01)
        self.assertLess(np.max(np.abs(np.diff(x))), 0.2)


if __name__ == "__main__":
    unittest.main()
