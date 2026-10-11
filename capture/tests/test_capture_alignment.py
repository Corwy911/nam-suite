import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _helpers import ROOT, make_take
from capture import alignment as al


class LatencyTests(unittest.TestCase):
    def test_detects_the_true_delay(self):
        for delay in (0, 1, 173, 1056, 5000, 9000):
            with self.subTest(delay=delay):
                r = al.detect_latency(make_take(delay))
                self.assertEqual(r["delay"], delay)
                self.assertEqual(r["recommended"], delay - 1)

    def test_no_response_is_reported(self):
        r = al.detect_latency(make_take(173, blips=False))
        self.assertIsNone(r["delay"])

    def test_too_short_recording(self):
        with self.assertRaises(ValueError):
            al.detect_latency(np.zeros(100_000))


class TrainerEquivalenceTests(unittest.TestCase):
    """The same take through each installed trainer's own `_calibrate_latency_v3` gives the same delay."""

    def _trainer_delay(self, trainer_dir: Path, take: np.ndarray):
        with tempfile.TemporaryDirectory() as tmp:
            npy = Path(tmp) / "take.npy"
            np.save(npy, take)
            out = subprocess.run([sys.executable, str(Path(__file__).with_name("_trainer_latency.py")), str(trainer_dir), str(npy)],
                                 capture_output=True, text=True, timeout=300)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")][-1]
        return json.loads(line)

    def test_matches_each_trainer(self):
        take = make_take(1056)
        mine = al.detect_latency(take)
        ran = 0
        for name in ("official", "reloaded"):
            folder = ROOT / "trainers" / name
            if not folder.exists():
                continue
            with self.subTest(trainer=name):
                theirs = self._trainer_delay(folder, take)
                if "error" in theirs:
                    self.skipTest(f"{name}: {theirs['error']}")
                self.assertEqual(theirs["delays"], [mine["delay"]])
                # the raw delay is identical; each trainer then subtracts its own "safety factor" (official 1, Reloaded 0)
                self.assertIn(theirs["recommended"], (mine["delay"], mine["delay"] - 1))
                ran += 1
        if not ran:
            self.skipTest("no trainer folder")


class CutTests(unittest.TestCase):
    def test_cut_and_pad(self):
        x = np.arange(10, dtype=float)
        self.assertEqual(list(al.cut_and_pad(x, 4, 3)), [3, 4, 5, 6])
        self.assertEqual(list(al.cut_and_pad(x, 4, -2)), [0, 0, 0, 1])
        self.assertEqual(list(al.cut_and_pad(x, 5, 8)), [8, 9, 0, 0, 0])

    def test_click_latency(self):
        rec = np.zeros(100_000)
        for p in (1000, 30_000, 60_000):
            rec[p + 1056] = 0.5
        self.assertEqual(al.click_latency(rec, (1000, 30_000, 60_000)), 1056)
        self.assertIsNone(al.click_latency(np.zeros(1000), (0,), floor=0.01))


class AnalyzeTakeTests(unittest.TestCase):
    def test_aligned_take_is_exact_and_the_trainer_sees_no_delay(self):
        for delay in (173, 1056):
            with self.subTest(delay=delay):
                aligned, rep = al.analyze_take(make_take(delay))
                self.assertEqual(len(aligned), al.INPUT_V3_FRAMES)
                self.assertEqual(rep["latency"]["delay"], delay)
                self.assertEqual(rep["latency"]["cut"], delay - 1)
                self.assertEqual(rep["residual"]["delay"], 1)
                self.assertEqual(rep["residual"]["recommended"], 0)      # what the trainer would apply to the cut take
                self.assertLess(rep["replicate_esr"], 0.01)
                self.assertTrue(rep["ok"], rep["warnings"])

    def test_the_end_of_the_response_is_not_lost(self):
        delay = 5000
        take = make_take(delay)
        aligned, _ = al.analyze_take(take)
        # the last samples of the aligned take are the last samples the input produced (recorded after the input ended)
        np.testing.assert_allclose(aligned[-1000:], take[delay - 1 + al.INPUT_V3_FRAMES - 1000: delay - 1 + al.INPUT_V3_FRAMES])

    def test_missing_response_blocks(self):
        aligned, rep = al.analyze_take(make_take(173, blips=False))
        self.assertIn("no_blip", rep["warnings"])
        self.assertFalse(rep["ok"])
        self.assertEqual(len(aligned), al.INPUT_V3_FRAMES)

    def test_clipping_blocks(self):
        _, rep = al.analyze_take(make_take(173, clip=True))
        self.assertIn("clip", rep["warnings"])
        self.assertFalse(rep["ok"])

    def test_short_recording_is_flagged(self):
        take = make_take(173)[:al.INPUT_V3_FRAMES - 500]
        _, rep = al.analyze_take(take)
        self.assertIn("short_recording", rep["warnings"])
        self.assertFalse(rep["ok"])

    def test_quiet_return_and_weak_blip_warn(self):
        _, rep = al.analyze_take(make_take(173, gain=0.0005, noise_db=-75))
        self.assertTrue({"low_signal", "blip_weak"} & set(rep["warnings"]) or rep["latency"]["delay"] is None, rep)

    def test_replicates_that_disagree_block(self):
        take = make_take(173)
        take[173 + al.INPUT_V3_FRAMES - al.T_VALIDATE:173 + al.INPUT_V3_FRAMES] *= 0.2      # the end differs from the start
        _, rep = al.analyze_take(take)
        self.assertIn("esr_high", rep["warnings"])
        self.assertFalse(rep["ok"])

    def _glitched(self, edit):
        take = make_take(173)
        base = 173 + al.INPUT_V3_FRAMES - al.T_VALIDATE            # where the second replicate starts in the raw recording
        edit(take, base + 52_800, base + 76_800)                    # 1.1 s .. 1.6 s into it
        return al.analyze_take(take)[1]

    def _slipped_start(self, seconds):
        """A take whose first `seconds` run one sample late (the stream still settling), the rest in step."""
        take = make_take(173)
        end = int(seconds * 48_000)
        take[173 + 1:173 + end + 1] = take[173:173 + end].copy()
        return take

    def test_a_one_sample_slip_at_the_start_is_put_back_in_step_and_said(self):
        rep = al.analyze_take(self._slipped_start(4.3))[1]
        self.assertIn("esr_repaired", rep["warnings"])               # the take is fine: the first part of the file is not trained on, it is mended
        self.assertNotIn("esr_glitch", rep["warnings"])
        self.assertTrue(rep["ok"], rep["warnings"])
        self.assertLess(rep["replicate_esr"], 0.01)
        fix = rep["replicate"]["repaired"]
        self.assertEqual(abs(fix["slip"]), 1)
        self.assertGreater(fix["esr_before"], 0.01)

    def test_the_repair_only_touches_the_first_nine_seconds(self):
        aligned_slip, rep = al.analyze_take(self._slipped_start(4.3))
        self.assertIn("esr_repaired", rep["warnings"])
        untouched, _ = al.analyze_take(make_take(173))
        changed = np.nonzero(np.abs(aligned_slip - untouched) > 1e-3)[0]            # (the splice leaves a few samples of difference, never more)
        self.assertLess(changed.max(), al.T_VALIDATE)                # nothing after the first replicate (the blips, the training part, the last replicate)

    def test_a_step_after_the_first_replicate_moves_all_of_it_back(self):
        # what a Focusrite 18i20 did (Trirec, 11 Oct): the step came after the first replicate, so ALL of it ran one sample early
        # against the second (ESR 0.108); moving it as a whole makes the two agree, and nothing from the blips on is touched
        for slip in (1, -1, 2):
            with self.subTest(slip=slip):
                take = make_take(173)
                end = 173 + al.T_VALIDATE + 24_000                     # rep1 and the quiet half second after it
                if slip > 0:
                    take[173:end - slip] = take[173 + slip:end].copy()   # the early part arrives `slip` samples early
                else:
                    take[173 - slip:end] = take[173:end + slip].copy()
                untouched, _ = al.analyze_take(make_take(173))
                aligned, rep = al.analyze_take(take)
                self.assertIn("esr_repaired", rep["warnings"])
                self.assertTrue(rep["ok"], rep["warnings"])
                self.assertLess(rep["replicate_esr"], 0.001)
                self.assertEqual(rep["replicate"]["repaired"]["slip"], slip)
                self.assertGreater(rep["replicate"]["repaired"]["until_s"], 8.9)
                self.assertLess(rep["replicate"]["repaired"]["until_s"], al.FIRST_BLIPS_START / 48_000)
                changed = np.nonzero(np.abs(aligned - untouched) > 1e-3)[0]
                self.assertLess(changed.max(), al.FIRST_BLIPS_START - 9_600)

    def test_noise_is_not_fixed_by_a_shift(self):
        take = make_take(173)
        take[173:173 + al.T_VALIDATE] += np.random.default_rng(3).standard_normal(al.T_VALIDATE) * 0.012     # an amp that hisses
        rep = al.analyze_take(take)[1]
        self.assertNotIn("esr_repaired", rep["warnings"])
        self.assertFalse(rep["ok"])

    def test_a_slip_that_the_repair_cannot_explain_stays_a_blocking_glitch(self):
        def slip(take, start, end):
            take[start + 1:end + 1] = take[start:end].copy()
            take[start + 40_000:start + 41_000] += 0.2               # and something else wrong in the same replicate, with a different shape
        rep = self._glitched(slip)
        self.assertNotIn("esr_repaired", rep["warnings"])
        self.assertFalse(rep["ok"])

    def test_a_short_disturbance_has_a_stretch_but_no_slip(self):
        def pop(take, start, end):
            take[start:end] += np.random.default_rng(5).standard_normal(end - start) * 0.05
        rep = self._glitched(pop)
        self.assertIn("esr_glitch", rep["warnings"])
        self.assertIsNone(rep["replicate"]["slip"])
        self.assertAlmostEqual(rep["replicate"]["stretch"][0], 1.1, delta=0.11)

    def test_trouble_everywhere_is_not_called_a_glitch(self):
        take = make_take(173)
        take[173 + al.INPUT_V3_FRAMES - al.T_VALIDATE:173 + al.INPUT_V3_FRAMES] *= 0.2
        rep = al.analyze_take(take)[1]
        self.assertIsNone(rep["replicate"]["stretch"])
        self.assertNotIn("esr_glitch", rep["warnings"])

    def test_negative_latency_is_padded(self):
        # a recording that is early by 26 samples (some digital gear compensates): the cut goes negative and silence is inserted
        take = make_take(0)
        take = np.concatenate([take[26:], np.zeros(26)])
        aligned, rep = al.analyze_take(take)
        self.assertEqual(rep["latency"]["delay"], -26)
        self.assertLess(rep["latency"]["cut"], 0)
        self.assertEqual(len(aligned), al.INPUT_V3_FRAMES)
        self.assertEqual(rep["residual"]["recommended"], 0)


def _stream_with_blips(blips, step_at=None, step=1, length=None, noise=1e-4, seed=2):
    """A recorded stream: an amp-like ringing response to each blip (delay 317), noise, and optionally the driver's step: from
    `step_at` on everything arrives `step` samples later."""
    rng = np.random.default_rng(seed)
    length = length or max(blips) + 48_000
    y = rng.standard_normal(length) * noise
    ring = np.exp(-np.arange(600) / 80.0) * np.sin(np.arange(600) * 0.9) * 0.3
    for p in blips:
        at = p + 317 + (step if step_at is not None and p >= step_at else 0)
        y[at:at + 600] += ring
    return y


class TimingTests(unittest.TestCase):
    def test_lag_between_two_responses_is_sub_sample_precise(self):
        y = _stream_with_blips([24_000, 48_000], step_at=40_000)
        lag, corr = al.response_lag(y, 24_000, 48_000, al.blip_onset(y, 24_000))
        self.assertAlmostEqual(lag, 1.0, delta=0.05)
        self.assertGreater(corr, 0.99)

    def test_the_warmup_waits_for_the_step_then_lets_the_input_start(self):
        blips = list(range(al.WARMUP_FIRST, al.WARMUP_MAX + 1, al.BLIP_EVERY))
        y = _stream_with_blips(blips, step_at=200_000, length=al.WARMUP_MAX + 200_000)
        w = al.WarmupWatch()
        for pos in range(0, len(y) - 64, 64):                  # fed like the audio callback, block by block
            w.feed(y[pos:pos + 64], pos)
            if w.start is not None:
                break
        self.assertEqual(w.reason, "step_seen")
        self.assertEqual(w.steps, [[round(216_000 / 48_000, 2), 1]])              # seen at the first blip after it
        self.assertGreaterEqual(w.start, 216_000 + 2 * al.BLIP_EVERY)           # two agreeing blips after it
        self.assertLess(w.start, 216_000 + 3 * al.BLIP_EVERY + al.RESP_SPAN + al.WARMUP_GAP + 128)

    def test_no_step_means_waiting_until_the_maximum(self):
        blips = list(range(al.WARMUP_FIRST, al.WARMUP_MAX + 1, al.BLIP_EVERY))
        y = _stream_with_blips(blips, length=al.WARMUP_MAX + 100_000)
        w = al.WarmupWatch()
        for pos in range(0, len(y) - 4_800, 4_800):
            w.feed(y[pos:pos + 4_800], pos)
        self.assertEqual(w.reason, "max")
        self.assertEqual(w.steps, [])

    def test_nothing_coming_back_starts_at_the_minimum(self):
        w = al.WarmupWatch()
        y = np.random.default_rng(1).standard_normal(400_000) * 1e-4
        for pos in range(0, len(y) - 4_800, 4_800):
            w.feed(y[pos:pos + 4_800], pos)
        self.assertEqual(w.reason, "no_response")
        self.assertEqual(w.start, al.WARMUP_MIN)

    def test_a_step_inside_the_training_part_is_found(self):
        start, n = 480_000, al.INPUT_V3_FRAMES
        warm = [24_000, 48_000, 72_000]
        tail = [start + n + 48_000 + al.TAIL_FIRST + k * al.BLIP_EVERY for k in range(al.TAIL_BLIPS)]
        blips = warm + [start + p for p in al.BLIP_LOCATIONS] + tail
        for step_at, moved, before in ((100_000, False, -1.0), (start + 2_000_000, True, 0.0), (start + al.BLIP_LOCATIONS[0] + 10_000, True, 0.0)):
            with self.subTest(step_at=step_at):
                y = _stream_with_blips(blips, step_at=step_at, length=tail[-1] + 40_000)
                tm = al.timing_check(y, start, n, warm, tail)
                self.assertTrue(tm["checked"])
                self.assertEqual(tm["moved_in_training"], moved)
                self.assertAlmostEqual(tm["before_input"], before, delta=0.05)
                rep = al.analyze_take(make_take(173), timing=tm)[1]
                self.assertEqual("timing_step" in rep["warnings"], moved)
                self.assertEqual(rep["ok"], not moved)
                self.assertEqual(rep["timing"], tm)


if __name__ == "__main__":
    unittest.main()
