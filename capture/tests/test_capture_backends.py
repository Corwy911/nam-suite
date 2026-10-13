"""The PortAudio backends (ASIO / Core Audio) against a stand-in for `sounddevice`: no audio hardware needed."""

import types
import unittest
from unittest import mock

import numpy as np

import _helpers  # noqa: F401  (puts nam-web on sys.path)
from capture import backends
from capture.backends import AsioBackend, AudioError, CoreAudioBackend


class FakeSettings:
    def __init__(self, **kw):
        self.kw = kw


class FakeStream:
    """A duplex stream whose callback is called from a thread-less loop by `__enter__`: enough for the open() probe."""

    latency = (0.004, 0.005)
    blocksize = 0

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSd:
    def __init__(self, devices, apis=("Core Audio",)):
        self._devices = devices
        self._apis = [{"name": a} for a in apis]
        self.restarts = 0
        self.checked = []

    def query_hostapis(self):
        return self._apis

    def query_devices(self, index=None):
        return self._devices if index is None else self._devices[index]

    def CoreAudioSettings(self, **kw):
        return FakeSettings(**kw)

    def AsioSettings(self, **kw):
        return FakeSettings(**kw)

    def check_input_settings(self, **kw):
        self.checked.append(("in", kw))

    def check_output_settings(self, **kw):
        self.checked.append(("out", kw))

    def _terminate(self):
        self.restarts += 1

    def _initialize(self):
        pass


def dev(name, n_in, n_out, rate=48000.0, api=0):
    return {"name": name, "hostapi": api, "max_input_channels": n_in, "max_output_channels": n_out, "default_samplerate": rate}


def make(cls, devices, apis=("Core Audio",)):
    backend = cls.__new__(cls)                       # skip __init__: it imports the real sounddevice
    backend.sd = FakeSd(devices, apis)
    backend.stream, backend.device = None, None
    backend.in_ch, backend.out_ch, backend.rate, backend.blocksize = [], [], 48_000, 0
    return backend


class CoreAudioTests(unittest.TestCase):
    DEVICES = [dev("Monitor", 0, 2), dev("Fireface", 18, 18), dev("Mic", 1, 0), dev("Odd rate", 2, 2, rate=44100.0)]

    def test_lists_only_duplex_devices(self):
        b = make(CoreAudioBackend, self.DEVICES)
        found = b.list_devices()
        self.assertEqual([d["name"] for d in found], ["Fireface", "Odd rate"])
        self.assertEqual(found[0]["id"], "coreaudio:1")
        self.assertEqual((found[0]["in_channels"], found[0]["out_channels"], found[0]["kind"]), (18, 18, "coreaudio"))

    def test_software_devices_come_after_the_real_interface(self):
        b = make(CoreAudioBackend, [dev("BlackHole 2ch", 2, 2), dev("Microsoft Teams Audio", 1, 1), dev("Fireface UC", 18, 18)])
        self.assertEqual([d["name"] for d in b.list_devices()], ["Fireface UC", "BlackHole 2ch", "Microsoft Teams Audio"])

    def test_other_host_apis_are_ignored(self):
        b = make(CoreAudioBackend, [dev("Fireface", 18, 18, api=1)], apis=("Core Audio", "Something else"))
        self.assertEqual(b.list_devices(), [])

    def test_the_list_is_refreshed_only_while_nothing_is_open(self):
        b = make(CoreAudioBackend, self.DEVICES)
        b.list_devices()
        self.assertEqual(b.sd.restarts, 1)
        b.device = 1                                     # a device is open now
        b.list_devices()
        self.assertEqual(b.sd.restarts, 1)

    def test_a_device_at_another_rate_is_refused_not_resampled(self):
        b = make(CoreAudioBackend, self.DEVICES)
        with self.assertRaises(AudioError) as e:
            b.open("coreaudio:3", [0], [0], 48_000)
        self.assertEqual(e.exception.key, "rate_mismatch")
        self.assertIn("44100", e.exception.detail)
        self.assertEqual(b.sd.checked, [])               # refused before anything was asked of the driver
        self.assertIsNone(b.device)

    def test_channels_beyond_the_device_are_refused(self):
        b = make(CoreAudioBackend, self.DEVICES)
        for ins, outs in (([18], [0]), ([0], [18]), ([], [0])):
            with self.assertRaises(AudioError) as e:
                b.open("coreaudio:1", ins, outs, 48_000)
            self.assertEqual(e.exception.key, "no_channels" if not ins else "bad_channels")

    def test_channels_are_selected_with_a_channel_map(self):
        b = make(CoreAudioBackend, self.DEVICES)
        ain, aout = b._settings(self.DEVICES[1], [2], [5])             # the Fireface: 18 outputs
        self.assertEqual(ain.kw, {"channel_map": [2]})                # input: the device channels wanted
        self.assertEqual(aout.kw, {"channel_map": [-1] * 5 + [0] + [-1] * 12})   # output: one slot per device channel, stream channel 0 at #5

    def test_output_map_follows_the_documented_example(self):
        # sounddevice's own example: a 6-output device, stream channels 0 and 1 to the third and fifth outputs
        self.assertEqual(CoreAudioBackend.output_map(6, [2, 4]), [-1, -1, 0, -1, 1, -1])

    def test_a_failed_open_leaves_the_backend_closed(self):
        b = make(CoreAudioBackend, self.DEVICES)

        def boom(**kw):
            raise RuntimeError("no stream")

        b.sd.Stream = boom
        with self.assertRaises(AudioError) as e:
            b.open("coreaudio:1", [0], [0], 48_000)
        self.assertEqual(e.exception.key, "open_failed")
        self.assertIsNone(b.device)


class AsioBufferTests(unittest.TestCase):
    """One buffer size for the whole program: what the page asks for (0 = what the driver has now), rounded to what the driver accepts, is what PortAudio
    is told AND what the native engine is handed later - so no hand-over moves the driver's buffer (seen on a Scarlett 18i20)."""
    RANGE = (16, 1024, 96, 16)                       # min, max, preferred (= the control panel's value), granularity

    def backend(self, rng=RANGE):
        b = make(AsioBackend, [dev("Focusrite USB ASIO", 20, 20)], apis=("ASIO",))
        b._driver_range = lambda index: rng
        return b

    def test_automatic_keeps_the_size_the_driver_has_now(self):
        self.assertEqual(self.backend().resolve_blocksize(0, 0), 96)
        self.assertEqual(self.backend((16, 1024, 256, 16)).resolve_blocksize(0, 0), 256)      # the user set 256 in the control panel: so it stays

    def test_a_request_is_rounded_to_a_valid_size_and_clamped(self):
        b = self.backend()
        self.assertEqual([b.resolve_blocksize(0, n) for n in (64, 100, 120, 5000, 8)], [64, 96, 128, 1024, 16])
        self.assertEqual(self.backend((32, 4096, 64, -1)).resolve_blocksize(0, 300), 256)     # granularity -1: powers of two

    def test_no_answer_from_the_driver_leaves_the_request_alone(self):
        self.assertEqual(self.backend(None).resolve_blocksize(0, 96), 96)
        self.assertEqual(self.backend(None).resolve_blocksize(0, 0), 0)

    def test_a_driver_still_settling_is_asked_again(self):
        answers = [(64, 64, 64, 16), (64, 64, 64, 16), (16, 1024, 96, 16)]                 # min == max: the transient answer seen right after a close
        b = make(AsioBackend, [dev("Focusrite USB ASIO", 20, 20)], apis=("ASIO",))
        b._driver_range = lambda index: answers.pop(0) if len(answers) > 1 else answers[0]
        with mock.patch.object(backends.time, "sleep"):
            self.assertEqual(b.resolve_blocksize(0, 0), 96)

    def test_portaudio_is_told_exactly_that_size(self):
        b = self.backend()
        b._probe = lambda rate, blocksize: None
        b.open("asio:0", [0], [0], 48_000, 100)                               # asked 100: the driver takes 96
        self.assertEqual(b.blocksize, 96)
        seen = {}
        b.sd.Stream = lambda **kw: seen.update(kw) or FakeStream(**kw)
        b._make_stream(lambda *a: None)
        self.assertEqual(seen["blocksize"], 96)
        self.assertAlmostEqual(seen["latency"], 96 / 48_000)                  # one buffer of buffering target, not the word "low"
        b.open("asio:0", [0], [0], 48_000, 0)                                  # automatic: the driver's own size
        self.assertEqual(b.blocksize, 96)

    def test_core_audio_keeps_the_requested_size(self):
        b = make(CoreAudioBackend, [dev("Fireface", 18, 18)])
        self.assertEqual(b.resolve_blocksize(0, 128), 128)
        self.assertEqual(b._suggested_latency(), "low")             # until a size is set on the backend (open)


class CoreAudioBufferTests(unittest.TestCase):
    """The buffer chosen on the page has to reach the driver on macOS too (1083 samples of round trip "at 64" on a Fireface UC whose own panel said 512:
    PortAudio does not touch the device's buffer unless told it may), and the page has to show what the driver really has, not what was asked."""
    DEVICES = [dev("Fireface", 18, 18)]

    def test_an_explicit_buffer_lets_portaudio_set_exactly_that_size(self):
        b = make(CoreAudioBackend, self.DEVICES)
        b._probe = lambda rate, blocksize: None
        b.open("coreaudio:0", [0], [0], 48_000, 64)
        self.assertTrue(all(s.kw.get("change_device_parameters") for s in b._extra))
        self.assertEqual(b.blocksize, 64)
        seen = {}
        b.sd.Stream = lambda **kw: seen.update(kw) or FakeStream(**kw)
        b._make_stream(lambda *a: None)
        self.assertEqual(seen["blocksize"], 64)
        self.assertAlmostEqual(seen["latency"], 64 / 48_000)       # one buffer of buffering target, not the word "low" (PortAudio then picked 448 for 64)

    def test_automatic_keeps_the_buffer_the_interface_has_now(self):
        b = make(CoreAudioBackend, self.DEVICES)
        b._probe = lambda rate, blocksize: None
        with mock.patch.object(backends.coreaudio_info, "buffer_frames", return_value=512):
            b.open("coreaudio:0", [0], [0], 48_000, 0)
        self.assertEqual(b.blocksize, 512)                          # handed on as an explicit size, so PortAudio does not move it
        self.assertAlmostEqual(b._suggested_latency(), 512 / 48_000)

    def test_automatic_with_no_answer_from_core_audio_leaves_everything_to_portaudio(self):
        b = make(CoreAudioBackend, self.DEVICES)
        b._probe = lambda rate, blocksize: None
        with mock.patch.object(backends.coreaudio_info, "buffer_frames", return_value=None):
            b.open("coreaudio:0", [0], [0], 48_000, 0)
        self.assertEqual(b.blocksize, 0)
        self.assertEqual(b._suggested_latency(), "low")
        self.assertTrue(all("change_device_parameters" not in s.kw for s in b._extra))

    def _open_with_real_probe(self, granted, asked=64):
        b = make(CoreAudioBackend, self.DEVICES)
        streams = []

        def stream(**kw):
            s = FakeStream(**kw)
            streams.append(s)
            return s

        def tick(seconds):                                   # the interface "delivers" blocks while the probe waits
            for s in streams:
                for _ in range(6):
                    s.kw["callback"](None, np.zeros((64, 1), dtype=np.float32), 64, None, None)

        b.sd.Stream = stream
        with mock.patch.object(backends.time, "sleep", tick), mock.patch.object(backends.coreaudio_info, "buffer_frames", return_value=granted):
            b.open("coreaudio:0", [0], [0], 48_000, asked)
        return b

    def test_the_page_gets_the_buffer_the_driver_really_has(self):
        self.assertEqual(self._open_with_real_probe(512).info()["blocksize"], 512)      # asked 64, the panel still says 512: say so
        self.assertEqual(self._open_with_real_probe(64).info()["blocksize"], 64)

    def test_without_an_answer_from_core_audio_the_request_is_shown(self):
        self.assertEqual(self._open_with_real_probe(None).info()["blocksize"], 64)

    def test_reading_an_unknown_device_gives_none(self):
        from capture import coreaudio_info
        self.assertIsNone(coreaudio_info.buffer_frames("no such device"))
        self.assertIsNone(coreaudio_info.buffer_frames(""))


class PlatformChoiceTests(unittest.TestCase):
    def test_asio_backend_keeps_listing_asio_devices(self):
        b = make(AsioBackend, [dev("ASIO Fireface", 2, 2), dev("Speakers", 0, 2)], apis=("ASIO",))
        found = b.list_devices()
        self.assertEqual([(d["id"], d["kind"]) for d in found], [("asio:0", "asio"), ("asio:1", "asio")])

    def test_real_backend_class_follows_the_platform(self):
        with mock.patch.object(backends, "sys", types.SimpleNamespace(platform="win32")):
            self.assertIs(backends.real_backend_class(), AsioBackend)
        with mock.patch.object(backends, "sys", types.SimpleNamespace(platform="darwin")):
            self.assertIs(backends.real_backend_class(), CoreAudioBackend)


if __name__ == "__main__":
    unittest.main()
