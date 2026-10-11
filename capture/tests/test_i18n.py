"""static/i18n.json: Italian and English have the same keys and placeholders, and every key the code and the pages use exists."""

import json
import re
import unittest

from _helpers import WEB

CATALOG = json.loads((WEB / "static" / "i18n.json").read_text(encoding="utf-8"))
IT, EN = CATALOG["it"], CATALOG["en"]
PREFIXES = r"(?:srv|cap)"

# keys built at run time from a variable part: the members that can occur
DYNAMIC = (
    {f"cap.in.judge.{j}" for j in ("ok", "hot", "low", "silent", "clip")}
    | {f"cap.warn.{w}" for w in ("no_blip", "blip_at_limit", "short_recording", "clip", "hot", "low_signal", "blip_weak", "esr_high",
                                  "xruns", "stale_input_cal", "clip_live", "xruns_live")}
    | {f"cap.err.{k}" for k in ("devices_failed", "no_channels", "bad_channels", "rate_refused", "open_failed", "stalled", "stream_failed",
                                 "busy", "not_open", "input_format", "engine_exited", "timeout", "internal", "no_device", "unknown_command")}
)
NOT_KEYS = {"upd.some_key", "job.json"}       # a docstring example / a file name that only looks like a key


class CatalogTests(unittest.TestCase):
    def test_same_keys_in_both_languages(self):
        self.assertEqual(sorted(set(IT) ^ set(EN)), [])

    def test_same_placeholders(self):
        ph = lambda s: sorted(set(re.findall(r"\{(\w+)\}", s)))
        for key in IT:
            with self.subTest(key=key):
                self.assertEqual(ph(IT[key]), ph(EN[key]))
                self.assertTrue(IT[key].strip() and EN[key].strip())

    def test_every_used_key_exists(self):
        used = {}
        files = list(WEB.glob("*.py")) + list((WEB / "capture").glob("*.py")) + list((WEB / "capture_api").glob("*.py")) + list((WEB / "static").glob("*.html")) + list((WEB / "static").glob("*.js"))
        for f in files:
            for m in re.finditer(r"""["'](%s\.[a-z0-9_.]+)["']""" % PREFIXES, f.read_text(encoding="utf-8")):
                key = m.group(1)
                if not key.endswith(".") and key not in NOT_KEYS:
                    used.setdefault(key, set()).add(f.name)
        missing = {k: sorted(v) for k, v in used.items() if k not in IT}
        self.assertEqual(missing, {})
        self.assertEqual(sorted(DYNAMIC - set(IT)), [])


if __name__ == "__main__":
    unittest.main()
