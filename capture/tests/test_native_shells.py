"""The two native shells (Core Audio / Swift on the Mac, ASIO / C++ on Windows) speak the same HTTP contract to the Python server: this keeps them
from drifting apart. It reads the route tables of packaging/ControlServer.swift and packaging/windows/ControlServerWin.cpp (nothing is compiled
or run) and compares them, and checks that every route the Python side calls exists in both."""

import re
import unittest

from _helpers import ROOT, WEB

SWIFT = (ROOT / "packaging" / "ControlServer.swift").read_text(encoding="utf-8")
CPP = (ROOT / "packaging" / "windows" / "ControlServerWin.cpp").read_text(encoding="utf-8")


def swift_routes():
    return set(re.findall(r'case \("(GET|POST|DELETE)", "(/[a-z_/]+)"\)', SWIFT))


def cpp_routes():
    return set(re.findall(r'method == "(GET|POST|DELETE)" && path == "(/[a-z_/]+)"', CPP))


def python_calls():
    """Every ("POST", "/route") the capture_api modules send to the native engine."""
    calls = set()
    for f in (WEB / "capture_api").glob("*.py"):
        for method, path in re.findall(r'_native_call\(\s*"(GET|POST|DELETE)",\s*"(/[a-z_/]+)"', f.read_text(encoding="utf-8")):
            calls.add((method, path))
    return calls


class NativeShellContractTests(unittest.TestCase):
    def test_both_shells_have_the_same_routes(self):
        swift, cpp = swift_routes(), cpp_routes()
        self.assertTrue(len(swift) >= 19 and len(cpp) >= 19, (len(swift), len(cpp)))        # the patterns still find the tables
        self.assertEqual(sorted(swift - cpp), [], "routes only in ControlServer.swift")
        self.assertEqual(sorted(cpp - swift), [], "routes only in ControlServerWin.cpp")

    def test_every_route_python_calls_exists_in_both_shells(self):
        calls = python_calls()
        self.assertIn(("POST", "/open_player"), calls)                                      # the patterns do see the player's calls
        self.assertEqual(sorted(calls - swift_routes()), [])
        self.assertEqual(sorted(calls - cpp_routes()), [])

    def test_both_shells_know_the_player_state_fields(self):
        for source in (SWIFT, CPP):
            for field in ("player", "leg", "a_peak_dbfs", "b_peak_dbfs", "ir_a_loaded"):
                self.assertIn(f'"{field}"', source)


if __name__ == "__main__":
    unittest.main()
