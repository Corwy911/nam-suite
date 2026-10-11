"""Loaded (via PYTHONPATH) only by the worker processes that selftest.py starts.

The trainers refuse any input file that is not one of the official NAM test signals, and
the self-test trains on synthetic audio. When `nam.train.core` gets imported this hook
makes it accept the synthetic input as the oldest input version, so that the rest of the
real pipeline (data loading, training, export) runs unchanged. It never touches the
production workers: they don't have this folder on PYTHONPATH.
"""

import importlib.abc
import importlib.util
import sys


class _PatchDetection(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "nam.train.core":
            return None
        sys.meta_path.remove(self)  # let the normal machinery find the real spec
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            module._detect_input_version = lambda input_path, verbose=False: (module._Version(1, 0, 0), False)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PatchDetection())
