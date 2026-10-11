"""Catalogue of the trainers/architectures offered in the GUI dropdown.

Both trainers are plain folders under ../trainers, each holding an unmodified copy of a
`nam` package plus a `source.json` describing where it came from (see updater.py, which
replaces them safely when the upstream repos publish a new release):

* "official": neural-amp-modeler (sdatkinson) -> ../trainers/official. If the folder is
  missing the worker falls back to the copy pip installed in the venv.
* "reloaded:<architecture>": NAM Trainer Reloaded (SlamminMofo) -> ../trainers/reloaded.

The Reloaded architecture list comes from ../trainers/reloaded/catalog.json, generated
from the package itself by selftest.py whenever a release is installed (so new
architectures show up on their own). When that file is missing or invalid the static
tables below are used; they mirror `_ARCHITECTURE_ORDER` of the Reloaded 1.0.1 desktop
GUI, with the learning-rate / decay it recommends for the exponential scheduler at 300
epochs. Architecture strings must match the values of `nam.train.core.Architecture`.
"""

import json
from pathlib import Path

import hardware
import paths
from i18n import tr

OFFICIAL_ID = "official"
RELOADED_PREFIX = "reloaded:"
TRAINERS_DIR = paths.TRAINERS_DIR
OFFICIAL_PATH = TRAINERS_DIR / "official"
RELOADED_PATH = TRAINERS_DIR / "reloaded"
TRAINER_PATHS = {"official": OFFICIAL_PATH, "reloaded": RELOADED_PATH}

# ------------------------------------------------------------- installed version


def installed_info(name: str) -> dict:
    """Contents of trainers/<name>/source.json ({} if absent or unreadable)."""
    try:
        return json.loads((TRAINER_PATHS[name] / "source.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return {}


def trainer_version(backend: str) -> str:
    return installed_info(backend).get("version", "")


def official_label() -> str:
    """Language-neutral label stored in the job state; the page shows its own translated one."""
    version = trainer_version("official")
    return f"NAM {version}" if version else "NAM"


def reloaded_available() -> bool:
    return (RELOADED_PATH / "nam" / "__init__.py").exists()


# ------------------------------------------------------------- Reloaded catalogue

# Dropdown group names and hints are texts in i18n.json: trainer.group.<id>.name / .hint
GROUP_ORDER = ["a2", "light", "extended", "lstm"]

# (architecture value, label, recommended lr, recommended lr decay)
_A2 = [
    ("A2 Full+Lite", "A2 Full+Lite", 0.004, 0.006),
    ("A2 Complex+Lite", "A2 Complex+Lite", 0.003, 0.005),
    ("A2 Complex+RevYLo", "A2 Complex+RevYLo", 0.003, 0.005),
    ("A2 Complex+Nano64x4", "A2 Complex+Nano64x4", 0.003, 0.005),
    ("A2 Complex+Nano125x3", "A2 Complex+Nano125x3", 0.003, 0.005),
    ("A2 Double-Lite", "A2 Double-Lite", 0.003, 0.005),
    ("A2 xDouble-Lite", "A2 xDouble-Lite", 0.003, 0.005),
    ("A2 DoublePlus-Lite", "A2 DoublePlus-Lite", 0.003, 0.005),
]
_WAVENET_LIGHT = [
    ("standard", "standard", 0.004, 0.0033),
    ("lite", "lite", 0.004, 0.0033),
    ("feather", "feather", 0.004, 0.0033),
    ("nano", "nano", 0.004, 0.0033),
    ("Nano64x4", "Nano64x4", 0.004, 0.0033),
    ("Nano125x3", "Nano125x3", 0.004, 0.0033),
]
_WAVENET_EXTENDED = [
    ("complex", "complex", 0.00352, 0.0033),
    ("xComplex", "xComplex", 0.00352, 0.0033),
    ("xComplex Lite", "xComplexLite", 0.004, 0.0033),
    ("aComplex", "aComplex", 0.003, 0.005),
    ("xstd", "xstd", 0.004, 0.0033),
    ("xstd3", "xstd3", 0.004, 0.0033),
    ("xhi3", "xhi3", 0.004, 0.0033),
    ("xHV_12", "xHV_12", 0.004, 0.0033),
    ("xHV_16", "xHV_16", 0.004, 0.0033),
    ("xHV24", "xHV_24", 0.004, 0.0033),
    ("revyhi", "revyhi", 0.004, 0.0033),
    ("revystd", "revystd", 0.004, 0.0033),
    ("revylo", "revylo", 0.004, 0.0033),
    ("revxstd", "revxstd", 0.004, 0.0033),
    ("ComplexRF300", "ComplexRF300", 0.004, 0.0033),
    ("ComplexRF300Lite", "ComplexRF300Lite", 0.004, 0.0033),
    ("ComplexRF600", "ComplexRF600", 0.004, 0.0033),
    ("ComplexRF600Lite", "ComplexRF600Lite", 0.004, 0.0033),
    ("double", "double", 0.00352, 0.0033),
    ("xDouble", "xDouble", 0.00352, 0.0033),
    ("yDouble", "yDouble", 0.00352, 0.0033),
    ("DoublePlus", "DoublePlus", 0.00352, 0.0033),
    ("ULTRA", "ULTRA", 0.00352, 0.0033),
    ("XHQ", "XHQ", 0.00352, 0.0033),
    ("UHQ", "UHQ", 0.00352, 0.0033),
]
_LSTM = [
    ("LSTM Compressor HQ 48x3", "LSTM Compressor HQ", 0.00312, 0.0033),
    ("LSTM Compressor Light 30x3", "LSTM Compressor Lite", 0.00312, 0.0033),
    ("LSTM UHQ", "LSTM UHQ", 0.00312, 0.0033),
    ("LSTM TONEX-like 16", "LSTM TX 16", 0.00312, 0.0033),
    ("TONEX-like Causal Conv LSTM 128-16-2048", "LSTM TX CC", 0.0028, 0.0033),
    ("Tonex HQ", "TX HQ", 0.0028, 0.0033),
]
_STATIC_ENTRIES = [
    {"value": v, "label": l, "group": g, "lr": lr, "lr_decay": d}
    for g, table in (("a2", _A2), ("light", _WAVENET_LIGHT), ("extended", _WAVENET_EXTENDED), ("lstm", _LSTM))
    for v, l, lr, d in table
]

_STATIC_SCHEDULERS = [
    "reduce_on_plateau",
    "exponential",
    "cosine_annealing",
    "cosine_annealing_warm_restarts",
    "warmup_cosine_decay",
    "one_cycle",
    "linear_warmup_reduce_on_plateau",
]
_STATIC_STAGE_MODES = ["single_stage", "two_stage"]  # refinement_only needs a .ckpt upload
_STATIC_STAGE2_FOCUS = ["Lows", "Mids", "Highs"]

DEFAULTS = {
    "official": {"epochs": 100, "batch_size": 16},
    "reloaded": {
        "epochs": 300,
        "batch_size": 32,
        "lr_scheduler": "exponential",
        "fit_mrstft": True,
        "stage_mode": "single_stage",
        "stage2_epochs": 100,
        "stage2_lr": 0.0025,
        "stage2_lr_decay": 0.6625,
        "stage2_lr_scheduler": "reduce_on_plateau",
        "stage2_focus": "Lows",
    },
}
# Without an accelerator the trainers' own desktop GUIs propose much smaller runs.
CPU_OVERRIDES = {"official": {"epochs": 20, "batch_size": 1}, "reloaded": {"epochs": 100}}


def defaults() -> dict:
    """DEFAULTS adapted to the hardware this installation is set up for."""
    out = {name: dict(values) for name, values in DEFAULTS.items()}
    if hardware.variant() == "cpu":
        for name, override in CPU_OVERRIDES.items():
            out[name].update(override)
    return out


def _is_str_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value)


def _valid_catalog(data) -> bool:
    """Structural check of catalog.json: anything odd and we fall back to the static tables."""
    try:
        entries = data["entries"]
        if not isinstance(entries, list) or not entries:
            return False
        for e in entries:
            if not (isinstance(e["value"], str) and isinstance(e["label"], str) and isinstance(e["group"], str)):
                return False
            if not all(isinstance(e[k], (int, float)) and not isinstance(e[k], bool) and e[k] > 0 for k in ("lr", "lr_decay")):
                return False
        return all(_is_str_list(data[k]) for k in ("schedulers", "stage_modes", "stage2_focus"))
    except (KeyError, TypeError):
        return False


def _catalog_file() -> dict:
    try:
        data = json.loads((RELOADED_PATH / "catalog.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if _valid_catalog(data) else {}


def reloaded_entries() -> list:
    return _catalog_file().get("entries") or _STATIC_ENTRIES


def reloaded_options() -> dict:
    data = _catalog_file()
    return {
        "schedulers": data.get("schedulers") or _STATIC_SCHEDULERS,
        "stage_modes": data.get("stage_modes") or _STATIC_STAGE_MODES,
        "stage2_focus": data.get("stage2_focus") or _STATIC_STAGE2_FOCUS,
    }


def catalog() -> list:
    """Groups of dropdown entries, as served to the GUI."""
    groups = [{
        "group": tr("trainer.group.official.name"),
        "hint": tr("trainer.group.official.hint"),
        "items": [{"id": OFFICIAL_ID, "label": tr("trainer.official.label", version=trainer_version("official")),
                   "backend": "official"}],
    }]
    if reloaded_available():
        by_group: dict = {}
        for e in reloaded_entries():
            by_group.setdefault(e["group"] if e["group"] in GROUP_ORDER else "extended", []).append(e)
        for gid in GROUP_ORDER:
            if gid in by_group:
                groups.append({"group": tr(f"trainer.group.{gid}.name"), "hint": tr(f"trainer.group.{gid}.hint"), "items": [
                    {"id": RELOADED_PREFIX + e["value"], "label": e["label"], "backend": "reloaded",
                     "lr": e["lr"], "lr_decay": e["lr_decay"]}
                    for e in by_group[gid]
                ]})
    return groups


def resolve(trainer_id: str):
    """Return (backend, architecture or None, display label) or None if unknown."""
    if trainer_id == OFFICIAL_ID:
        return "official", None, official_label()
    if trainer_id.startswith(RELOADED_PREFIX) and reloaded_available():
        arch = trainer_id[len(RELOADED_PREFIX):]
        for e in reloaded_entries():
            if e["value"] == arch:
                return "reloaded", arch, f"Reloaded - {e['label']}"
    return None
