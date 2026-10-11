# File: settings.py
# Created Date: Tuesday May 14th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Union

_THIS_DIR = Path(__file__).parent.resolve()
_SETTINGS_JSON_PATH = Path(_THIS_DIR, "settings.json")
_LAST_PATHS_KEY = "last_paths"
_UPDATE_KEY = "update"
_GUI_STATE_KEY = "gui_state"
_ADVANCED_OPTIONS_KEY = "advanced_options"
_ADVANCED_OPTION_PRESETS_KEY = "advanced_option_presets"
_METADATA_KEY = "metadata"
_CHECKBOXES_KEY = "checkboxes"
_PATH_SELECTIONS_KEY = "path_selections"
_NEWEST_AVAILABLE_VERSION_KEY = "newest_available_version"
_NEVER_SHOW_AGAIN_KEY = "never_show_again"


_BUILT_IN_ADVANCED_OPTION_PRESETS = {}


class PathKey(Enum):
    INPUT_FILE = "input_file"
    OUTPUT_FILE = "output_file"
    TRAINING_DESTINATION = "training_destination"
    LIGHTNING_FOLDER = "lightning_folder"
    CHECKPOINT_FILE = "checkpoint_file"


def get_last_path(
    path_key: PathKey, *, settings_path: Path = _SETTINGS_JSON_PATH
) -> Optional[Path]:
    s = _get_settings(settings_path)
    if _LAST_PATHS_KEY not in s:
        return None
    last_path = s[_LAST_PATHS_KEY].get(path_key.value)
    if last_path is None:
        return None
    assert isinstance(last_path, str)
    return Path(last_path)


def set_last_path(
    path_key: PathKey, path: Path, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    s = _get_settings(settings_path)
    if _LAST_PATHS_KEY not in s:
        s[_LAST_PATHS_KEY] = {}
    s[_LAST_PATHS_KEY][path_key.value] = str(path)
    _write_settings(s, settings_path=settings_path)

def get_path_selection(
    path_key: PathKey, *, settings_path: Path = _SETTINGS_JSON_PATH
) -> Optional[Union[str, tuple]]:
    selections = _get_gui_state(settings_path).get(_PATH_SELECTIONS_KEY) or {}
    selection = selections.get(path_key.value)
    if selection is None:
        return None
    if isinstance(selection, list):
        values = tuple(str(path) for path in selection if path)
        return values or None
    if isinstance(selection, str):
        return selection or None
    return None


def set_path_selection(
    path_key: PathKey,
    selection: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    settings_path: Path = _SETTINGS_JSON_PATH,
):
    s = _get_settings(settings_path)
    if _GUI_STATE_KEY not in s:
        s[_GUI_STATE_KEY] = {}
    if _PATH_SELECTIONS_KEY not in s[_GUI_STATE_KEY]:
        s[_GUI_STATE_KEY][_PATH_SELECTIONS_KEY] = {}

    if isinstance(selection, (list, tuple)):
        value = [str(path) for path in selection if path]
    else:
        value = str(selection)
    s[_GUI_STATE_KEY][_PATH_SELECTIONS_KEY][path_key.value] = value
    _write_settings(s, settings_path=settings_path)


def get_update_settings(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    """
    Return update-related settings: newest_available_version (str or None),
    never_show_again (bool).
    """
    s = _get_settings(settings_path)
    update = s.get(_UPDATE_KEY) or {}
    return {
        _NEWEST_AVAILABLE_VERSION_KEY: update.get(_NEWEST_AVAILABLE_VERSION_KEY),
        _NEVER_SHOW_AGAIN_KEY: bool(update.get(_NEVER_SHOW_AGAIN_KEY, False)),
    }


def set_update_settings(
    newest_available_version: Optional[str] = None,
    never_show_again: Optional[bool] = None,
    *,
    settings_path: Path = _SETTINGS_JSON_PATH,
):
    """
    Update one or more update settings. Pass None for a key to leave it unchanged.
    """
    s = _get_settings(settings_path)
    if _UPDATE_KEY not in s:
        s[_UPDATE_KEY] = {}
    if newest_available_version is not None:
        s[_UPDATE_KEY][_NEWEST_AVAILABLE_VERSION_KEY] = newest_available_version
    if never_show_again is not None:
        s[_UPDATE_KEY][_NEVER_SHOW_AGAIN_KEY] = never_show_again
    _write_settings(s, settings_path=settings_path)


def get_advanced_options_settings(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    return dict(_get_gui_state(settings_path).get(_ADVANCED_OPTIONS_KEY) or {})


def set_advanced_options_settings(
    advanced_options: dict, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    _set_gui_state_section(
        _ADVANCED_OPTIONS_KEY, advanced_options, settings_path=settings_path
    )


def get_advanced_options_presets(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    s = _get_settings(settings_path)
    saved_presets = s.get(_ADVANCED_OPTION_PRESETS_KEY) or {}
    presets = {
        name: dict(values)
        for name, values in _BUILT_IN_ADVANCED_OPTION_PRESETS.items()
    }
    presets.update(
        {
            str(name): dict(values or {})
            for name, values in saved_presets.items()
            if isinstance(name, str)
        }
    )
    return presets


def set_advanced_options_preset(
    name: str, advanced_options: dict, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    s = _get_settings(settings_path)
    if _ADVANCED_OPTION_PRESETS_KEY not in s:
        s[_ADVANCED_OPTION_PRESETS_KEY] = {}
    s[_ADVANCED_OPTION_PRESETS_KEY][name] = advanced_options
    _write_settings(s, settings_path=settings_path)


def get_metadata_settings(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    return dict(_get_gui_state(settings_path).get(_METADATA_KEY) or {})


def set_metadata_settings(metadata: dict, *, settings_path: Path = _SETTINGS_JSON_PATH):
    _set_gui_state_section(_METADATA_KEY, metadata, settings_path=settings_path)


def get_checkbox_settings(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    return dict(_get_gui_state(settings_path).get(_CHECKBOXES_KEY) or {})


def set_checkbox_settings(
    checkboxes: dict, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    _set_gui_state_section(_CHECKBOXES_KEY, checkboxes, settings_path=settings_path)


def _get_gui_state(settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    s = _get_settings(settings_path)
    return dict(s.get(_GUI_STATE_KEY) or {})


def _set_gui_state_section(
    key: str, value: dict, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    s = _get_settings(settings_path)
    if _GUI_STATE_KEY not in s:
        s[_GUI_STATE_KEY] = {}
    s[_GUI_STATE_KEY][key] = value
    _write_settings(s, settings_path=settings_path)


def _get_settings(settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    """
    Make sure that ./settings.json exists; if it does, then read it. If not, empty dict.
    """
    if not settings_path.exists():
        return dict()
    else:
        with open(settings_path, "r") as fp:
            return json.load(fp)


class _WriteSettings(object):
    def __init__(self):
        self._oserror = False

    def __call__(self, *args, **kwargs):
        if self._oserror:
            return
        # Try-catch for Issue 448
        try:
            return _write_settings_unsafe(*args, **kwargs)
        except OSError as e:
            if "Read-only filesystem" in str(e):
                print(
                    "Failed to write settings--NAM appears to be installed to a "
                    "read-only filesystem. This is discouraged; consider installing to "
                    "a location with user-level access."
                )
                self._oserror = True
            else:
                raise e


_write_settings = _WriteSettings()


def _write_settings_unsafe(obj: dict, settings_path: Path = _SETTINGS_JSON_PATH):
    with open(settings_path, "w") as fp:
        json.dump(obj, fp, indent=4)
