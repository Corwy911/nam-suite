"""Italian / English texts shared by the server and the web page (static/i18n.json).

The catalogue maps a key to a text with {placeholders}. The page uses the "ui.*" keys itself and asks the
server for everything else: every API call carries an `X-Lang` header (it | en), the middleware in app.py
stores it in a context variable, and `tr(key, **params)` renders in that language (Italian if unknown).

Messages that are produced in the background (updater log, self-test lines, failed jobs) are stored as
language-neutral {"k": key, "p": params} dicts and rendered when they are served, so switching language in the
page translates them as well. Parameter values that are themselves {"k", "p"} dicts are rendered recursively.
"""

import contextvars
import json
from pathlib import Path
from typing import Any, Optional

CATALOG_FILE = Path(__file__).resolve().parent / "static" / "i18n.json"
LANGS = ("it", "en")
DEFAULT_LANG = "it"

_current: contextvars.ContextVar = contextvars.ContextVar("nam_lang", default=DEFAULT_LANG)
_catalog: dict = {}
_catalog_mtime = 0.0


def load() -> dict:
    """The catalogue, re-read when the file changes (so editing it needs no restart)."""
    global _catalog, _catalog_mtime
    try:
        mtime = CATALOG_FILE.stat().st_mtime
    except OSError:
        return _catalog
    if mtime != _catalog_mtime:
        _catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        _catalog_mtime = mtime
    return _catalog


def normalize(value: Optional[str]) -> str:
    value = (value or "").strip().lower()[:2]
    return value if value in LANGS else DEFAULT_LANG


def set_language(value: Optional[str]):
    return _current.set(normalize(value))


def reset_language(token) -> None:
    _current.reset(token)


def get_language() -> str:
    return _current.get()


def is_message(value: Any) -> bool:
    return isinstance(value, dict) and set(value) <= {"k", "p"} and isinstance(value.get("k"), str)


def msg(key: str, **params: Any) -> dict:
    """A language-neutral message, to be stored and rendered later with render()."""
    return {"k": key, "p": params} if params else {"k": key}


def tr(key: str, lang: Optional[str] = None, **params: Any) -> str:
    lang = normalize(lang) if lang else get_language()
    catalog = load()
    text = catalog.get(lang, {}).get(key) or catalog.get(DEFAULT_LANG, {}).get(key) or key
    def resolve(value):
        if is_message(value):
            return render(value, lang)
        if isinstance(value, list) and all(is_message(v) or isinstance(v, str) for v in value):
            return "; ".join(render(v, lang) for v in value)  # a list of messages becomes one "a; b" text
        return value

    resolved = {k: resolve(v) for k, v in params.items()}
    try:
        return text.format(**resolved)
    except (KeyError, IndexError, ValueError):
        return text


def render(message: Any, lang: Optional[str] = None) -> str:
    """Render a stored message ({"k", "p"}); plain strings are returned unchanged."""
    if is_message(message):
        return tr(message["k"], lang, **message.get("p", {}))
    return "" if message is None else str(message)


class LocalizedError(Exception):
    """An error whose text depends on the language of whoever reads it."""

    def __init__(self, key: str, **params: Any):
        super().__init__(key)
        self.key = key
        self.params = params

    def message(self) -> dict:
        return msg(self.key, **self.params)

    def __str__(self) -> str:  # server console / tests: Italian, the default language
        return tr(self.key, DEFAULT_LANG, **self.params)
