# The shell contract: what a native shell must answer

NAM Capture has two native shells that host the same real-time engine (`native/engine/RTEngine`) and talk to the same
Python server (`capture_tool`): `packaging/ControlServer.swift` (macOS, CoreAudio) and
`packaging/windows/ControlServerWin.cpp` (Windows, ASIO/WASAPI). Everything else - the Python server, the web page,
the DSP - is shared. The only thing that must stay identical is this small local HTTP API.

* Listens on `127.0.0.1:8767`, plain HTTP/1.1, one request per connection, JSON bodies, `Connection: close`.
* Python calls it only through `_native_call()` in `capture_tool/capture_api/core.py` (`live.py`, `player.py`, `tuner.py`, ...).
* Success: HTTP 200 with `{"ok": true, ...}` (the Swift shell may omit `ok` on routes whose body carries the data - callers must not rely on it).
* Failure: HTTP 400 with `{"ok": false, "error": "<text>"}`; unknown route: HTTP 404 `{"error": "unknown route"}`.
* Gains are in dB; levels are in dBFS (peaks) or dBu (calibration levels read from a `.nam`); `null` means "not available".

| Route | Body | Does | Reply (besides `ok`) |
|---|---|---|---|
| `POST /open` | `device`, `in_channel`, `out_channel` (required); `extra_out_channels` [int], `blocksize`, `gain_db`, `ceiling_db` (default -1.0) | opens the interface for one input -> NAM/IR chain -> 1..N outputs | `blocksize`, `sample_rate` |
| `POST /open_dual` | `device`, `guitar_channel`, `return_channel`, `pedal_out_channel`, `monitor_out_channels` [int] (required); `blocksize`, `guitar_gain_db`, `ceiling_db` | guitar -> pedal output (unity gain) while the return runs through NAM/IR to the monitor outputs | `blocksize`, `sample_rate` |
| `POST /gain` | `gain_db` | output gain | `gain_db` |
| `POST /pregain` | `pre_gain_db` | gain before the model | `pre_gain_db` |
| `POST /guitar_gain` | `gain_db` | guitar leg gain (dual mode) | `gain_db` |
| `POST /gate` | `threshold_db` | noise gate threshold (-100 = off) | `threshold_db` |
| `POST /ir_gain` | `in_db` and/or `out_db` | gain before / after the IR convolver | - |
| `POST /eq` | `bass_db`, `mid_db`, `treble_db` (any subset) | 3-band tone stack | - |
| `POST /model` | `path` (`.nam`) | loads a model (hot swap, no gap) | `input_level_dbu`, `output_level_dbu` (or `null`) |
| `POST /model/unload` | - | removes the model | - |
| `POST /ir` | `path` (`.wav`) | loads a cabinet IR (hot swap) | - |
| `POST /ir/unload` | - | removes the IR | - |
| `POST /stop` | - | closes the interface | - |
| `GET /state` | - | meters and status | `opened`, `in_peak_dbfs`, `out_peak_dbfs`, `xruns`, `blocksize`, `sample_rate`, `model_loaded`, `ir_loaded`, `model_input_level_dbu`, `model_output_level_dbu`, `guitar_in_peak_dbfs`, `guitar_out_peak_dbfs` |

### Routes added with the NAM player (A/B against the real amp) and "Test Pedali"

| Route | Body | Does |
|---|---|---|
| `POST /open_player` | `device`, `guitar_channel`, `return_channel`, `reamp_out_channel`, `monitor_out_channels` [int] (required); `blocksize`, `feed_gain_db`, `ceiling_db` | opens the player: guitar -> real amp (leg A) and the guitar through the model (leg B), switchable |
| `POST /leg` | `leg` ("A" or "B") | which leg is heard on the monitor outputs |
| `POST /amp_gain`, `POST /amp_pregain` | `gain_db` | gain of the real amp's return / before the amp's model |
| `POST /b_trim`, `POST /b_delay` | `trim_db` / `samples` | level and delay of leg B, to match it to A |
| `POST /model_size` | `index` | WaveNet size (A2 "Lite" etc.) of the amp's model; reply `index`, `sizes` |
| `POST /pedal_model`, `POST /pedal_model/unload` | `path` (`.nam`) | the pedal's own NAM in front of the amp's model (Test Pedali); reply `input_level_dbu`, `output_level_dbu`, `sizes` |
| `POST /pedal_controls` | `pre_db`, `post_db`, `size` (any subset) | gains around the pedal's model and its size |

`/model`, `/model/unload`, `/ir`, `/ir/unload` also accept a legs mask (which legs get the file). `/state` reports, besides the fields above, the tuner
(`tuner_hz`, `tuner_confidence`). The authoritative list is the two route tables in the sources; the test below keeps them equal.

Missing required fields answer 400 with the same text in both shells (`missing device/in_channel/out_channel`, ...).

## How it is kept true

`capture/tests/test_native_shells.py` reads the route tables of both sources (nothing is compiled or run), checks that the two shells
implement exactly the same routes, and that every route Python calls with `_native_call()` exists in both. The fake native engine used by the API tests
(`FakeNativeEngine` in `tests/_helpers.py`) is the stand-in for both shells. A new route means: edit both shells and this document.
