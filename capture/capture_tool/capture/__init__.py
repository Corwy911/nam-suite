"""Calibrated reamp capture: audio engine, level calibration maths and take alignment.

Kept independent of the web server so it can be unit-tested without hardware:
  calibration.py  level maths (dBu / Vrms / dBFS), tone measurement, clipping
  alignment.py    latency detection, exact cut and quality checks (mirrors the NAM trainer for the standard v3 input)
  generators.py   what is sent to the interface (tone, the input file, clicks), always with fades
  backends.py     the audio backends: real ASIO devices (sounddevice) and a simulated interface
  engine.py       the audio process (JSON lines on stdin/stdout); ASIO calls stay on its main thread
  session.py      server side: starts the engine and talks to it
  store.py        persisted calibrations (nam-web/calibration.json)
The HTTP API is nam-web/capture_api.py, the page nam-web/static/capture.html.
"""
