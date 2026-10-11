"""Level-calibration maths for the capture tool. Pure functions: no audio I/O, no server code.

NAM metadata conventions (trainers/*/nam/models/metadata.py):
  input_level_dbu   ("Co")  analog dBu at the interface OUTPUT (the gear side) for a sine whose PEAK is 0 dBFS
  output_level_dbu  ("Ci")  analog dBu at the interface INPUT that reads 0 dBFS
with dBu = 20*log10(Vrms / 0.7746), Vrms being the RMS voltage of the sine (NAM calibration guide).
"""

import math
from typing import Optional

import numpy as np

RATE = 48_000
TONE_HZ = 1_000.0
DBU_REF_VRMS = 0.7746          # the figure printed in the NAM calibration guide
CLIP_LEVEL = 0.9995            # a sample at or above this counts as clipped


def count_clipped(x, level: float = CLIP_LEVEL, min_run: int = 2) -> int:
    """Samples that sit on full scale in runs of at least `min_run` (a flat top). A single sample touching full scale, which
    a 0 dBFS sine does at every peak, is not clipping."""
    hit = (np.abs(np.asarray(x)) >= level).astype(np.int8)
    if not hit.any():
        return 0
    edges = np.diff(np.concatenate(([0], hit, [0])))
    runs = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    return int(runs[runs >= min_run].sum())


def vrms_to_dbu(vrms: float) -> float:
    if not vrms > 0:
        raise ValueError("Vrms must be positive")
    return 20.0 * math.log10(vrms / DBU_REF_VRMS)


def dbu_to_vrms(dbu: float) -> float:
    return DBU_REF_VRMS * 10.0 ** (dbu / 20.0)


def to_db(linear: float) -> float:
    return 20.0 * math.log10(max(float(linear), 1e-12))


def from_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def sine(seconds: float = 1.0, peak_dbfs: float = 0.0, freq: float = TONE_HZ, rate: int = RATE) -> np.ndarray:
    """A sine starting at phase zero. With the default 1 s / 1 kHz / 48 kHz it holds exactly 1000 cycles, like the
    sine_1k.wav of the NAM guide (which it matches to within 24-bit quantisation)."""
    n = int(round(seconds * rate))
    return from_db(peak_dbfs) * np.sin(2.0 * np.pi * freq * np.arange(n) / rate)


def measure_tone(x, rate: int = RATE, freq: float = TONE_HZ) -> dict:
    """Level of a steady sine in a recording: single-bin DFT over a whole number of cycles taken from the end of `x`
    (so the start-up of the stream is ignored).

    peak_dbfs      peak of the fundamental (a sine of peak 1.0 reads 0 dBFS)
    residual_dbfs  RMS of everything else (noise + distortion), DC removed
    thdn_db        that residual relative to the fundamental's RMS
    sample_peak_dbfs / clipped  largest absolute sample, and how many samples touch full scale
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    cycles = int(len(x) * freq / rate)
    if cycles < 1:
        raise ValueError("recording too short to measure the tone")
    n = int(round(cycles * rate / freq))
    seg = x[len(x) - n:]
    dc = float(seg.mean())
    w = 2.0 * np.pi * freq / rate
    k = np.arange(n)
    c = 2.0 * np.sum((seg - dc) * np.exp(-1j * w * k)) / n
    amp = float(abs(c))
    residual = (seg - dc) - amp * np.cos(w * k + np.angle(c))
    resid_rms = float(np.sqrt(np.mean(residual ** 2)))
    tone_rms = amp / math.sqrt(2.0)
    return {
        "peak_dbfs": to_db(amp),
        "rms_dbfs": to_db(tone_rms),
        "residual_dbfs": to_db(resid_rms),
        "thdn_db": to_db(resid_rms / max(tone_rms, 1e-12)),
        "sample_peak_dbfs": to_db(float(np.max(np.abs(seg)))),
        "clipped": count_clipped(seg),
        "dc": dc,
    }


# ------------------------------------------------------------------- the calibrations

def output_level_dbu(vrms: float, sent_peak_dbfs: float = 0.0) -> float:
    """Co: what a 0 dBFS-peak sine gives at the output, from the multimeter reading of a sine sent at `sent_peak_dbfs`."""
    return vrms_to_dbu(vrms) - sent_peak_dbfs


def input_level_dbu(co: float, sent_peak_dbfs: float, read_peak_dbfs: float) -> float:
    """Ci by loopback: the calibrated output sent a sine at `sent_peak_dbfs` (so `co + sent` dBu reach the input) and the input
    read `read_peak_dbfs`. This is the guide's dBu_return = dBu_send + delta - L."""
    return co + sent_peak_dbfs - read_peak_dbfs


def unity_gain_db(ci: float, co: float) -> float:
    """Digital gain that makes the voltage at the amp equal the voltage at the guitar input: a signal of x dBu becomes
    x - Ci dBFS, then + G, then + Co dBu again."""
    return ci - co


def nam_input_gain_db(model_input_dbu: float, ci_source: float) -> float:
    """Digital gain in front of a NAM model so that it sees the level it was captured at. A signal of x dBu reads x - Ci dBFS
    at the interface input, but the model's input is calibrated so that 0 dBFS = `model_input_dbu`, i.e. it expects x - M:
    the gain is Ci - M (same as the official plugin: input calibration minus the model's input level)."""
    return ci_source - model_input_dbu


def nam_output_gain_db(model_output_dbu: float, co: float) -> float:
    """Digital gain after a NAM model so that its output leaves the interface at the analog level it had in the capture.
    The model's output is calibrated so that 0 dBFS = `model_output_dbu`; the interface output gives Co dBu for 0 dBFS:
    the gain is model_output - Co."""
    return model_output_dbu - co


# ---- the whole chains, for the player (guitar at the Hi-Z input; the same analog level goes to the real gear and to the model)

def player_gains(ci_guitar, ci_return, co, model_in=None, model_out=None) -> dict:
    """Every gain of the player's two legs, in dB (a value is None when the numbers it needs are missing):
      feed   guitar -> reamp output, exact unity gain in voltage (Ci_guitar - Co)
      amp    return -> monitor (leg A, the real gear): its voltage reproduced as it is (Ci_return - Co)
      pre    guitar -> NAM input (leg B): Ci_guitar - model input level
      post   NAM output -> monitor (leg B): model output level - Co
    Both legs end on the same Co, so what changes between A and B is the gear, not the level. The difference with the real
    gear is therefore only what the model gets wrong."""
    def diff(a, b):
        return None if a is None or b is None else a - b
    return {"feed": diff(ci_guitar, co), "amp": diff(ci_return, co),
            "pre": diff(ci_guitar, model_in), "post": diff(model_out, co)}


def suggest_send_level(sent_dbfs: float, read_peak_dbfs: float, clipped: bool = False, target_dbfs: float = -9.0,
                       lowest_dbfs: float = -60.0, highest_dbfs: float = -0.5) -> float:
    """Next level for the automatic loopback search: aim the reading at `target_dbfs`, back off hard if it clipped."""
    if clipped or read_peak_dbfs > -0.5:
        step = -12.0
    elif read_peak_dbfs < -90.0:
        step = 20.0                      # nothing arrives yet (cable? gain?): try louder, capped below
    else:
        step = target_dbfs - read_peak_dbfs
    return max(lowest_dbfs, min(highest_dbfs, sent_dbfs + step))


def judge_level(peak_dbfs: float, clipped: int = 0) -> str:
    """'clip' | 'hot' | 'ok' | 'low' | 'silent' for a return level."""
    if clipped or peak_dbfs >= -0.3:
        return "clip"
    if peak_dbfs >= -3.0:
        return "hot"
    if peak_dbfs < -60.0:
        return "silent"
    if peak_dbfs < -30.0:
        return "low"
    return "ok"


def linearity_levels(co: float, ci: float, targets=(-40.0, -30.0, -20.0, -12.0, -6.0)) -> list:
    """Software send levels (0.5 dB grid) whose loopback reading should land on `targets` dBFS, given Co and Ci: read = Co + sent - Ci."""
    out = []
    for target in targets:
        level = round((target - co + ci) * 2.0) / 2.0
        if -60.0 <= level <= 0.0 and level not in out:
            out.append(level)
    return out


def linearity_verdict(points, ok_db: float = 0.1, warn_db: float = 0.3, floor_dbfs: float = -50.0) -> dict:
    """Judges a set of loopback readings {level_dbfs, peak_dbfs, ci_dbu, clipped}: a linear input gives the same Ci at every level.
    status 'ok' (spread within `ok_db`), 'warn' (within `warn_db`), 'bad' (beyond, clipped, or too few usable readings).
    Readings below `floor_dbfs` are noise-limited and left out of the comparison."""
    usable = [p for p in points if p["peak_dbfs"] > floor_dbfs]
    clipped = any(p.get("clipped") for p in points)
    if len(usable) < 3:
        return {"status": "bad", "reason": "too_few", "max_dev_db": None, "median_ci": None, "span_db": None, "deviations": []}
    cis = np.array([p["ci_dbu"] for p in usable], dtype=float)
    median = float(np.median(cis))
    dev = cis - median
    max_dev = float(np.max(np.abs(dev)))
    span = float(max(p["peak_dbfs"] for p in usable) - min(p["peak_dbfs"] for p in usable))
    status = "ok" if max_dev <= ok_db else "warn" if max_dev <= warn_db else "bad"
    if clipped:
        status = "bad"
    deviations = {id(p): float(d) for p, d in zip(usable, dev)}
    return {"status": status, "reason": "clip" if clipped else None, "max_dev_db": round(max_dev, 3), "median_ci": round(median, 2),
            "span_db": round(span, 1), "deviations": [None if id(p) not in deviations else round(deviations[id(p)], 3) for p in points]}


def suggest_trim_db(sample_peak_dbfs: float, target_dbfs: float = -6.0) -> float:
    """How far to lower the send level (a negative number, 0.5 dB grid, rounded so as to be at least enough) to bring a return
    peak of `sample_peak_dbfs` (taken as 0 when clipped: the true peak is unknown) down to `target_dbfs`."""
    peak = 0.0 if sample_peak_dbfs >= -0.3 else sample_peak_dbfs
    return -math.ceil(max(peak - target_dbfs, 0.0) * 2.0) / 2.0


def parse_vrms(value) -> Optional[float]:
    """A multimeter reading typed by the user ('6.4', '6,40 V', '6400 mV'); None if it is not a plausible number."""
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        text = str(value).strip().lower().replace(",", ".").replace(" ", "")
        scale = 1.0
        if text.endswith("mv"):
            text, scale = text[:-2], 1e-3
        elif text.endswith("v"):
            text = text[:-1]
        try:
            v = float(text) * scale
        except ValueError:
            return None
    return v if 0.0005 <= v <= 200.0 else None
