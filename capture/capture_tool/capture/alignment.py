"""Take alignment for the standard NAM v3 input: latency detection, exact cut and quality checks.

The maths mirrors trainers/*/nam/train/core.py (`_calibrate_latency_v_all` with `_V3_DATA_INFO`, `_check_v3`) so that a take cut here
is seen by the trainer as already aligned. numpy only (no torch): the audio process stays light. tests/test_capture_alignment.py
asserts the equivalence against the installed trainer.

Why cutting is safe: the return of the amp always lags the playback by the round trip of the interface (driver buffers, converters)
plus whatever the gear adds. The take is recorded LONGER than the input (extra tail), the delay is measured from the two blips the v3
input carries at 10.5 s and 11.5 s, and the recording is then cut from `delay - 1` for exactly len(input) samples: nothing of the
response is lost. (The official trainer subtracts a "safety factor" of 1 sample from the delay it measures, the Reloaded one 0: the
raw delay is the same, and the cut never lands later than the first sample of the response. Each trainer then makes its own final
one-sample correction, so the take needs no sub-sample perfection.)
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import calibration as cal

RATE = 48_000
INPUT_V3_MD5 = "36cd1af62985c2fac3e654333e36431e"   # the hash the trainer treats as the "strong" v3 match
INPUT_V3_FRAMES = 9_120_000                          # 190 s at 48 kHz

# _V3_DATA_INFO (samples at 48 kHz)
FIRST_BLIPS_START = 480_000
T_BLIPS = 96_000
NOISE_INTERVAL = (492_000, 498_000)
BLIP_LOCATIONS = (504_000, 552_000)
T_VALIDATE = 432_000
# _calibrate_latency_v_all defaults
ABS_THRESHOLD = 0.0003
REL_THRESHOLD = 0.001
SAFETY_FACTOR = 1
LOOKAHEAD = 1_000
LOOKBACK = 10_000
ESR_REPLICATE_THRESHOLD = 0.01                       # _check_v3

MIN_LENGTH_FOR_LATENCY = FIRST_BLIPS_START + T_BLIPS


def describe_input(path) -> dict:
    """Is this file the standard v3 input? (`v3` needs the same md5 as the trainer's list, 48 kHz and mono.)"""
    import soundfile as sf

    path = Path(path)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            md5.update(chunk)
    info = sf.info(str(path))
    digest = md5.hexdigest()
    return {"md5": digest, "frames": info.frames, "rate": info.samplerate, "channels": info.channels,
            "v3": digest == INPUT_V3_MD5 and info.samplerate == RATE and info.channels == 1}


def detect_latency(y) -> dict:
    """Port of `_calibrate_latency_v_all` for the v3 input. `y` is the whole recording (or the aligned take).

    delay        samples between the first blip and the first sample of the response above the trigger (None: no response found)
    recommended  delay - safety factor, what the trainer applies
    background   loudest sample of the quiet interval before the blips
    response_peak  loudest sample of the averaged blip responses
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < MIN_LENGTH_FOR_LATENCY:
        raise ValueError("recording too short to hold the blips")
    seg = y[FIRST_BLIPS_START:FIRST_BLIPS_START + T_BLIPS]
    background = float(np.max(np.abs(seg[NOISE_INTERVAL[0] - FIRST_BLIPS_START:NOISE_INTERVAL[1] - FIRST_BLIPS_START])))
    threshold = max(background + ABS_THRESHOLD, (1.0 + REL_THRESHOLD) * background)
    scans = []
    for i_abs in BLIP_LOCATIONS:
        i_rel = i_abs - FIRST_BLIPS_START
        scans.append(seg[i_rel - LOOKAHEAD:i_rel + LOOKBACK])
    average = np.mean(np.stack(scans), axis=0)
    triggered = np.where(np.abs(average) > threshold)[0]
    result = {"delay": None, "recommended": None, "background": background, "threshold": float(threshold),
              "response_peak": float(np.max(np.abs(average)))}
    if len(triggered):
        delay = int(triggered[0]) - LOOKAHEAD
        result.update(delay=delay, recommended=delay - SAFETY_FACTOR, at_lookahead_limit=delay == -LOOKAHEAD)
    return result


def cut_and_pad(rec, n: int, cut: int) -> np.ndarray:
    """Exactly `n` samples starting `cut` samples into `rec` (a negative cut puts silence in front). Short recordings are
    padded with zeros at the end; the caller decides whether that is acceptable."""
    rec = np.asarray(rec, dtype=np.float64).ravel()
    if cut < 0:
        rec = np.concatenate([np.zeros(-cut), rec])
        cut = 0
    out = rec[cut:cut + n]
    if len(out) < n:
        out = np.concatenate([out, np.zeros(n - len(out))])
    return out


def replicate_esr(y, n: int) -> float:
    """`_check_v3`: the first and the last validation segments (the same signal played twice) must look alike."""
    y = np.asarray(y, dtype=np.float64).ravel()
    a = y[:T_VALIDATE]
    b = y[n - T_VALIDATE:n]
    return float(np.mean((a - b) ** 2) / max(float(np.mean(b ** 2)), 1e-30))


def replicate_detail(y, n: int, window: int = 4_800, excess_db: float = 8.0, max_slip: int = 8) -> dict:
    """Where the two replicates disagree, not just how much (`replicate_esr` is one number for the whole validation segment).

    The error of each 100 ms window is compared with the typical (median) one: windows that stand out by `excess_db` are "the stretch".
    Noise or a drifting amp spoil every window alike and leave no stretch; a timing glitch of the interface or a pop spoils a short
    part only. For that stretch the second replicate is slid by whole samples: if it matches again at some offset, the stretch is a
    slip of that many samples (a clock hiccup of the converter/USB driver, not the amp).

    stretch      None, or [start_s, end_s] (seconds into the replicate, first to last window that stands out)
    slip         samples the second replicate is late (+) or early (-) in the stretch, None if no whole offset explains it
    esr_rest     the ESR without the stretch (what the take is like everywhere else)
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    a = y[:T_VALIDATE]
    b = y[n - T_VALIDATE:n]
    count = T_VALIDATE // window
    err = ((a - b) ** 2)[:count * window].reshape(count, window).sum(axis=1)
    sig = (b ** 2)[:count * window].reshape(count, window).sum(axis=1)
    typical = max(float(np.median(err)), 1e-30)
    hot = err > typical * 10 ** (excess_db / 10)
    rest_sig = float(sig[~hot].sum())
    detail = {"stretch": None, "slip": None, "esr_rest": float(err[~hot].sum() / rest_sig) if rest_sig > 0 else None}
    if not hot.any() or hot.all():
        return detail
    first, last = int(np.argmax(hot)), int(count - 1 - np.argmax(hot[::-1]))
    detail["stretch"] = [round(first * window / RATE, 2), round((last + 1) * window / RATE, 2)]
    lo, hi = first * window, (last + 1) * window
    margin = max_slip
    lo = max(lo, margin)                                  # a stretch from the very first sample: the lags need `margin` samples of room before it
    if hi + margin <= len(b) and hi > lo:
        best, best_err = 0, float(np.sum((a[lo:hi] - b[lo:hi]) ** 2))
        for lag in range(-max_slip, max_slip + 1):
            e = float(np.sum((a[lo:hi] - b[lo + lag:hi + lag]) ** 2))
            if e < best_err:
                best, best_err = lag, e
        if best != 0 and best_err < 0.25 * float(np.sum((a[lo:hi] - b[lo:hi]) ** 2)):
            detail["slip"] = best
    return detail


def realign_first_replicate(aligned, n: int, max_slip: int = 2, window: int = 4_800):
    """Puts the first replicate back in step with the second when the only thing between them is a whole-sample timing step.

    Some USB interfaces move the recording by one whole sample against the playback once, at a random moment after the stream starts
    (a Focusrite 18i20 did it 1.6 s, 4.3 s and more than 13.8 s in, always the same way). Whatever came before the step is then that
    many samples off against whatever came after: part of the first replicate, or all of it. That first part of the file (0-9 s, before
    the blips at 10 s) is not trained on - the trainer only compares it with the last 9 s to check the take - so moving it back costs the
    model nothing and lets the check measure the gear instead of the driver.

    The fit: the first replicate is shifted by s (1..max_slip samples, either way) up to a boundary (anywhere in it, or all of it) and
    left alone after it; the boundary and s that make it closest to the second replicate win. It is used only when it makes the two agree
    (ESR within the trainer's limit) and removes most of the difference - noise, a drifting amp or a moved knob are not fixed by a shift.
    Returns (realigned samples, info) or None."""
    x = np.asarray(aligned, dtype=np.float64)
    a, b = x[:T_VALIDATE], x[n - T_VALIDATE:n]
    count = T_VALIDATE // window
    span = count * window
    signal = float(np.sum(b[:span] ** 2))
    if signal <= 0:
        return None

    def per_window(r):
        return ((r[:span] - b[:span]) ** 2).reshape(count, window).sum(axis=1)

    def shifted(s):
        r = np.zeros(T_VALIDATE)
        if s > 0:
            r[s:] = a[:-s]
        else:
            r[:s] = a[-s:]
        return r

    e0 = np.concatenate([[0.0], np.cumsum(per_window(a))])
    best = None
    for s in [v for k in range(1, max_slip + 1) for v in (k, -k)]:
        es = np.concatenate([[0.0], np.cumsum(per_window(shifted(s)))])
        total = es[1:] + (e0[-1] - e0[1:])                     # windows before the boundary shifted, the others as they are
        c = int(np.argmin(total))
        if best is None or total[c] < best[0]:
            best = (float(total[c]), s, c + 1)
    error, s, c = best
    k = abs(s)
    if c == count:            # all of it: the step came after the first replicate; join in the quiet second before the blips
        p = np.arange(T_VALIDATE, FIRST_BLIPS_START - 9_600)
        # the splice drops (s > 0) or repeats (s < 0) `k` samples; do it where the waveform joins most smoothly
        jump = np.abs(x[p] - x[p - 1 - k]) if s > 0 else np.abs(x[p - k] - x[p - 1])
        cut = int(p[int(np.argmin(jump))])
    else:                     # inside it: the exact sample where the two parts meet best (where the step was)
        lo, hi = max((c - 1) * window, k), min((c + 1) * window, T_VALIDATE)
        r = shifted(s)
        moved = np.concatenate([[0.0], np.cumsum((r[lo:hi] - b[lo:hi]) ** 2)])
        kept = np.concatenate([[0.0], np.cumsum((a[lo:hi] - b[lo:hi]) ** 2)])
        joined = moved + (kept[-1] - kept)
        cut = lo + int(np.argmin(joined))
        error = float(np.sum((np.concatenate([r[:cut], a[cut:]])[:span] - b[:span]) ** 2))
    if error / signal > ESR_REPLICATE_THRESHOLD or error > 0.25 * e0[-1]:
        return None
    out = x.copy()
    if s > 0:                          # the first part runs `k` samples early against the rest: delay it
        out[k:cut] = x[:cut - k]
        out[:k] = 0.0
    else:                              # ... or late: advance it
        out[:cut - k] = x[k:cut]
    return out, {"slip": int(s), "until_s": round(cut / RATE, 2)}


# ------------------------------------------------------------------ timing reference around the input
# The v3 input cannot show where a step of the driver happened: its only identical stimuli are the two replicates and the two blips. So the
# capture frames it with more copies of its blip (one sample at 0.99): a train of them before the input - the warm-up, which goes on until
# the step has been seen, so that it happens before the input instead of inside it - and a few after it. The same impulse into the same
# gear gives the same waveform back, so comparing two responses says, to a hundredth of a sample, whether the timing moved in between.
BLIP_VALUE = 0.99
BLIP_EVERY = 24_000                  # 0.5 s between warm-up blips (the v3 blips are 1 s apart)
WARMUP_FIRST = 24_000                # the first one 0.5 s into the stream (the very first input samples of a stream can be stale)
WARMUP_MIN = 144_000                 # at least 3 s of warm-up
WARMUP_MAX = 1_440_000               # at most 30 s: then the input starts anyway, and the check after it says what happened
WARMUP_GAP = 48_000                  # silence between the last warm-up blip and the input
TAIL_FIRST = 12_000                  # after the input and its tail of silence: TAIL_BLIPS blips, the first 0.25 s in
TAIL_BLIPS = 3
RESP_SPAN = 12_000                   # the response of a blip is looked for in the 0.25 s after it
STEP_MIN = 0.5                       # two responses this many samples apart (or more): the timing moved
MIN_CORR = 0.8                       # below this the two responses are not alike enough to compare (noise, a gate)


def blip_onset(y, p: int) -> Optional[int]:
    """Where the response to a blip played at stream position `p` starts (samples after p): the first sample above the noise of the
    125 ms before it, with the trainer's thresholds. None: nothing came back."""
    y = np.asarray(y)
    noise = float(np.max(np.abs(y[max(p - 6_000, 0):p]))) if p > 0 else 0.0
    threshold = max(noise + ABS_THRESHOLD, (1.0 + REL_THRESHOLD) * noise)
    hits = np.where(np.abs(y[p:p + RESP_SPAN]) > threshold)[0]
    return int(hits[0]) if len(hits) else None


def response_lag(y, p_ref: int, p: int, onset: int, width: int = 4_096, max_lag: int = 3) -> Tuple[float, float]:
    """How many samples later (sub-sample, + = later) the response to the blip at `p` arrives than the one to the blip at `p_ref`, and how
    alike the two are (correlation, 1 = the same waveform). A handful of dot products: cheap enough for the audio callback."""
    y = np.asarray(y)
    lo = max(onset - 64, max_lag)
    a = np.asarray(y[p_ref + lo:p_ref + lo + width], dtype=np.float64)
    if len(a) < width or p + lo - max_lag < 0 or p + lo + width + max_lag > len(y):
        return 0.0, 0.0
    c = np.array([float(np.dot(a, y[p + lo + l:p + lo + l + width])) for l in range(-max_lag, max_lag + 1)])
    k = int(np.argmax(c))
    d = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2.0 * c[k] + c[k + 1]
        d = 0.5 * (c[k - 1] - c[k + 1]) / den if den else 0.0
    b = np.asarray(y[p + lo + k - max_lag:p + lo + k - max_lag + width], dtype=np.float64)
    norm = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return k - max_lag + d, (c[k] / norm if norm > 0 else 0.0)


class WarmupWatch:
    """Watches the warm-up while it is being recorded and decides when the input may start.

    Fed with every recorded block (`feed`, from the audio callback: per blip only a few dot products). Each blip's response is compared
    with the previous one; a difference of half a sample or more is the driver's step. The input starts once a step has been seen and the
    two blips after it agree with each other (the timing has settled), never before WARMUP_MIN; with no step by WARMUP_MAX it starts anyway
    (the step may have come before the first blip, or this interface never does it). With no response to the blips at all (nothing is
    connected, the gear is off) there is nothing to watch: it starts at WARMUP_MIN."""

    def __init__(self, min_frames: int = WARMUP_MIN, max_frames: int = WARMUP_MAX):
        self.min_frames, self.max_frames = min_frames, max_frames
        self.blips = list(range(WARMUP_FIRST, max_frames + 1, BLIP_EVERY))
        self.buf = np.zeros(max_frames + BLIP_EVERY + RESP_SPAN + 8_192, dtype=np.float32)
        self.filled = 0
        self.checked = 0                       # blips whose response has been looked at
        self.onset: Optional[int] = None
        self.steps: list = []                  # [stream seconds, samples] of each step seen
        self.last_step = None                  # index of the blip after the last step
        self.start: Optional[int] = None       # stream position where the input starts, once decided
        self.reason: Optional[str] = None      # step_seen | max | no_response
        self.played: list = []                 # blips actually played (filled by the generator)

    def feed(self, block, pos: int) -> None:
        if self.start is not None:
            return
        block = np.asarray(block, dtype=np.float32).ravel()
        n = min(len(block), len(self.buf) - pos) if pos < len(self.buf) else 0
        if n > 0:
            self.buf[pos:pos + n] = block[:n]
        self.filled = pos + len(block)
        while self.checked < len(self.blips) and self.filled >= self.blips[self.checked] + RESP_SPAN + 8:
            self._look(self.checked)
            self.checked += 1
        if self.start is None and self.filled >= self.max_frames:
            self._decide("max")

    def _look(self, i: int) -> None:
        p = self.blips[i]
        if i == 0:
            self.onset = blip_onset(self.buf, p)
            if self.onset is None:
                self._decide("no_response", at_least=self.min_frames)
            return
        if self.onset is None:
            return
        lag, corr = response_lag(self.buf, self.blips[i - 1], p, self.onset)
        if corr >= MIN_CORR and abs(lag) >= STEP_MIN:
            self.steps.append([round(p / RATE, 2), int(round(lag))])
            self.last_step = i
        elif self.last_step is not None and i - self.last_step >= 2 and corr >= MIN_CORR:
            self._decide("step_seen", at_least=self.min_frames)

    def _decide(self, reason: str, at_least: int = 0) -> None:
        if self.start is None:
            self.reason = reason
            self.start = max(self.filled + WARMUP_GAP, at_least)

    def summary(self) -> dict:
        return {"warmup_s": None if self.start is None else round(self.start / RATE, 2), "reason": self.reason, "steps": self.steps}


def timing_check(y, start: int, n: int, warmup_blips, tail_blips) -> dict:
    """After the take: was the timing the same before the input, at its two blips and after it? `y` is the whole recorded stream (warm-up
    included), `start` where the input began in it, the blip lists are stream positions. Lags are against the response to the input's
    first blip (10.5 s in), the reference the trainer aligns with.

    checked        False when the responses could not be compared (no response, not alike)
    before_input   lag of the last warm-up blip (a step between it and 10.5 s falls on the first replicate: realigned above)
    between_blips  lag of the input's second blip, 11.5 s in
    after_input    lag of the blips after the input (median)
    moved_in_training  True when the timing moved after 10.5 s: the part the model is trained on is not all in step"""
    y = np.asarray(y)
    ref = start + BLIP_LOCATIONS[0]
    onset = blip_onset(y, ref)
    out = {"checked": False, "before_input": None, "between_blips": None, "after_input": None, "moved_in_training": False}
    if onset is None:
        return out
    corrs = []

    def lag(p):
        v, c = response_lag(y, ref, p, onset)
        corrs.append(float(c))
        return round(float(v), 3)

    if warmup_blips:
        out["before_input"] = lag(warmup_blips[-1])
    out["between_blips"] = lag(start + BLIP_LOCATIONS[1])
    tail = [lag(p) for p in tail_blips if p + RESP_SPAN + 8_192 <= len(y)]
    out["after_input"] = round(float(np.median(tail)), 3) if tail else None
    out["checked"] = bool(corrs) and min(corrs) >= MIN_CORR
    out["min_corr"] = round(min(corrs), 4) if corrs else None
    if out["checked"]:
        out["moved_in_training"] = bool(abs(out["between_blips"]) >= STEP_MIN or (out["after_input"] is not None and abs(out["after_input"]) >= STEP_MIN))
    return out


def analyze_take(rec, n_input: int = INPUT_V3_FRAMES, timing: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Align and cut a raw take; returns (aligned samples, report). The report never raises for a bad take: it lists `warnings`
    (keys the page translates) and `ok` says whether the take can go to the trainer as it is. `timing`: what the recording measured
    around the input (WarmupWatch.summary() + timing_check()), when it did; kept in the report and used for the `timing_step` check."""
    rec = np.asarray(rec, dtype=np.float64).ravel()
    warnings = []
    latency = detect_latency(rec)
    if latency["delay"] is None:
        cut = 0
        warnings.append("no_blip")
    else:
        cut = int(latency["recommended"])
        if latency["at_lookahead_limit"]:
            warnings.append("blip_at_limit")
    aligned = cut_and_pad(rec, n_input, cut)
    if len(rec) - max(cut, 0) < n_input:
        warnings.append("short_recording")
    residual = detect_latency(aligned) if latency["delay"] is not None else None
    esr = replicate_esr(aligned, n_input)
    replicate = replicate_detail(aligned, n_input)
    repaired = None
    if esr > ESR_REPLICATE_THRESHOLD:
        fix = realign_first_replicate(aligned, n_input)
        if fix is not None and replicate_esr(fix[0], n_input) <= ESR_REPLICATE_THRESHOLD:     # only if it really makes the two replicates agree
            repaired = {**fix[1], "esr_before": esr}
            aligned, esr = fix[0], replicate_esr(fix[0], n_input)
            replicate = {**replicate_detail(aligned, n_input), "repaired": repaired}
    peak = float(np.max(np.abs(aligned)))
    clipped = cal.count_clipped(aligned)
    peak_dbfs = cal.to_db(peak)
    noise_dbfs = cal.to_db(latency["background"])
    response_dbfs = cal.to_db(latency["response_peak"])
    if clipped:
        warnings.append("clip")
    elif peak_dbfs > -1.0:
        warnings.append("hot")
    if peak_dbfs < -24.0:
        warnings.append("low_signal")
    if latency["delay"] is not None and response_dbfs - noise_dbfs < 20.0:
        warnings.append("blip_weak")
    if esr > ESR_REPLICATE_THRESHOLD:
        warnings.append("esr_glitch" if replicate["stretch"] and replicate["esr_rest"] is not None and replicate["esr_rest"] <= ESR_REPLICATE_THRESHOLD
                        else "esr_high")
    elif repaired:
        warnings.append("esr_repaired")                           # informative: the take is fine, it says what was done
    if timing and timing.get("moved_in_training"):
        warnings.append("timing_step")                            # the driver moved the timing inside the part the model learns from
    report = {
        "n_input": n_input,
        "n_recorded": int(len(rec)),
        "latency": {"delay": latency["delay"], "cut": cut, "ms": None if latency["delay"] is None else round(latency["delay"] / RATE * 1000, 3)},
        "residual": None if residual is None else {"delay": residual["delay"], "recommended": residual["recommended"]},
        "replicate_esr": esr,
        "replicate": replicate,
        "peak_dbfs": peak_dbfs,
        "clipped_samples": clipped,
        "noise_dbfs": noise_dbfs,
        "blip_over_noise_db": None if latency["delay"] is None else response_dbfs - noise_dbfs,
        "warnings": warnings,
    }
    if timing:
        report["timing"] = timing
    blocking = {"no_blip", "clip", "short_recording", "esr_high", "esr_glitch", "timing_step"}
    report["ok"] = not (set(warnings) & blocking)
    return aligned, report


def click_latency(rec, positions, window: int = 20_000, floor: Optional[float] = None) -> Optional[int]:
    """Round trip of the interface alone (output cabled to input): median delay, in samples, of the first sample above a
    threshold after each click sent at `positions`. None if no click came back."""
    rec = np.asarray(rec, dtype=np.float64).ravel()
    if floor is None:
        floor = max(8.0 * float(np.percentile(np.abs(rec), 50)), 0.003)
    delays = []
    for p in positions:
        seg = np.abs(rec[p:p + window])
        hits = np.where(seg > floor)[0]
        if len(hits):
            delays.append(int(hits[0]))
    return int(np.median(delays)) if delays else None
