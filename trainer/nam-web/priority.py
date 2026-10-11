"""Gives a training run everything the machine has: top scheduling priority, no power throttling, the fast cores, no sleep.

Why it is needed: a training of a NAM model is a small network driven by a Python loop - the GPU waits for the CPU more than the other way round
(20-30 % GPU load is normal for it, and what decides the speed is how fast ONE CPU thread runs). Windows 11 treats a process without a window as a
background one and may put it on the "efficiency" class (EcoQoS): on a hybrid CPU (Intel 12th gen and later: P-cores + E-cores) that thread then
lands on an E-core, a fraction of the speed, while the GPU idles. The training worker therefore, on Windows:
  - raises its priority class to HIGH (not REALTIME: that one can starve the mouse, the keyboard and the audio drivers),
  - opts out of power throttling (execution speed and timer resolution),
  - on a hybrid CPU, restricts itself to the fastest class of cores (CPU sets: a preference the system keeps, not a hard affinity),
  - tells Windows the machine must stay awake while it runs.
Every step is best effort: a failure of one is reported in the log and never stops the training. `NAM_NO_BOOST=1` in the environment turns it all off
(only for measuring what it is worth). Elsewhere (macOS, Linux) the process gets a better `nice` where the system allows it.
"""

import os
import sys

HIGH_PRIORITY_CLASS = 0x00000080
PROCESS_POWER_THROTTLING = 4                    # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
THREAD_POWER_THROTTLING = 3                     # THREAD_INFORMATION_CLASS.ThreadPowerThrottling
THROTTLE_EXECUTION_SPEED = 0x1
THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def _fast_core_ids(kernel32) -> list:
    """System CPU set ids of the cores with the highest efficiency class (the P-cores of a hybrid CPU); [] if all cores are alike."""
    import ctypes
    from ctypes import wintypes
    needed = wintypes.ULONG(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), kernel32.GetCurrentProcess(), 0)
    if needed.value == 0:
        return []
    buffer = (ctypes.c_ubyte * needed.value)()
    if not kernel32.GetSystemCpuSetInformation(ctypes.addressof(buffer), needed.value, ctypes.byref(needed), kernel32.GetCurrentProcess(), 0):
        return []
    entries, offset = [], 0
    while offset + 24 <= needed.value:
        size = int.from_bytes(bytes(buffer[offset:offset + 4]), "little")
        kind = int.from_bytes(bytes(buffer[offset + 4:offset + 8]), "little")
        if size < 24:
            break
        if kind == 0:                            # CpuSetInformation: Id @8, Group @12, ..., EfficiencyClass @18
            entries.append((int.from_bytes(bytes(buffer[offset + 8:offset + 12]), "little"), buffer[offset + 18]))
        offset += size
    classes = {c for _, c in entries}
    if len(classes) < 2:
        return []
    best = max(classes)
    return [i for i, c in entries if c == best]


def boost_process() -> str:
    """Applies the above to THIS process and returns one line saying what was done (for the log of the run)."""
    if os.environ.get("NAM_NO_BOOST"):
        return "priority boost: off (NAM_NO_BOOST)"
    done, failed = [], []
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        void = ctypes.c_void_p                       # handles: the "current process / thread" pseudo-handles are -1 / -2, which must not be turned into an int
        k.GetCurrentProcess.restype = void
        k.GetCurrentThread.restype = void
        k.SetPriorityClass.argtypes = [void, wintypes.DWORD]
        k.SetProcessInformation.argtypes = [void, ctypes.c_int, void, wintypes.DWORD]
        k.SetThreadInformation.argtypes = [void, ctypes.c_int, void, wintypes.DWORD]
        k.GetSystemCpuSetInformation.argtypes = [void, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), void, wintypes.ULONG]
        k.SetProcessDefaultCpuSets.argtypes = [void, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG]
        k.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        process = k.GetCurrentProcess()

        class ThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        def attempt(name, ok):
            (done if ok else failed).append(name)

        attempt("HIGH priority", bool(k.SetPriorityClass(process, HIGH_PRIORITY_CLASS)))
        state = ThrottlingState(1, THROTTLE_EXECUTION_SPEED | THROTTLE_IGNORE_TIMER_RESOLUTION, 0)    # StateMask 0 with the control bit set = opted OUT
        attempt("no power throttling", bool(k.SetProcessInformation(process, PROCESS_POWER_THROTTLING, ctypes.addressof(state), ctypes.sizeof(state))))
        k.SetThreadInformation(k.GetCurrentThread(), THREAD_POWER_THROTTLING, ctypes.addressof(state), ctypes.sizeof(state))
        try:
            fast = _fast_core_ids(k)
            if fast:
                ids = (wintypes.ULONG * len(fast))(*fast)
                attempt(f"fast cores only ({len(fast)} threads)", bool(k.SetProcessDefaultCpuSets(process, ids, len(fast))))
        except Exception as e:                                                                         # an older Windows without CPU sets
            failed.append(f"fast cores ({type(e).__name__})")
        attempt("stay awake", bool(k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)))
    else:
        try:
            os.nice(-10)
            done.append("nice -10")
        except (OSError, AttributeError):
            failed.append("nice (not allowed)")
    return "priority boost: " + (", ".join(done) or "nothing") + (f" | not possible: {', '.join(failed)}" if failed else "")
