"""Run the trainer's own latency calibration on a saved take: `python _trainer_latency.py <trainer folder> <take.npy>`.
Prints one JSON line. A separate process because the two trainers ship the same package name (`nam`)."""

import json
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

try:
    import numpy as np

    sys.path.insert(0, sys.argv[1])
    from nam.train import core

    y = np.load(sys.argv[2])
    result = core._calibrate_latency_v3(y, manual_available=False, show_plots=False)
    print(json.dumps({"delays": [int(d) for d in result.delays], "recommended": result.recommended}))
except Exception as e:  # the caller skips the comparison if the trainer does not offer this function
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
