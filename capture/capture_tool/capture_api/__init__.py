"""HTTP API of the capture tool: /api/capture/*. Only for requests coming from this very PC (it drives sound into gear).

The audio itself lives in the engine process (capture/engine.py); these modules start it on demand, keep the calibrations
(capture/store.py), run the analysis of a finished take (capture/alignment.py) and hand the result over to the trainer's WAV library.
app.py passes its helpers in with `configure()` (no import of app here: it would be circular).

    core     shared context, engine session, error helpers, native-engine relay, shared state
    devices  interface + calibration steps      monitor  native engine's generic state proxy
    live     NAM + IR live tool                 takes    input file library + captures
    player   NAM player: model vs real gear, A/B
    trainer  hand-off to nam-trainer-gui, via a shared folder (no network call to the trainer itself)
"""

from fastapi import APIRouter, Depends, HTTPException, Request

import settings
from i18n import tr

from . import core, devices, live, monitor, player, takes, trainer, tuner
from .core import configure, shutdown
from .takes import organize_library, seed_standard_input


def require_local(request: Request) -> None:
    """The capture tool drives sound into gear: only requests coming from this very PC, even with the LAN switch on."""
    host = request.client.host if request.client else ""
    if not settings.is_local_client(host):
        raise HTTPException(403, tr("srv.cap.local_only"))


router = APIRouter(prefix="/api/capture", dependencies=[Depends(require_local)])
for _module in (devices, monitor, live, player, tuner, takes, trainer):
    router.include_router(_module.router)
