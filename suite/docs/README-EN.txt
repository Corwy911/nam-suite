NAM Suite
=========

Two programs for Neural Amp Modeler models, one installer:

  NAM Trainer   trains the models (.nam). Opens from its desktop icon and stays running next to the Windows
                clock (notification area). From that icon: open the page, open the models / WAV folders,
                "Start with Windows", "Quit" (right click). Quitting stops everything.
  NAM Capture   calibrated capture with your audio interface (ASIO). Opens from its icon and closes
                completely when you quit: nothing stays in the background.

Where things are
----------------
  NAM Trainer    <install folder>\trainer\   models in "NAM Generati", WAV files in "File WAV caricati"
  NAM Capture    <install folder>\capture\   its data (captures, IRs, models, calibrations):
                 %LOCALAPPDATA%\NAM Capture
  The default install folder is %LOCALAPPDATA%\Programs\NAM Suite (no administrator rights needed).

Together
--------
  With NAM Trainer installed, set NAM Capture's "Trainer address" to  localhost :
  "Send to NAM Trainer" puts the capture (with its dBu levels) into the trainer's library, ready for "New training".

Network (LAN)
-------------
  The trainer's page can also be used from other PCs on the network (LAN switch in the page).
  The first time, Windows asks to allow Python on the network: allow it to use the page over the LAN.

Uninstalling
------------
  Settings > Apps > NAM Suite > Uninstall (or the Start menu shortcut).
  EVERYTHING is removed: programs, models, WAV files, the capture data, shortcuts, registry entries and
  (if you agree) the firewall rule. Before deleting, the uninstaller shows how many models and files there are
  and asks for confirmation: they are your work and cannot be recovered.
