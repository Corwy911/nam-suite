# NAM Suite

**Train, capture and try [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) profiles — two apps, one installer.**

*[Versione italiana: README.it.md](README.it.md)*

| | |
|---|---|
| **NAM Trainer** | A web page (opens from the system-tray icon) that trains NAM models on your NVIDIA or AMD GPU — or just the CPU — with a queue, live progress, ESR and plots. Usable from the PC itself or from any device on your network. |
| **NAM Capture** | A calibrated **reamp capture** tool: it measures the levels of your audio interface in dBu, records the standard NAM test signal through your amp or pedal, checks and aligns the take, writes the levels into the model and sends everything to the Trainer. It also plays your models live: **NAM + IR test**, a **NAM player** with A/B against the real amp, a **pedal test** (pedal NAM in front of the amp's NAM) and a precise **tuner**. |

> **macOS: coming soon.** This first release (1.0) is for Windows. The code is already shared with the Mac version in progress.

## Download and install (Windows)

1. Open the **[Releases](../../releases)** page and download **all four files** into the **same folder**:
   `NAM-Suite-Setup.exe` and its three data parts `NAM-Suite-Setup-1.bin`, `-2.bin`, `-3.bin`
   (GitHub does not accept files over 2 GiB, so the installer is split).
   *Shortcut:* download only `Scarica-NAM-Suite.cmd` and double-click it: it downloads the four files, checks them and starts the setup.
2. Double-click `NAM-Suite-Setup.exe`. Choose the language and the components (Trainer, Capture, or both); the installer detects your graphics
   card and installs the right PyTorch (NVIDIA, AMD or CPU) **without needing internet**. Windows may warn about an unknown publisher
   (the files are not code-signed): *More info → Run anyway*.
3. Open **NAM Trainer** (it lives in the notification area) and **NAM Capture** from the Start menu or the desktop icons.

Check the downloads with `SHA256SUMS.txt` (same release page).

**Requirements:** Windows 10/11 64-bit, ~10 GB free while installing. For training: NVIDIA RTX 20-series or newer with driver 570+, *or* a
supported Radeon (RX 7000/9000, Adrenalin 26.2.2+, **experimental**: never tried by the authors on real hardware), *or* any PC (CPU only,
much slower). For NAM Capture: an audio interface with an **ASIO** driver, and a reamp box to capture amps and pedals.
The uninstaller removes everything (it first shows how many models, WAV files and captures it is about to delete and asks).

## What is in this repository

| Path | What |
|---|---|
| `trainer/` | NAM Trainer: FastAPI server and web page (`nam-web/`), the two trainers it can run (`trainers/`: the official `neural-amp-modeler` and *NAM Trainer Reloaded*, unmodified, with their licenses), tests |
| `capture/` | NAM Capture: Python server and web page (`capture_tool/`), real-time C++ audio engine (`native/`), the Windows shell (Win32 + WebView2 + ASIO) and the macOS shell in progress (`packaging/`), tests |
| `suite/` | The installer: `build.py` (stages Python, packages and both apps), `installer/NAMSuite.iss` (Inno Setup), the .NET tray launcher (`launcher/`), pinned package lists (`locks/`) |

Tests: `python -m unittest discover -s trainer/tests` and `python -m unittest discover -s capture/tests`; the C++ engine's own tests: `capture/native/run_tests.bat`.

### Build the installer yourself

Needs Windows with Python 3.12, Visual Studio Build Tools (C++), Inno Setup 6, and the free [Steinberg ASIO SDK](https://www.steinberg.net/developers/)
(**not included**: Steinberg's license does not allow redistributing it; see `capture/native/third_party/ASIO_SDK_SETUP.md`).
Then `build-installer.cmd all` (the first run downloads ~5.5 GB of pinned packages; the installer lands next to the repository).

## Credits and license

The code of NAM Suite is released under the **MIT License** ([LICENSE](LICENSE)). It bundles, vendors and installs third-party software under
their own licenses — see [THIRD_PARTY.md](THIRD_PARTY.md): above all Steven Atkinson's [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler)
and [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore), SlamminMofo's NAM Trainer Reloaded, PyTorch, and the
[NAM-Oversampler](https://github.com/DLC86/NAM-Oversampler) tuner code. *ASIO is a trademark and software of Steinberg Media Technologies GmbH.*
This project is independent: it is not affiliated with, or endorsed by, the authors of Neural Amp Modeler, NVIDIA, AMD, Steinberg or Microsoft.
