# Third-party software / Software di terze parti

NAM Suite's own code is MIT-licensed (see `LICENSE`). It includes, vendors, downloads or redistributes the following software, each under its
own license. Nothing here is legal advice; check the licenses before redistributing.

## In the repository

| Software | Author | License | Where |
|---|---|---|---|
| [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) 0.13.0 | Steven Atkinson | MIT | `trainer/trainers/official/` (unmodified `nam` package from PyPI, with its `LICENSE`) |
| [NAM Trainer Reloaded](https://github.com/SlamminMofo/NAM-Trainer-Reloaded) 1.0.1 | SlamminMofo (fork of neural-amp-modeler) | MIT | `trainer/trainers/reloaded/` (unmodified, with its `LICENSE`) |
| auraloss (inside both trainers) | Christian Steinmetz | Apache-2.0 | `trainer/trainers/*/nam/_dependencies/auraloss/LICENSE` |
| [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) v0.5.4 (`NAM/`, nlohmann/json) | Steven Atkinson | MIT | `capture/native/third_party/nam/` |
| [FFTConvolver](https://github.com/HiFi-LoFi/FFTConvolver) | HiFi-LoFi | MIT | `capture/native/third_party/fftconvolver/` |
| [Eigen](https://eigen.tuxfamily.org) (headers) | Eigen authors | MPL-2.0 | `capture/native/third_party/eigen/` |
| Tuner (YIN pitch detector, adapted) from [NAM-Oversampler](https://github.com/DLC86/NAM-Oversampler) | DLC86 / Steven Atkinson | MIT | `capture/native/engine/Tuner.h` (notice in `capture/native/third_party/tuner/LICENSE.txt`) |

Details of what was copied and where: `capture/native/third_party/VENDORED.md`. The Trainer's page can update the two trainer folders from the upstream releases.

## Not in the repository: Steinberg ASIO SDK

The Windows audio engine is built against the Steinberg ASIO SDK, which is **not** included (its license does not allow redistribution; see
`capture/native/third_party/ASIO_SDK_SETUP.md`). The compiled NAM Capture application is distributed as the SDK's license allows.
*ASIO is a trademark and software of Steinberg Media Technologies GmbH.*

## In the installer release (not in the repository)

The installer carries software that is installed offline:

- **Python 3.12** ([python-build-standalone](https://github.com/astral-sh/python-build-standalone)) — PSF License and the licenses of its components.
- **PyTorch / torchaudio** (BSD-3-Clause). The NVIDIA build (`+cu128`) contains NVIDIA CUDA runtime libraries, distributed under the
  [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/). The AMD build uses AMD's public ROCm-on-Windows preview packages (`repo.radeon.com`),
  under AMD's and the components' own licenses.
- **NumPy, SciPy, scikit-learn, matplotlib, librosa, PyTorch Lightning, FastAPI, Starlette, uvicorn, pydantic, sounddevice, soundfile, Hugging Face
  libraries, …** — BSD / MIT / Apache-2.0 style licenses (full pinned lists: `suite/locks/` and `suite/installer/requirements-capture.lock`).
- **.NET 8 runtime and Avalonia** (the tray launcher) — MIT.
- **Microsoft Visual C++ Redistributable 2015-2022 (x64)** and **Microsoft Edge WebView2 Runtime** — Microsoft's redistribution terms (each is installed only
  if the PC does not have it).
- **Inno Setup** (the installer's tool) — Jordan Russell's / Martijn Laan's license.

This project is independent and is not affiliated with or endorsed by any of the above.

---

## Italiano

Il codice di NAM Suite è sotto licenza MIT (`LICENSE`). Il repository contiene i due trainer non modificati (`trainer/trainers/`, entrambi MIT, con i
rispettivi `LICENSE`; dentro c'è anche auraloss, Apache-2.0) e codice di terzi copiato nel motore audio (NeuralAmpModelerCore, FFTConvolver, Eigen,
l'accordatore adattato da NAM-Oversampler; dettagli in `capture/native/third_party/VENDORED.md`). L'ASIO SDK di Steinberg **non** è incluso (la sua
licenza non ne permette la ridistribuzione). Il pacchetto dell'installatore (nella release, non nel repository) ridistribuisce Python, PyTorch (con le
librerie CUDA di NVIDIA nella versione NVIDIA, lo stack ROCm di AMD nella versione AMD), i pacchetti Python elencati in `suite/locks/`, il launcher .NET,
i runtime Visual C++ e WebView2 di Microsoft: ognuno resta sotto la propria licenza. Non è consulenza legale: controlla le licenze prima di ridistribuire.
