# NAM Suite

**Addestra, cattura e prova i profili [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler): due app, un solo installatore.**

*[English version: README.md](README.md)*

| | |
|---|---|
| **NAM Trainer** | Una pagina web (si apre dall'icona nell'area di notifica) che addestra i modelli NAM sulla tua GPU NVIDIA o AMD — o solo sulla CPU — con coda, avanzamento in diretta, ESR e grafici. Si usa dal PC stesso o da qualunque dispositivo della rete. |
| **NAM Capture** | Uno strumento di **cattura reamp calibrata**: misura i livelli della tua scheda audio in dBu, registra il segnale di prova standard di NAM attraverso il tuo ampli o pedale, controlla e allinea la registrazione, scrive i livelli nel modello e manda tutto al Trainer. Suona anche i modelli dal vivo: **prova NAM + IR**, un **Lettore NAM** con confronto A/B con l'ampli vero, un **Test Pedali** (NAM del pedale davanti a quello dell'ampli) e un **accordatore** preciso. |

> **macOS: in arrivo.** Questa prima versione (1.0) è per Windows. Il codice è già condiviso con la versione Mac in lavorazione.

## Scarica e installa (Windows)

1. Apri la pagina **[Releases](../../releases)** e scarica **tutti e quattro i file** nella **stessa cartella**:
   `NAM-Suite-Setup.exe` e le sue tre parti di dati `NAM-Suite-Setup-1.bin`, `-2.bin`, `-3.bin`
   (GitHub non accetta file oltre i 2 GiB, quindi l'installatore è diviso).
   *Scorciatoia:* scarica solo `Scarica-NAM-Suite.cmd` e fai doppio clic: scarica i quattro file, li controlla e avvia l'installazione.
2. Doppio clic su `NAM-Suite-Setup.exe`. Scegli la lingua e i componenti (Trainer, Capture o entrambi); l'installatore riconosce la scheda video e
   installa il PyTorch giusto (NVIDIA, AMD o CPU) **senza internet**. Windows può avvisare che l'editore è sconosciuto (i file non sono firmati):
   *Ulteriori informazioni → Esegui comunque*.
3. Apri **NAM Trainer** (sta nell'area di notifica) e **NAM Capture** dal menu Start o dalle icone sul desktop.

Verifica i download con `SHA256SUMS.txt` (nella stessa pagina).

**Requisiti:** Windows 10/11 a 64 bit, circa 10 GB liberi durante l'installazione. Per l'addestramento: NVIDIA RTX serie 20 o successive con driver 570+,
*oppure* una Radeon supportata (RX 7000/9000, Adrenalin 26.2.2+, **sperimentale**: gli autori non l'hanno mai provata su una Radeon vera), *oppure*
un PC qualsiasi (solo CPU, molto più lento). Per NAM Capture: una scheda audio con driver **ASIO** e una reamp box per catturare ampli e pedali.
Il disinstallatore toglie tutto (prima mostra quanti modelli, WAV e catture sta per cancellare e chiede conferma).

## Cosa c'è nel repository

| Cartella | Cosa |
|---|---|
| `trainer/` | NAM Trainer: server FastAPI e pagina web (`nam-web/`), i due trainer che sa lanciare (`trainers/`: `neural-amp-modeler` ufficiale e *NAM Trainer Reloaded*, non modificati, con le loro licenze), test |
| `capture/` | NAM Capture: server Python e pagina web (`capture_tool/`), motore audio C++ in tempo reale (`native/`), il guscio Windows (Win32 + WebView2 + ASIO) e quello macOS in lavorazione (`packaging/`), test |
| `suite/` | L'installatore: `build.py` (prepara Python, pacchetti e le due app), `installer/NAMSuite.iss` (Inno Setup), il launcher .NET nell'area di notifica (`launcher/`), le liste dei pacchetti fissate (`locks/`) |

Test: `python -m unittest discover -s trainer/tests` e `python -m unittest discover -s capture/tests`; i test del motore C++: `capture/native/run_tests.bat`.

### Costruire l'installatore da soli

Servono Windows con Python 3.12, Visual Studio Build Tools (C++), Inno Setup 6 e il gratuito [Steinberg ASIO SDK](https://www.steinberg.net/developers/)
(**non incluso**: la licenza di Steinberg non permette di ridistribuirlo; vedi `capture/native/third_party/ASIO_SDK_SETUP.md`).
Poi `build-installer.cmd all` (la prima volta scarica circa 5,5 GB di pacchetti fissati; l'installatore finisce accanto al repository).

## Crediti e licenza

Il codice di NAM Suite è sotto **licenza MIT** ([LICENSE](LICENSE)). Include, vendorizza e installa software di terzi, ognuno sotto la propria licenza —
vedi [THIRD_PARTY.md](THIRD_PARTY.md): soprattutto [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) e
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) di Steven Atkinson, NAM Trainer Reloaded di SlamminMofo, PyTorch e il codice
dell'accordatore di [NAM-Oversampler](https://github.com/DLC86/NAM-Oversampler). *ASIO è un marchio e un software di Steinberg Media Technologies GmbH.*
Questo progetto è indipendente: non è affiliato né approvato dagli autori di Neural Amp Modeler, né da NVIDIA, AMD, Steinberg o Microsoft.
