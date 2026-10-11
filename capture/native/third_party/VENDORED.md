# Librerie vendorizzate

Codice di terzi copiato qui (non un submodule): niente CMake/FetchContent in questo progetto, si
compila tutto con lo stesso `clang++` semplice di `native/build.sh`. Nessuna di queste è mai
modificata a mano - solo copiata dal repository originale.

## fftconvolver/

github.com/HiFi-LoFi/FFTConvolver, licenza MIT (vedi `fftconvolver/COPYING.txt`).
Commit `f2cdeb04c42141d2caec19ca4f137398b2a76b85` (2017-04-01), tutto il repo (è piccolo: 8 file,
nessuna dipendenza esterna). Convoluzione partizionata in due stadi per le risposte all'impulso (IR).

## eigen/

gitlab.com/libeigen/eigen, licenza MPL2 (vedi `eigen/COPYING.MPL2`). Solo la cartella `Eigen/`
(header-only, nessun sorgente .cpp da compilare) - non l'intero repository (test, doc, benchmark
esclusi). Commit `bc3b39870ecb690a623a3f49149a358b95c5781d`, lo stesso pinnato come submodule da
NeuralAmpModelerCore v0.5.4.

## nam/

github.com/sdatkinson/NeuralAmpModelerCore, licenza MIT (vedi `nam/LICENSE.txt`). Tag `v0.5.4`,
solo la cartella `NAM/` (il motore DSP vero e proprio: caricamento del file `.nam`, WaveNet) e
`Dependencies/nlohmann/json.hpp` (per leggere il JSON del modello). Non l'intero repository (tool
di build, test, esempi esclusi - i nostri test nativi restano in `native/engine/tests/`).

## tuner/ (codice adattato, non copiato)

github.com/DLC86/NAM-Oversampler (`NeuralAmpModeler/Tuner.h`), licenza MIT, copyright Steven Atkinson (testo in `tuner/LICENSE.txt`).
A differenza delle cartelle sopra, qui il codice è **modificato**: `native/engine/Tuner.h` parte dal rilevatore YIN di quel file (finestra a
24 kHz, soglia 0,15, mediana su 5 letture, smorzamento) e vi aggiunge la rifinitura a frequenza piena su più periodi, la misura della
fondamentale (picco spettrale), il campo 28-1400 Hz e l'analisi fuori dal thread audio. Misure e test in `native/engine/tests/test_tuner.cpp`.
