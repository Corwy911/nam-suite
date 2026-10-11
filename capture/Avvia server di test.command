#!/bin/bash
# Doppio click per avviare un server di prova, separato dalla vera app in Applicazioni: stessi dati
# (calibrazioni, WAV, modelli NAM, IR), ma niente ricompilazione - serve solo a provare le pagine nel
# browser mentre si sviluppa. Ferma con "Ferma server di test.command", nella stessa cartella.
set -euo pipefail
cd "$(dirname "$0")"
PORT=8790
LOG=".test-server.log"

if lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Il server di test è già acceso: http://127.0.0.1:$PORT"
else
  if [ ! -x venv/bin/python ]; then
    echo "Manca venv/: esegui prima  bash setup-mac.sh"
    read -n 1 -s -r -p "Premi un tasto per chiudere..."
    echo
    exit 1
  fi
  echo "Per calibrazione, cattura e monitor chitarra: usa solo questa pagina, lascia stare la finestra"
  echo "di \"NAM Capture\" se è aperta (non serve chiuderla, basta non toccarla)."
  echo "Per \"Prova dal vivo\" (NAM + IR): quella parte gira nel motore nativo, che esiste SOLO dentro"
  echo "l'app vera - apri (o lascia aperta) \"NAM Capture\" prima di premere \"Avvia\" su questa pagina,"
  echo "altrimenti dà \"il motore nativo non risponde\". Non serve toccarne la finestra, basta che sia avviata."
  echo
  nohup venv/bin/python capture_tool/app.py --port $PORT > "$LOG" 2>&1 &
  disown
  sleep 1
  if ! lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Non è partito: guarda \"$LOG\" per l'errore."
    read -n 1 -s -r -p "Premi un tasto per chiudere..."
    echo
    exit 1
  fi
  echo "Server di test avviato: http://127.0.0.1:$PORT"
fi
open "http://127.0.0.1:$PORT"
echo
echo "Questa finestra si può chiudere: il server resta acceso finché non usi \"Ferma server di test.command\"."
read -n 1 -s -r -p "Premi un tasto per chiudere questa finestra..."
echo
