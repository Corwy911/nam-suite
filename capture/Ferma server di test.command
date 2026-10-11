#!/bin/bash
# Doppio click per fermare il server di prova avviato da "Avvia server di test.command".
set -euo pipefail
cd "$(dirname "$0")"
PORT=8790

PIDS=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
if [ -z "$PIDS" ]; then
  echo "Il server di test non era acceso."
else
  kill $PIDS
  sleep 1
  if lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    kill -9 $PIDS 2>/dev/null || true
  fi
  echo "Server di test fermato."
fi
read -n 1 -s -r -p "Premi un tasto per chiudere questa finestra..."
echo
