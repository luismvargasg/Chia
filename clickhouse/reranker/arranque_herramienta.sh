#!/usr/bin/env bash
# B72: one command that leaves the curation tool backend running on the laptop (D-097): ClickHouse and Ollama containers, the reranker service
# (127.0.0.1:11437) and quyca from the working tree (port 8081). Idempotent: whatever already answers is left alone.
# Usage: bash arranque_herramienta.sh              start what is missing, wait for /health, write the log
#        bash arranque_herramienta.sh --reiniciar  stop the reranker and quyca 8081 and start them again (Ollama is not touched)
#        bash arranque_herramienta.sh --detener    stop the reranker and quyca 8081 (frees ~3.1 GB of GPU)
# Log: ~/ASCUN/logs/arranque_herramienta.log (this script), reranker.log and quyca_8081.log (the processes). Exit code 0 = everything answers, 1 = something failed.
# Nothing here reads or prints credentials: quyca reads its own .env.dev. SENTRY_DSN is emptied so local tests do not report to Sentry.
# Settings for the 4 GB laptop (contract 02, section 3.10.4): reranker on the GPU, OLLAMA_NUM_GPU=20 for the embeddings (measured with distinct seeds: 0.4-1.1 s each; 0 would take 7-11 s). On a server with >= 8 GB leave OLLAMA_NUM_GPU empty.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; QUYCA="${QUYCA_DIR:-$HOME/ASCUN/backend/quyca}"; LOGS="$HOME/ASCUN/logs"; LOG="$LOGS/arranque_herramienta.log"
PORT_Q="${QUYCA_PORT:-8081}"; PORT_R="${RERANKER_PORT:-11437}"; NUMGPU="${OLLAMA_NUM_GPU-20}"; OLLAMA="http://127.0.0.1:11436"
mkdir -p "$LOGS"; say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
pid_on(){ ss -ltnp 2>/dev/null | grep ":$1 " | sed -E 's/.*pid=([0-9]+).*/\1/' | head -1; }
wait_for(){ for _ in $(seq 1 "${3:-60}"); do curl -s -m 3 -o /dev/null -f "$1" && return 0; sleep 2; done; say "ERROR: $2 no respondió en $(( ${3:-60} * 2 )) s"; return 1; }
stop_port(){ p="$(pid_on "$1")"; if [ -n "$p" ]; then kill "$p" 2>/dev/null && say "Detenido $2 (pid $p)"; for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(pid_on "$1")" ] && break; sleep 1; done; fi; }
case "${1:-}" in --detener) stop_port "$PORT_Q" "quyca $PORT_Q"; stop_port "$PORT_R" "reranker"; say "Herramienta detenida"; exit 0;; --reiniciar) stop_port "$PORT_Q" "quyca $PORT_Q"; stop_port "$PORT_R" "reranker";; esac
RC=0; say "=== arranque de la herramienta ==="
for c in clickhouse ollama-qwen; do
  if [ "$(docker inspect -f '{{.State.Running}}' $c 2>/dev/null)" != "true" ]; then say "El contenedor $c no corre: lo inicio"; docker start $c >/dev/null 2>&1 || { say "ERROR: no pude iniciar $c"; RC=1; }; fi
done
wait_for "$OLLAMA/api/tags" "Ollama" 30 || RC=1
if curl -s -m 3 -f "http://127.0.0.1:$PORT_R/health" >/dev/null; then say "Reranker ya responde en $PORT_R"; else
  say "Inicio el reranker (carga del modelo ~6 s)"; ( cd "$HERE" && exec setsid bash run_reranker.sh ) < /dev/null >> "$LOGS/reranker.log" 2>&1 &
  wait_for "http://127.0.0.1:$PORT_R/health" "reranker" 60 || RC=1; fi
if curl -s -m 3 -f "http://127.0.0.1:$PORT_Q/app/theme" >/dev/null; then say "Quyca ya responde en $PORT_Q"; else
  say "Inicio quyca (rama $(git -C "$QUYCA" branch --show-current), puerto $PORT_Q)"
  ( cd "$QUYCA" && exec setsid env ENVIRONMENT=dev APP_PORT="$PORT_Q" APP_DEBUG=False SENTRY_DSN= RERANKER_URL="http://127.0.0.1:$PORT_R" OLLAMA_NUM_GPU="$NUMGPU" .venv/bin/python quyca/app.py ) < /dev/null >> "$LOGS/quyca_$PORT_Q.log" 2>&1 &
  wait_for "http://127.0.0.1:$PORT_Q/app/theme" "quyca" 60 || RC=1; fi
# warm the embedding model with the same options quyca sends, only if Ollama has it unloaded (keeps the first user request fast; the container keeps it 24 h)
if ! docker exec ollama-qwen ollama ps 2>/dev/null | grep -q "qwen3-embedding"; then
  OPT=""; [ -n "$NUMGPU" ] && OPT=",\"options\":{\"num_gpu\":$NUMGPU}"
  curl -s -m 120 -o /dev/null -w "%{http_code}" "$OLLAMA/api/embed" -d "{\"model\":\"qwen3-embedding:4b\",\"input\":\"calentamiento\"$OPT}" | { read c; say "Embeddings precalentados (HTTP $c)"; }
else say "Los embeddings ya estaban cargados"; fi
say "Reranker: $(curl -s -m 3 http://127.0.0.1:$PORT_R/health)"; say "Quyca: HTTP $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT_Q/app/theme) en :$PORT_Q | GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"
[ $RC = 0 ] && say "OK: herramienta encendida" || say "FALLA: revise $LOG y los logs de $LOGS"; exit $RC
