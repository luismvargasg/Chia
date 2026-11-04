#!/usr/bin/env bash
# B72: one command that leaves the curation tool backend running on the laptop (D-097): ClickHouse and Ollama containers, the reranker service
# (127.0.0.1:11437) and the quyca-dev container (port 8080, D-099). Idempotent: whatever already answers is left alone.
# Usage: bash arranque_herramienta.sh              start what is missing, wait for /health, write the log
#        bash arranque_herramienta.sh --reiniciar  stop the reranker and rebuild quyca-dev (make down-dev && make up-dev, ~2 min; 8080 is down meanwhile). Ollama is not touched
#        bash arranque_herramienta.sh --detener    stop the reranker and quyca-dev (frees ~3.1 GB of GPU)
#        bash arranque_herramienta.sh --solo-reranker   restart only the reranker service
# Log: ~/ASCUN/logs/arranque_herramienta.log (this script), reranker.log (the service) and quyca_build.log (make output). Exit code 0 = everything answers, 1 = something failed.
# Nothing here reads or prints credentials. quyca-dev is built by the Makefile of quyca and reads its own .env.dev (git-ignored), which holds OLLAMA_NUM_GPU=20, RERANKER_URL and RERANKER_TIMEOUT_S
# (laptop of 4 GB, contract 02 section 3.10.4; on a server with >= 8 GB leave OLLAMA_NUM_GPU empty). The container uses the host network, so it reaches ClickHouse, ollama-qwen and the reranker on localhost.
# The code in the container is the working tree of quyca at build time: commit or stash what you do not want in it.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; QUYCA="${QUYCA_DIR:-$HOME/ASCUN/backend/quyca}"; LOGS="$HOME/ASCUN/logs"; LOG="$LOGS/arranque_herramienta.log"
PORT_Q="${QUYCA_PORT:-8080}"; PORT_R="${RERANKER_PORT:-11437}"; NUMGPU="${OLLAMA_NUM_GPU-20}"; OLLAMA="http://127.0.0.1:11436"
mkdir -p "$LOGS"; say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
pid_on(){ ss -ltnp 2>/dev/null | grep ":$1 " | sed -E 's/.*pid=([0-9]+).*/\1/' | head -1; }
wait_for(){ for _ in $(seq 1 "${3:-60}"); do curl -s -m 3 -o /dev/null -f "$1" && return 0; sleep 2; done; say "ERROR: $2 no respondió en $(( ${3:-60} * 2 )) s"; return 1; }
stop_port(){ p="$(pid_on "$1")"; if [ -n "$p" ]; then kill "$p" 2>/dev/null && say "Detenido $2 (pid $p)"; for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(pid_on "$1")" ] && break; sleep 1; done; fi; }
quyca_down(){ say "Detengo quyca-dev (make down-dev)"; (cd "$QUYCA" && make down-dev) < /dev/null >> "$LOGS/quyca_build.log" 2>&1; }
case "${1:-}" in
  --detener) quyca_down; stop_port "$PORT_R" "reranker"; say "Herramienta detenida"; exit 0;;
  --reiniciar) quyca_down; stop_port "$PORT_R" "reranker";;
  --solo-reranker) stop_port "$PORT_R" "reranker";;
esac
RC=0; say "=== arranque de la herramienta ==="
for c in clickhouse ollama-qwen; do
  if [ "$(docker inspect -f '{{.State.Running}}' $c 2>/dev/null)" != "true" ]; then say "El contenedor $c no corre: lo inicio"; docker start $c >/dev/null 2>&1 || { say "ERROR: no pude iniciar $c"; RC=1; }; fi
done
wait_for "$OLLAMA/api/tags" "Ollama" 30 || RC=1
if curl -s -m 3 -f "http://127.0.0.1:$PORT_R/health" >/dev/null; then say "Reranker ya responde en $PORT_R"; else
  say "Inicio el reranker (carga del modelo ~6 s)"; ( cd "$HERE" && exec setsid bash run_reranker.sh ) < /dev/null >> "$LOGS/reranker.log" 2>&1 &
  wait_for "http://127.0.0.1:$PORT_R/health" "reranker" 60 || RC=1; fi
if curl -s -m 3 -f "http://127.0.0.1:$PORT_Q/theme" >/dev/null; then say "Quyca-dev ya responde en $PORT_Q"; else
  say "Construyo y levanto quyca-dev (rama $(git -C "$QUYCA" branch --show-current), commit $(git -C "$QUYCA" rev-parse --short HEAD)); make up-dev tarda ~2 min"
  [ "$(docker ps -a --format '{{.Names}}' | grep -cx quyca-dev)" != "0" ] && (cd "$QUYCA" && make down-dev) < /dev/null >> "$LOGS/quyca_build.log" 2>&1
  (cd "$QUYCA" && make up-dev) < /dev/null >> "$LOGS/quyca_build.log" 2>&1 || { say "ERROR: make up-dev falló (ver $LOGS/quyca_build.log)"; RC=1; }
  wait_for "http://127.0.0.1:$PORT_Q/theme" "quyca-dev" 60 || RC=1; fi
# warm the embedding model with the same options quyca sends, only if Ollama has it unloaded (keeps the first user request fast; the container keeps it 24 h)
if ! docker exec ollama-qwen ollama ps 2>/dev/null | grep -q "qwen3-embedding"; then
  OPT=""; [ -n "$NUMGPU" ] && OPT=",\"options\":{\"num_gpu\":$NUMGPU}"
  curl -s -m 120 -o /dev/null -w "%{http_code}" "$OLLAMA/api/embed" -d "{\"model\":\"qwen3-embedding:4b\",\"input\":\"calentamiento\"$OPT}" | { read c; say "Embeddings precalentados (HTTP $c)"; }
else say "Los embeddings ya estaban cargados"; fi
say "Reranker: $(curl -s -m 3 http://127.0.0.1:$PORT_R/health)"; say "Quyca-dev: HTTP $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT_Q/theme) en :$PORT_Q | GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"
[ $RC = 0 ] && say "OK: herramienta encendida" || say "FALLA: revise $LOG y los logs de $LOGS"; exit $RC
