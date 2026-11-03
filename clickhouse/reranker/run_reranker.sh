#!/usr/bin/env bash
# Starts the reranker service (D-097) with the durable CUDA venv (~/ASCUN/reranker_venv). Usage: bash run_reranker.sh   (foreground; log to stdout)
# Needs the GPU free enough for the 4B model (about 3.2 GB): on the 4 GB laptop unload the Ollama embedding model first or start Ollama with a reduced num_gpu.
V="${RERANKER_VENV:-$HOME/ASCUN/reranker_venv}"
export LD_LIBRARY_PATH="$(ls -d "$V"/lib/python3*/site-packages/nvidia/*/lib | tr '\n' ':')"
cd "$(dirname "${BASH_SOURCE[0]}")" && exec "$V/bin/python" -I rerank_service.py
