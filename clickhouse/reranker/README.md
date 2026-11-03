# Servicio del reranker (D-097, B72)

Reordena los 120 tópicos de la fase 1 con **Qwen3-Reranker-4B (Q4_K_M)** en llama.cpp. Un proceso aparte, junto a Ollama; escucha solo en `127.0.0.1:11437`. Quyca lo llama desde `POST /app/theme/candidates/rerank` (contrato `02` §3.10).

## Qué hace
- Carga el modelo **una vez** al arrancar (todas las capas en GPU) y no lo descarga.
- **Una petición a la vez**, cola FIFO. Una petición que espera más de su `timeout_s` (60 s desde quyca) recibe 503 y se descarta.
- Receta de B62/B63: instrucción 1, documento `"{name}. {keywords}. {summary}"`, puntaje = P(sí) de los logits «yes»/«no», caché del prefijo compartido. Semillas de más de 2.000 tokens se cortan para el reranker (`seed_truncated` en la respuesta).
- API interna: `GET /health` y `POST /rerank` (ver el encabezado de `rerank_service.py`). No registra el texto de las peticiones.

## Arranque
```bash
bash run_reranker.sh        # primer plano; usa ~/ASCUN/reranker_venv y ~/ASCUN/modelos/qwen3-reranker-4b-q4_k_m.gguf
```
Variables: `RERANKER_GGUF`, `RERANKER_PORT` (11437), `RERANKER_NGL` (99), `RERANKER_THREADS` (6), `RERANKER_CTX` (3072), `RERANKER_VENV`.
En quyca: `RERANKER_URL=http://127.0.0.1:11437` (vacío desactiva la fase 2), `RERANKER_TIMEOUT_S=60`, `OLLAMA_NUM_GPU`.

## Memoria de la GPU de 4 GB (laptop)
El reranker residente ocupa ≈ 3,1 GB; los embeddings de Ollama al 100 % en GPU (3,8 GB) no caben juntos. Configuración elegida y medida en B72: **reranker en GPU y `OLLAMA_NUM_GPU=0` en quyca** (embeddings en CPU: 0,1-0,2 s en caliente, ≈ 13 s la primera carga tras 30 min sin uso; reordenar 120 tarda 9,5-11 s). Sin reranker (`out/b71/arranque_gpu.sh`) los embeddings van 100 % en GPU. En un servidor de ≥ 8 GB, `OLLAMA_NUM_GPU` vacío.

## Entorno
`llama-cpp-python` 0.3.36 compilado con CUDA 12.4 y las bibliotecas `nvidia-*-cu12` en un venv fuera del repositorio (`~/ASCUN/reranker_venv`, copia del entorno de B62). Los GGUF están en `~/ASCUN/modelos/` con `SHA256SUMS` (4B `941f7d1d…af49`, 0,6B `22c9979c…c9a48`).
