"""
load_topics.py

Carga única del catálogo de tópicos de OpenAlex a ClickHouse.

Lee el CSV con los 4.516 tópicos (topic_id, topic_name, subfield_*, field_*,
domain_*, keywords, summary, wikipedia_url), calcula el embedding de cada uno
vía Ollama (modelo qwen3-embedding:4b, sin prefijo; ver embedding_profile.py) y hace UN SOLO INSERT por fila, con todas las columnas
ya completas — incluida embedding.

Por qué no hay un paso de "cargar primero, vectorizar después": ClickHouse
(MergeTree) no está pensado para UPDATE frecuentes, son mutaciones
asíncronas y costosas. Insertar la fila completa de una sola vez evita esto
por completo.

Esto NO es un pipeline recurrente — no va en Airflow. El catálogo de OpenAlex
no cambia en la operación normal del sistema. Se corre a mano, una sola vez.
Si alguna vez OpenAlex actualiza su taxonomía, se vuelve a correr a mano con
--force.

Uso:
    # Primera carga
    python load_topics.py --csv-path /ruta/al/catalogo.csv

    # Probar con pocas filas antes de correr las 4.516 completas
    python load_topics.py --csv-path /ruta/al/catalogo.csv --limit 20

    # Solo verificar que Ollama responde, sin tocar ClickHouse
    python load_topics.py --csv-path /ruta/al/catalogo.csv --limit 5 --dry-run

    # Recargar desde cero (trunca la tabla primero)
    python load_topics.py --csv-path /ruta/al/catalogo.csv --force

Variables de entorno esperadas (se leen de clickhouse/.env si existe):
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DB, OLLAMA_URL, OLLAMA_MODEL
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import clickhouse_connect
import requests
from dotenv import load_dotenv

from embedding_profile import QWEN_MODEL, QWEN_URL, format_document, require_qwen
import os

CSV_COLUMNS = [
    "topic_id",
    "topic_name",
    "subfield_id",
    "subfield_name",
    "field_id",
    "field_name",
    "domain_id",
    "domain_name",
    "keywords",
    "summary",
    "wikipedia_url",
]

TABLE_COLUMNS = [
    "topic_id",
    "name",
    "summary",
    "keywords",
    "subfield_id",
    "subfield_name",
    "field_id",
    "field_name",
    "domain_id",
    "domain_name",
    "wikipedia_url",
    "embedding",
    "model_id",
]


def read_topics_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Al CSV le faltan columnas esperadas: {missing}")
        return list(reader)


def build_embedding_text(row: dict[str, str]) -> str:
    # Qwen3-Embedding: documents take no prefix (the seed takes an instruction).
    return format_document(f"{row['topic_name']}. {row['keywords']}. {row['summary']}")


def get_embeddings_batch(
    ollama_url: str,
    model: str,
    texts: list[str],
    max_retries: int = 3,
) -> list[list[float]]:
    """Llama a /api/embed (NO el /api/embeddings legado) con un lote de
    textos. Ollama soporta 'input' como lista y devuelve un vector por
    entrada, en el mismo orden."""
    payload = {"model": model, "input": texts}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{ollama_url}/api/embed", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                raise ValueError(
                    f"Ollama devolvió {len(embeddings or [])} embeddings "
                    f"para {len(texts)} textos — no coincide."
                )
            return embeddings
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  [intento {attempt}/{max_retries}] fallo llamando a Ollama: {exc}. "
                  f"Reintentando en {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Ollama falló después de {max_retries} intentos: {last_error}")


def get_existing_count(client: Any) -> int:
    result = client.query("SELECT count() FROM topics")
    return int(result.result_rows[0][0])


def truncate_topics(client: Any) -> None:
    client.command("TRUNCATE TABLE topics")


def insert_rows(client: Any, rows: list[list[Any]]) -> None:
    client.insert("topics", rows, column_names=TABLE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-db", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=16,
                         help="Cuántos tópicos se mandan juntos a Ollama por llamada.")
    parser.add_argument("--insert-batch-size", type=int, default=200,
                         help="Cuántas filas se insertan juntas en ClickHouse por INSERT.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Solo procesar las primeras N filas (para probar).")
    parser.add_argument("--force", action="store_true",
                         help="Si la tabla ya tiene datos, truncarla antes de recargar.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Calcular embeddings pero no insertar nada en ClickHouse.")
    args = parser.parse_args()

    load_dotenv(args.env_file)

    ollama_url = args.ollama_url or os.getenv("OLLAMA_URL", QWEN_URL)
    ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", QWEN_MODEL)
    require_qwen(ollama_model)
    ch_host = args.clickhouse_host or os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port = int(args.clickhouse_port or os.getenv("CLICKHOUSE_PORT", "8123"))
    ch_user = args.clickhouse_user or os.getenv("CLICKHOUSE_USER", "colav")
    ch_password = args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD", "")
    ch_db = args.clickhouse_db or os.getenv("CLICKHOUSE_DB", "theme_curation")

    print(f"Leyendo CSV: {args.csv_path}")
    rows = read_topics_csv(args.csv_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} tópicos a procesar.")

    client = None
    if not args.dry_run:
        client = clickhouse_connect.get_client(
            host=ch_host, port=ch_port, username=ch_user, password=ch_password, database=ch_db,
        )
        existing = get_existing_count(client)
        if existing > 0:
            if not args.force:
                print(
                    f"La tabla topics ya tiene {existing} filas. "
                    f"Usa --force si quieres truncarla y recargar desde cero. Abortando.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"--force: truncando {existing} filas existentes...")
            truncate_topics(client)

    pending_rows: list[list[Any]] = []
    total_inserted = 0
    start = time.time()

    for batch_start in range(0, len(rows), args.embed_batch_size):
        batch = rows[batch_start: batch_start + args.embed_batch_size]
        texts = [build_embedding_text(r) for r in batch]

        embeddings = get_embeddings_batch(ollama_url, ollama_model, texts)

        for row, embedding in zip(batch, embeddings):
            pending_rows.append([
                row["topic_id"],
                row["topic_name"],
                row["summary"],
                row["keywords"],
                row["subfield_id"],
                row["subfield_name"],
                row["field_id"],
                row["field_name"],
                row["domain_id"],
                row["domain_name"],
                row["wikipedia_url"],
                embedding,
                ollama_model,
            ])

        done = batch_start + len(batch)
        elapsed = time.time() - start
        print(f"  {done}/{len(rows)} tópicos vectorizados ({elapsed:.0f}s transcurridos)")

        if args.dry_run:
            continue

        if len(pending_rows) >= args.insert_batch_size:
            insert_rows(client, pending_rows)
            total_inserted += len(pending_rows)
            pending_rows = []

    if args.dry_run:
        print(f"\n--dry-run: {len(rows)} embeddings calculados correctamente, nada insertado.")
        print(f"Dimensión del primer embedding: {len(embeddings[0])}")
        return

    if pending_rows:
        insert_rows(client, pending_rows)
        total_inserted += len(pending_rows)

    print(f"\nListo. {total_inserted} tópicos insertados en {ch_db}.topics.")


if __name__ == "__main__":
    main()
