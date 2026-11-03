"""
load_hs_products.py

Carga única del catálogo de productos del Sistema Armonizado (HS 1992) a
ClickHouse, con su embedding. Es el equivalente de load_topics.py para
productos de comercio exterior.

Fuente: JSON de referencia de UNSD/Comtrade (H0.json, className "HS1992"):
    https://comtradeapi.un.org/files/v1/app/reference/H0.json
Cada entrada trae id (código, como texto), text ("0101 - Live horses..."),
parent, aggrlevel (0 total, 2 capítulo, 4 partida, 6 subpartida) e isLeaf.

Por qué el texto del embedding es jerárquico: las descripciones de 6 dígitos
casi nunca se entienden solas ("010290 - Other", "020110 - Carcasses and
half-carcasses"). Por eso se vectoriza:
    "{capítulo}. {partida}. {subpartida}"
y para las partidas de 4 dígitos "{capítulo}. {partida}". El prefijo
"search_document:" es de Nomic: hs_products sigue en Nomic (768 dim); el
catálogo de topics ya usa Qwen sin prefijo (D-061). La semilla usa "search_query:".

Esto NO es un pipeline recurrente: se corre a mano, una vez. Si UNSD cambia
la clasificación, se vuelve a correr con --force.

Uso:
    # Ver cómo queda el texto de cada producto, sin Ollama ni ClickHouse
    python load_hs_products.py --json-path data/H0.json --levels 4 --parse-only

    # Comprobar que Ollama responde, sin insertar nada
    python load_hs_products.py --json-path data/H0.json --levels 4 --limit 5 --dry-run

    # Carga real de las partidas de 4 dígitos (~1.240 productos)
    python load_hs_products.py --json-path data/H0.json --levels 4

    # Subpartidas de 6 dígitos (~5.000 productos) además de las de 4
    python load_hs_products.py --json-path data/H0.json --levels 4 6 --force

OJO con --force: trunca TODA la tabla hs_products (todos los niveles).

Variables de entorno (se leen de clickhouse/.env si existe):
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DB, OLLAMA_URL, OLLAMA_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import clickhouse_connect
from dotenv import load_dotenv

from embedding_profile import NOMIC_MODEL, NOMIC_URL

# Reutiliza la llamada a /api/embed (con reintentos) de load_topics.py.
from load_topics import get_embeddings_batch

CLASSIFICATION = "HS1992"
DDL_FILE = Path(__file__).resolve().parent.parent / "init" / "005_create_hs_products.sql"

TABLE_COLUMNS = [
    "hs_code",
    "level",
    "parent_code",
    "name",
    "chapter_code",
    "chapter_name",
    "heading_code",
    "heading_name",
    "embed_context",
    "classification",
    "embedding",
    "model_id",
]

# En HS, ":-" y ":--" marcan niveles intermedios dentro de la misma línea
# ("Horses :-- Pure-bred breeding animals"). Se normalizan a ": ".
SUBLEVEL_RE = re.compile(r"\s*:-+\s*")


def clean_text(raw: str, code: str) -> str:
    """Quita el prefijo "{código} - ", normaliza espacios y marcadores :-- y
    el punto final."""
    text = re.sub(rf"^\s*{re.escape(code)}\s*-\s*", "", raw or "")
    text = SUBLEVEL_RE.sub(": ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".").strip()


def build_context(level: int, chapter_name: str, heading_name: str, name: str) -> str:
    """Texto jerárquico que se vectoriza (sin el prefijo del modelo)."""
    if level == 2:
        parts = [name]
    elif level == 4:
        parts = [chapter_name, name]
    else:
        parts = [chapter_name, heading_name, name]
    cleaned: list[str] = []
    for part in parts:
        if part and (not cleaned or cleaned[-1] != part):
            cleaned.append(part)
    return ". ".join(cleaned)


def build_embedding_text(row: dict[str, Any]) -> str:
    # Prefijo obligatorio para texto de catálogo (igual que en topics).
    return f"search_document: {row['embed_context']}"


def parse_hs_json(path: Path, levels: list[int]) -> list[dict[str, Any]]:
    """Lee el JSON de UNSD y devuelve los productos de los niveles pedidos,
    con capítulo y partida resueltos y el texto de contexto ya armado."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results")
    if not results:
        raise ValueError("El JSON no trae la lista 'results'; no parece un H0.json de UNSD.")
    if data.get("className") not in (None, "HS1992"):
        print(f"AVISO: className es {data.get('className')!r}, no 'HS1992'.", file=sys.stderr)

    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in results}

    def ancestor(code: str, target_level: int) -> dict[str, Any] | None:
        node = by_id.get(code)
        guard = 0
        while node and guard < 10:
            if int(node["aggrlevel"]) == target_level:
                return node
            node = by_id.get(node["parent"])
            guard += 1
        return None

    products: list[dict[str, Any]] = []
    for r in results:
        code = r["id"]
        level = int(r["aggrlevel"])
        if not code.isdigit() or level not in levels:
            continue
        chapter = ancestor(code, 2)
        heading = ancestor(code, 4) if level >= 4 else None
        name = clean_text(r["text"], code)
        chapter_name = clean_text(chapter["text"], chapter["id"]) if chapter else ""
        heading_name = clean_text(heading["text"], heading["id"]) if heading and level == 6 else ""
        products.append({
            "hs_code": code,
            "level": level,
            "parent_code": r["parent"] if r["parent"] != "TOTAL" else "",
            "name": name,
            "chapter_code": chapter["id"] if chapter else "",
            "chapter_name": chapter_name,
            "heading_code": heading["id"] if heading and level == 6 else (code if level == 4 else ""),
            "heading_name": heading_name if level == 6 else (name if level == 4 else ""),
            "embed_context": build_context(level, chapter_name, heading_name, name),
        })
    return products


def run_ddl(client: Any, ddl_file: Path) -> None:
    """Ejecuta el DDL (una sentencia a la vez; clickhouse-connect no admite varias)."""
    lines = [re.sub(r"--.*$", "", ln) for ln in ddl_file.read_text(encoding="utf-8").splitlines()]
    for statement in "\n".join(lines).split(";"):
        if statement.strip():
            client.command(statement)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-path", required=True, type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[4], choices=[2, 4, 6],
                         help="Niveles a cargar: 2 capítulos, 4 partidas, 6 subpartidas (por defecto 4).")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-db", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--insert-batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Solo las primeras N filas (para probar).")
    parser.add_argument("--force", action="store_true", help="Truncar hs_products antes de cargar.")
    parser.add_argument("--dry-run", action="store_true", help="Calcular embeddings pero no insertar.")
    parser.add_argument("--parse-only", action="store_true",
                         help="Solo parsear y mostrar el texto de cada producto (sin Ollama ni ClickHouse).")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    ollama_url = args.ollama_url or NOMIC_URL
    ollama_model = args.ollama_model or NOMIC_MODEL  # hs_products stays on Nomic (768 dims); OLLAMA_MODEL is ignored
    ch_host = args.clickhouse_host or os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port = int(args.clickhouse_port or os.getenv("CLICKHOUSE_PORT", "8123"))
    ch_user = args.clickhouse_user or os.getenv("CLICKHOUSE_USER", "colav")
    ch_password = args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD", "")
    ch_db = args.clickhouse_db or os.getenv("CLICKHOUSE_DB", "theme_curation")

    print(f"Leyendo {args.json_path} (niveles {args.levels})")
    products = parse_hs_json(args.json_path, args.levels)
    if args.limit:
        products = products[: args.limit]
    by_level = {lvl: sum(1 for p in products if p["level"] == lvl) for lvl in sorted(set(args.levels))}
    print(f"{len(products)} productos a procesar: {by_level}")

    if args.parse_only:
        for p in products[:15]:
            print(f"  [{p['level']}] {p['hs_code']:>6}  {p['embed_context']}")
        short = [p for p in products if len(p["embed_context"].split()) < 4]
        print(f"\nTextos de menos de 4 palabras: {len(short)} de {len(products)} "
              f"(esos tendrán embeddings poco informativos).")
        return

    client = None
    if not args.dry_run:
        client = clickhouse_connect.get_client(
            host=ch_host, port=ch_port, username=ch_user, password=ch_password, database=ch_db,
        )
        run_ddl(client, DDL_FILE)
        existing = int(client.query("SELECT count() FROM hs_products").result_rows[0][0])
        if existing > 0:
            if not args.force:
                print(f"hs_products ya tiene {existing} filas. Usa --force para truncarla y recargar. Abortando.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"--force: truncando {existing} filas...")
            client.command("TRUNCATE TABLE hs_products")

    pending: list[list[Any]] = []
    total_inserted = 0
    start = time.time()
    embeddings: list[list[float]] = []

    for batch_start in range(0, len(products), args.embed_batch_size):
        batch = products[batch_start: batch_start + args.embed_batch_size]
        embeddings = get_embeddings_batch(ollama_url, ollama_model, [build_embedding_text(p) for p in batch])
        for p, emb in zip(batch, embeddings):
            pending.append([
                p["hs_code"], p["level"], p["parent_code"], p["name"],
                p["chapter_code"], p["chapter_name"], p["heading_code"], p["heading_name"],
                p["embed_context"], CLASSIFICATION, emb, ollama_model,
            ])
        done = batch_start + len(batch)
        print(f"  {done}/{len(products)} vectorizados ({time.time() - start:.0f}s)")

        if args.dry_run:
            continue
        if len(pending) >= args.insert_batch_size:
            client.insert("hs_products", pending, column_names=TABLE_COLUMNS)
            total_inserted += len(pending)
            pending = []

    if args.dry_run:
        print(f"\n--dry-run: {len(products)} embeddings calculados, nada insertado. "
              f"Dimensión: {len(embeddings[0])}")
        return
    if pending:
        client.insert("hs_products", pending, column_names=TABLE_COLUMNS)
        total_inserted += len(pending)
    print(f"\nListo. {total_inserted} productos insertados en {ch_db}.hs_products.")


if __name__ == "__main__":
    main()
