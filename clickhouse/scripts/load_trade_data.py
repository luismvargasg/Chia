"""
load_trade_data.py

Procesa UNA SOLA VEZ un CSV del Atlas of Economic Complexity (Growth Lab,
Harvard) y deja en ClickHouse solo lo que necesita la herramienta:

  1. trade_country_year: las filas de los países pedidos (por defecto
     Colombia, country_id 170), con todas las columnas del CSV.
  2. trade_world_year: un agregado mundial por producto y año (suma de
     exportaciones, número de exportadores, mayor exportador y su valor).

El CSV (451,7 MB el de 4 dígitos, 1,0 GB el de 6) se lee con DuckDB en
streaming, no entra completo en memoria y no se guarda en ClickHouse.

Archivos soportados (Harvard Dataverse, doi 10.7910/DVN/T4CHWJ):
  - hs92_country_product_year_4.csv  -> trae export_rca, distance, cog, pci
  - hs92_country_product_year_6.csv  -> solo 8 columnas, sin complejidad
El nivel (4 o 6) se deduce del largo de product_hs92_code. Los códigos se leen
como TEXTO (empiezan con cero). No se unen niveles por product_id: los ids
solo valen dentro de su clasificación.

Cita obligatoria al usar estos datos:
  The Growth Lab at Harvard University, 2026, "International Trade Data (HS92)",
  https://doi.org/10.7910/DVN/T4CHWJ, Harvard Dataverse

Uso:
    # Probar sin ClickHouse: deja CSV de resultado en --out-dir
    python load_trade_data.py --csv-path hs92_country_product_year_4.csv --dry-run --out-dir salida/

    # Carga real (4 dígitos, Colombia)
    python load_trade_data.py --csv-path hs92_country_product_year_4.csv

    # Otros países además de Colombia (por ejemplo para comparar)
    python load_trade_data.py --csv-path hs92_country_product_year_4.csv --country-id 170 76 484

Es idempotente: las tablas son ReplacingMergeTree, así que volver a correrlo
reemplaza las filas con la misma clave. Leer siempre con FINAL.

Variables de entorno (se leen de clickhouse/.env si existe):
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DB
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import clickhouse_connect
import duckdb
from dotenv import load_dotenv

DDL_FILE = Path(__file__).resolve().parent.parent / "init" / "006_create_trade_data.sql"

# Tipos explícitos: evita el muestreo de DuckDB y, sobre todo, que lea los
# códigos HS como número (se perderían los ceros iniciales).
CSV_TYPES = {
    "country_id": "BIGINT",
    "country_iso3_code": "VARCHAR",
    "product_id": "BIGINT",
    "product_hs92_code": "VARCHAR",
    "year": "INTEGER",
    "export_value": "DOUBLE",
    "import_value": "DOUBLE",
    "global_market_share": "DOUBLE",
    "export_rca": "DOUBLE",
    "distance": "DOUBLE",
    "cog": "DOUBLE",
    "pci": "DOUBLE",
}
REQUIRED = ["country_id", "country_iso3_code", "product_hs92_code", "year",
            "export_value", "import_value", "global_market_share"]
OPTIONAL = ["export_rca", "distance", "cog", "pci"]

COUNTRY_COLUMNS = [
    "country_id", "country_iso3", "hs_code", "level", "year", "export_value", "import_value",
    "global_market_share", "export_rca", "distance", "cog", "pci", "source_file",
]
WORLD_COLUMNS = [
    "hs_code", "level", "year", "world_export_value", "n_exporters", "top_exporter_iso3",
    "top_export_value", "pci", "source_file",
]


def run_ddl(client: Any, ddl_file: Path) -> None:
    lines = [re.sub(r"--.*$", "", ln) for ln in ddl_file.read_text(encoding="utf-8").splitlines()]
    for statement in "\n".join(lines).split(";"):
        if statement.strip():
            client.command(statement)


def read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def open_csv(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Crea la vista 't' sobre el CSV con tipos explícitos. Devuelve las columnas."""
    header = read_header(path)
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError(f"Al CSV le faltan columnas esperadas: {missing}")
    types = {c: CSV_TYPES[c] for c in header if c in CSV_TYPES}
    types_sql = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    safe_path = str(path).replace("'", "''")
    con.execute(
        f"CREATE VIEW t AS SELECT * FROM read_csv('{safe_path}', header=true, types={{{types_sql}}})"
    )
    return header


def opt(col: str, header: list[str], expr: str | None = None) -> str:
    """Expresión SQL de una columna opcional, o NULL si el CSV no la trae."""
    return (expr or col) if col in header else "NULL"


def extract_country_rows(con, header: list[str], country_ids: list[int], source: str) -> list[list[Any]]:
    ids = ", ".join(str(int(i)) for i in country_ids)
    rows = con.execute(f"""
        SELECT country_id, country_iso3_code, product_hs92_code,
               length(product_hs92_code) AS level, year,
               CAST(round(export_value) AS BIGINT), CAST(round(import_value) AS BIGINT),
               global_market_share,
               {opt('export_rca', header)}, {opt('distance', header)},
               {opt('cog', header)}, {opt('pci', header)}
        FROM t
        WHERE country_id IN ({ids})
          AND regexp_matches(product_hs92_code, '^[0-9]+$')
        ORDER BY country_id, product_hs92_code, year
    """).fetchall()
    return [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], source] for r in rows]


def extract_world_rows(con, header: list[str], source: str) -> list[list[Any]]:
    rows = con.execute(f"""
        SELECT product_hs92_code AS hs_code, length(product_hs92_code) AS level, year,
               CAST(round(sum(export_value)) AS BIGINT) AS world_export_value,
               CAST(count(*) FILTER (WHERE export_value > 0) AS INTEGER) AS n_exporters,
               arg_max(country_iso3_code, export_value) AS top_exporter_iso3,
               CAST(round(max(export_value)) AS BIGINT) AS top_export_value,
               {opt('pci', header, 'avg(pci)')} AS pci
        FROM t
        WHERE regexp_matches(product_hs92_code, '^[0-9]+$')
        GROUP BY product_hs92_code, year
        ORDER BY product_hs92_code, year
    """).fetchall()
    return [list(r) + [source] for r in rows]


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--country-id", type=int, nargs="+", default=[170],
                         help="country_id del Atlas (por defecto 170 = Colombia).")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-db", default=None)
    parser.add_argument("--insert-batch-size", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true",
                         help="No inserta en ClickHouse; escribe los resultados como CSV en --out-dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    load_dotenv(args.env_file)
    ch_host = args.clickhouse_host or os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port = int(args.clickhouse_port or os.getenv("CLICKHOUSE_PORT", "8123"))
    ch_user = args.clickhouse_user or os.getenv("CLICKHOUSE_USER", "colav")
    ch_password = args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD", "")
    ch_db = args.clickhouse_db or os.getenv("CLICKHOUSE_DB", "theme_curation")

    start = time.time()
    con = duckdb.connect()
    header = open_csv(con, args.csv_path)
    source = args.csv_path.name
    missing_opt = [c for c in OPTIONAL if c not in header]
    print(f"CSV: {source}. Columnas opcionales ausentes: {missing_opt or 'ninguna'}")

    levels = [r[0] for r in con.execute(
        "SELECT DISTINCT length(product_hs92_code) FROM t ORDER BY 1").fetchall()]
    print(f"Niveles encontrados (largo del código): {levels}")

    print(f"Extrayendo filas de country_id {args.country_id}...")
    country_rows = extract_country_rows(con, header, args.country_id, source)
    print(f"  {len(country_rows)} filas ({time.time() - start:.0f}s)")
    if not country_rows:
        print("ERROR: no hay filas para esos country_id. Revisa el id (Colombia = 170).", file=sys.stderr)
        sys.exit(1)

    print("Calculando agregados mundiales (lee el CSV completo)...")
    world_rows = extract_world_rows(con, header, source)
    print(f"  {len(world_rows)} filas ({time.time() - start:.0f}s)")

    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "trade_country_year.csv", COUNTRY_COLUMNS, country_rows)
        write_csv(args.out_dir / "trade_world_year.csv", WORLD_COLUMNS, world_rows)
        print(f"\n--dry-run: resultados en {args.out_dir}/ (nada insertado).")
        return

    client = clickhouse_connect.get_client(
        host=ch_host, port=ch_port, username=ch_user, password=ch_password, database=ch_db,
    )
    run_ddl(client, DDL_FILE)
    for table, columns, rows in (
        ("trade_country_year", COUNTRY_COLUMNS, country_rows),
        ("trade_world_year", WORLD_COLUMNS, world_rows),
    ):
        for i in range(0, len(rows), args.insert_batch_size):
            client.insert(table, rows[i: i + args.insert_batch_size], column_names=columns)
        print(f"  {len(rows)} filas insertadas en {ch_db}.{table}")
    print(f"\nListo ({time.time() - start:.0f}s). Recuerda citar: The Growth Lab at Harvard University, 2026, "
          f"International Trade Data (HS92), https://doi.org/10.7910/DVN/T4CHWJ, Harvard Dataverse.")


if __name__ == "__main__":
    main()
