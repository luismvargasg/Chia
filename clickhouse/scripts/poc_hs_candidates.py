"""
poc_hs_candidates.py

Prueba de concepto del vínculo con el Atlas of Economic Complexity:
semilla -> productos HS más parecidos (coseno) -> indicadores de Colombia.

Cálculo análogo a poc_candidates.py pero con Nomic (semilla con prefijo
"search_query:", coseno contra embeddings "search_document:"), sobre hs_products, y luego
se cruzan los productos con trade_country_year / trade_world_year.

Indicadores por producto (USD corrientes, ver advertencias abajo):
  - exp_last        exportaciones de Colombia en el último año disponible
  - share_exports   % de las exportaciones totales de Colombia en ese nivel
  - cagr_5y         crecimiento anual compuesto 5 años (ruidoso: son USD
                    corrientes, sin deflactar, y los precios de commodities
                    mueven mucho la cifra)
  - rca_3y          RCA promedio de los últimos 3 años (>1 = ventaja comparativa)
  - pci_3y          complejidad del producto, promedio 3 años (varía año a año)
  - distance, cog   último año (solo en el archivo de 4 dígitos)
  - vs_leader       exportaciones de Colombia / las del mayor exportador mundial
  - world_last      exportaciones mundiales del producto en el último año

Advertencias que el resultado debe llevar siempre:
  * El Atlas mide solo exportaciones de bienes. No es empleo, empresas ni
    patentes; un eje con poco comercio (servicios, ciencias sociales) saldrá pobre.
  * HS92 no tiene productos emergentes (baterías de sodio, drones...) como
    categoría propia; aparecen dentro de partidas más amplias.
  * El coseno es un ranking, no un porcentaje.

Uso:
    python poc_hs_candidates.py --seed-file ../seed.txt --level 4 --top-n 25
    python poc_hs_candidates.py --seed-file ../seed.txt --csv-out hs_bioeconomia.csv

Variables de entorno: igual que load_hs_products.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import clickhouse_connect
import requests
from dotenv import load_dotenv

from embedding_profile import NOMIC_MODEL, NOMIC_URL

COLOMBIA_ID = 170
CLASSIFICATION = "HS1992"
OUT_COLUMNS = [
    "rank", "hs_code", "name", "chapter_name", "similarity", "last_year", "exp_last",
    "share_exports_pct", "cagr_5y_pct", "rca_3y", "pci_3y", "distance", "cog",
    "world_last", "vs_leader_pct", "top_exporter",
]


def get_seed_embedding(ollama_url: str, model: str, seed_text: str) -> list[float]:
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": [f"search_query: {seed_text}"]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def get_top_products(client: Any, seed_embedding: list[float], level: int, top_n: int) -> list[tuple]:
    return client.query(
        """
        SELECT hs_code, name, chapter_name, 1 - cosineDistance(embedding, {seed_embedding:Array(Float32)}) AS similarity
        FROM theme_curation.hs_products
        WHERE classification = {classification:String} AND level = {level:UInt8}
        ORDER BY similarity DESC
        LIMIT {top_n:UInt32}
        """,
        parameters={"seed_embedding": seed_embedding, "classification": CLASSIFICATION,
                    "level": level, "top_n": top_n},
    ).result_rows


def combine_max(rankings: list[list[tuple]]) -> tuple[list[tuple], dict[str, int]]:
    """Une varios rankings completos (uno por línea de la semilla): para cada producto
    se queda con su MEJOR similitud. Devuelve (lista ordenada, {código: nº de línea}).
    Evita que una lista heterogénea de bienes se promedie en un solo vector."""
    best: dict[str, tuple] = {}
    which: dict[str, int] = {}
    for i, ranking in enumerate(rankings, start=1):
        for code, name, chapter, sim in ranking:
            if code not in best or float(sim) > float(best[code][3]):
                best[code] = (code, name, chapter, float(sim))
                which[code] = i
    return sorted(best.values(), key=lambda r: r[3], reverse=True), which


def rank_of_codes(all_products: list[tuple], codes: list[str]) -> list[tuple[str, int | None, float | None, str]]:
    """Posición (1 = más parecido) y similitud de cada código en el ranking completo."""
    pos = {row[0]: (i, float(row[3]), row[1]) for i, row in enumerate(all_products, start=1)}
    return [(c, *(pos[c][:2] + (pos[c][2],))) if c in pos else (c, None, None, "(no está en el catálogo)")
            for c in codes]


def get_country_rows(client: Any, country_id: int, level: int, codes: list[str]) -> dict[str, dict[int, dict]]:
    rows = client.query(
        """
        SELECT hs_code, year, export_value, export_rca, pci, distance, cog
        FROM theme_curation.trade_country_year FINAL
        WHERE country_id = {cid:UInt32} AND level = {level:UInt8} AND has({codes:Array(String)}, hs_code)
        """,
        parameters={"cid": country_id, "level": level, "codes": codes},
    ).result_rows
    out: dict[str, dict[int, dict]] = {}
    for code, year, exp, rca, pci, dist, cog in rows:
        out.setdefault(code, {})[year] = {"exp": exp, "rca": rca, "pci": pci, "distance": dist, "cog": cog}
    return out


def get_country_totals(client: Any, country_id: int, level: int) -> dict[int, int]:
    rows = client.query(
        """
        SELECT year, sum(export_value)
        FROM theme_curation.trade_country_year FINAL
        WHERE country_id = {cid:UInt32} AND level = {level:UInt8}
        GROUP BY year
        """,
        parameters={"cid": country_id, "level": level},
    ).result_rows
    return {y: int(v) for y, v in rows}


def get_world_rows(client: Any, level: int, codes: list[str]) -> dict[str, dict[int, dict]]:
    rows = client.query(
        """
        SELECT hs_code, year, world_export_value, top_exporter_iso3, top_export_value
        FROM theme_curation.trade_world_year FINAL
        WHERE level = {level:UInt8} AND has({codes:Array(String)}, hs_code)
        """,
        parameters={"level": level, "codes": codes},
    ).result_rows
    out: dict[str, dict[int, dict]] = {}
    for code, year, world, top_iso, top_val in rows:
        out.setdefault(code, {})[year] = {"world": world, "top_iso": top_iso, "top_val": top_val}
    return out


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute_indicators(
    products: list[tuple], country: dict[str, dict[int, dict]], totals: dict[int, int],
    world: dict[str, dict[int, dict]],
) -> list[dict[str, Any]]:
    """Une candidatos con indicadores. Devuelve una fila por producto, en el orden de entrada."""
    if not totals:
        raise ValueError("No hay totales de Colombia; ¿se cargó trade_country_year?")
    last_year = max(totals)
    out: list[dict[str, Any]] = []
    for rank, (code, name, chapter, sim) in enumerate(products, start=1):
        years = country.get(code, {})
        last = years.get(last_year)
        exp_last = last["exp"] if last else 0
        first = years.get(last_year - 5)
        cagr = None
        if last and first and first["exp"] > 0 and last["exp"] > 0:
            cagr = ((last["exp"] / first["exp"]) ** (1 / 5) - 1) * 100
        recent = [years[y] for y in (last_year, last_year - 1, last_year - 2) if y in years]
        w = world.get(code, {}).get(last_year)
        vs_leader = (exp_last / w["top_val"] * 100) if w and w["top_val"] else None
        out.append({
            "rank": rank, "hs_code": code, "name": name, "chapter_name": chapter,
            "similarity": round(float(sim), 4), "last_year": last_year, "exp_last": exp_last,
            "share_exports_pct": round(exp_last / totals[last_year] * 100, 3) if totals.get(last_year) else None,
            "cagr_5y_pct": round(cagr, 1) if cagr is not None else None,
            "rca_3y": _round(_mean([r["rca"] for r in recent]), 2),
            "pci_3y": _round(_mean([r["pci"] for r in recent]), 2),
            "distance": _round(last["distance"], 3) if last and last["distance"] is not None else None,
            "cog": _round(last["cog"], 3) if last and last["cog"] is not None else None,
            "world_last": w["world"] if w else None,
            "vs_leader_pct": round(vs_leader, 2) if vs_leader is not None else None,
            "top_exporter": w["top_iso"] if w else None,
        })
    return out


def _round(v: float | None, n: int) -> float | None:
    return round(float(v), n) if v is not None else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen del conjunto: tamaño del eje en comercio, y cuántos productos
    sí exporta Colombia con ventaja comparativa (RCA > 1)."""
    exporting = [r for r in rows if r["exp_last"] > 0]
    with_rca = [r for r in rows if r["rca_3y"] is not None and r["rca_3y"] > 1]
    return {
        "n_products": len(rows),
        "n_exporting": len(exporting),
        "n_rca_gt_1": len(with_rca),
        "exp_total": sum(r["exp_last"] for r in rows),
        "share_total_pct": round(sum(r["share_exports_pct"] or 0 for r in rows), 2),
    }


def fmt_usd(v: int | float | None) -> str:
    if v is None:
        return "-"
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.1f}{suf}"
    return f"{v:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-file", required=True, type=Path)
    parser.add_argument("--level", type=int, default=4, choices=[4, 6])
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--country-id", type=int, default=COLOMBIA_ID)
    parser.add_argument("--multi", action="store_true",
                         help="Cada línea del archivo es una consulta aparte (una familia de bienes); "
                              "cada producto toma su mejor similitud entre las líneas.")
    parser.add_argument("--check-codes", nargs="+", default=[],
                         help="Códigos HS (ej. 0901 1511 0603) cuya posición en el ranking completo se quiere ver.")
    parser.add_argument("--csv-out", type=Path, default=Path("hs_candidatos.csv"))
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-db", default=None)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    ollama_url = args.ollama_url or NOMIC_URL
    ollama_model = args.ollama_model or NOMIC_MODEL  # hs_products stays on Nomic (768 dims); OLLAMA_MODEL is ignored
    client = clickhouse_connect.get_client(
        host=args.clickhouse_host or os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(args.clickhouse_port or os.getenv("CLICKHOUSE_PORT", "8123")),
        username=args.clickhouse_user or os.getenv("CLICKHOUSE_USER", "colav"),
        password=args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=args.clickhouse_db or os.getenv("CLICKHOUSE_DB", "theme_curation"),
    )

    if not args.seed_file.is_file():
        print(f"ERROR: no existe {args.seed_file}", file=sys.stderr)
        sys.exit(1)
    seed = args.seed_file.read_text(encoding="utf-8").strip()
    which: dict[str, int] = {}
    lines: list[str] = []
    if args.multi:
        lines = [ln.strip() for ln in seed.splitlines() if ln.strip()]
        if len(lines) < 2:
            print("ERROR: --multi necesita al menos 2 líneas (una familia de bienes por línea).", file=sys.stderr)
            sys.exit(1)
        print(f"Semilla multi-consulta: {len(lines)} líneas.")
        rankings = [get_top_products(client, get_seed_embedding(ollama_url, ollama_model, ln),
                                     args.level, 100000) for ln in lines]
        every, which = combine_max(rankings)
        products = every[: args.top_n]
    else:
        n_words = len(seed.split())
        if n_words < 20:
            print(f"ERROR: la semilla de {args.seed_file} tiene {n_words} palabras; está vacía o es "
                  f"demasiado corta (mínimo 20, recomendado 80-150).", file=sys.stderr)
            sys.exit(1)
        print(f"Semilla: {n_words} palabras (límite ~370).")
        emb = get_seed_embedding(ollama_url, ollama_model, seed)
        products = get_top_products(client, emb, args.level, args.top_n)
        every = get_top_products(client, emb, args.level, 100000) if args.check_codes else []
    if not products:
        print(f"No hay productos de nivel {args.level} en hs_products. ¿Corriste load_hs_products.py?",
              file=sys.stderr)
        sys.exit(1)
    if args.check_codes:
        print(f"\nPosición en el ranking completo ({len(every)} productos):")
        for code, rank, sim, name in rank_of_codes(every, args.check_codes):
            print(f"  {code}: " + (f"puesto {rank}, sim {sim:.3f}  {name[:55]}" if rank else name))
    codes = [p[0] for p in products]
    country = get_country_rows(client, args.country_id, args.level, codes)
    totals = get_country_totals(client, args.country_id, args.level)
    world = get_world_rows(client, args.level, codes)
    rows = compute_indicators(products, country, totals, world)

    print(f"\n{'#':>3} {'HS':>6} {'sim':>6} {'exp '+str(rows[0]['last_year']):>9} {'%exp':>6} "
          f"{'CAGR5':>7} {'RCA':>5} {'PCI':>5}  producto")
    for r in rows:
        print(f"{r['rank']:>3} {r['hs_code']:>6} {r['similarity']:>6.3f} {fmt_usd(r['exp_last']):>9} "
              f"{(r['share_exports_pct'] if r['share_exports_pct'] is not None else 0):>6.2f} "
              f"{(str(r['cagr_5y_pct']) if r['cagr_5y_pct'] is not None else '-'):>7} "
              f"{(str(r['rca_3y']) if r['rca_3y'] is not None else '-'):>5} "
              f"{(str(r['pci_3y']) if r['pci_3y'] is not None else '-'):>5}  "
              f"{('L'+str(which[r['hs_code']])+' ') if which else ''}{r['name'][:58]}")
    s = summarize(rows)
    print(f"\nResumen: {s['n_exporting']}/{s['n_products']} productos con exportaciones de Colombia; "
          f"{s['n_rca_gt_1']} con RCA>1; suman {fmt_usd(s['exp_total'])} USD "
          f"({s['share_total_pct']}% de las exportaciones de bienes).")
    if s["exp_total"] > 0:
        top = max(rows, key=lambda r: r["exp_last"])
        pct = top["exp_last"] / s["exp_total"] * 100
        if pct >= 40:
            print(f"AVISO: {top['hs_code']} ({top['name'][:35]}) aporta {pct:.0f}% de esa suma. "
                  f"Revisa a mano antes de usar el total.")
    print("Ojo: USD corrientes; solo bienes exportados; HS92 sin productos emergentes; coseno = ranking, no %.")

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
