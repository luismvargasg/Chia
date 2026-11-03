"""
poc_candidates.py

Prueba de concepto: toma una semilla en lenguaje natural, calcula su
embedding vía Ollama (modelo qwen3-embedding:4b con una instrucción de
consulta; ver embedding_profile.py) y devuelve el top N de
tópicos más parecidos en ClickHouse, por similitud coseno.

Esto es una versión de mano del futuro POST /api/curation/candidates de
quyca — mismo cálculo, sin backend ni UI todavía.

Uso:
    python poc_candidates.py
    python poc_candidates.py --top-n 65 --csv-out candidatos.csv

Lee las credenciales de ClickHouse desde ../.env (igual que load_topics.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import clickhouse_connect
import requests
from dotenv import load_dotenv

from embedding_profile import QWEN_MODEL, QWEN_URL, format_query, require_qwen

# --- Semilla del experto: editar acá para probar otra trayectoria ---------
SEED_TEXT = """
Transición hacia alternativas de transporte y movilidad inteligentes,
sostenibles y accesibles

Esta trayectoria busca migrar y consolidar modelos de movilidad sostenibles en la ciudad hacia
opciones que reduzcan la huella de carbono, mejoren la calidad del aire, utilicen datos para la gestión
y aseguren el acceso equitativo de los sistemas masivos de transporte.
Tendencias Globales asociadas a la trayectoria:
- Tecnológica: Edge AI y redes de baja latencia; Baterías de Iones de sodio; Sistemas corporales
y robótica autónoma
- Social: Robotaxis y movilidad autónoma; Infraestructura urbana adaptativa; Resiliencia
ambiental y competitividad industrial.
La trayectoria de Medellín hacia una movilidad inteligente y baja en carbono se cruza con tendencias
globales como la electrificación del transporte, los vehículos conectados y la gestión urbana basada
en datos. La ciudad ya exhibe avances tangibles: una red Metro consolidada, una flota creciente de
buses eléctricos, así como en la integración de vertipuertos para drones dentro del C5 y la puesta
en marcha de Zonas Urbanas de Aire Protegido que combinan regulación ambiental con
tecnología de monitoreo. Estas acciones muestran una intención clara de traducir la urgencia
climática en políticas concretas de transporte. Sin embargo, la convergencia aún es desigual.
Sectores como la logística de carga y las volquetas siguen anclados a motores diésel,
configurando el principal cuello de botella para mejorar la calidad del aire y dificultando el
cumplimiento de la meta de carbono neutralidad prevista en el Plan Decenal.
Medellín avanza en dos velocidades. Por un lado, fortalece su transporte masivo en línea con las mega
tendencias de sistemas emergentes de almacenamiento y sostenibilidad; por otro, mantiene rezagos
estructurales en la movilidad privada y el transporte de carga. Mientras la "Cultura Metro" representa
un referente de movilización colectiva eficiente, el uso intensivo de carros y motos, agravado por las
limitaciones topográficas del valle, perpetúa un modelo individualista difícil de transformar.
Sumado a lo anterior, la tendencia "Baterías de Iones de sodio", el análisis indica que la flota
logística de carga aún continúa operando con combustibles fósiles (foco de PM2.5); por ende, esta
trayectoria requiere tecnologías de almacenamiento escalables para electrificar la última milla.

En relación con la tendencia "Edge AI y redes de baja Latencia", el C5 requiere procesamiento en
el borde para optimizar el tráfico en tiempo real. A esto se añade un déficit de información: los datos
sobre siniestralidad y emisiones siguen fragmentados y poco actualizados, lo que impide una gestión
proactiva con analítica avanzada o infraestructura V2X. La ciudad, aunque bien encaminada, necesita
cerrar estas brechas si quiere capitalizar plenamente su alineación con las tendencias globales de
movilidad descarbonizada.
""".strip()
# ---------------------------------------------------------------------------

# Qwen cuts at 4096 tokens; the tool keeps a 1,800-character limit (D-061).
WORD_LIMIT_SOFT = 370


def get_seed_embedding(ollama_url: str, model: str, seed_text: str) -> list[float]:
    require_qwen(model)
    text = format_query(seed_text)
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def get_top_candidates(client, seed_embedding: list[float], top_n: int) -> list[tuple]:
    query = """
        SELECT
            topic_id,
            name,
            subfield_name,
            field_name,
            domain_name,
            keywords,
            1 - cosineDistance(embedding, {seed_embedding:Array(Float32)}) AS similarity
        FROM theme_curation.topics
        ORDER BY similarity DESC
        LIMIT {top_n:UInt32}
    """
    result = client.query(
        query,
        parameters={"seed_embedding": seed_embedding, "top_n": top_n},
    )
    return result.result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-db", default=None)
    parser.add_argument("--top-n", type=int, default=65)
    parser.add_argument("--csv-out", type=Path, default=Path("candidatos.csv"))
    parser.add_argument("--seed-file", type=Path, default=None,
                         help="Si se pasa, usa el texto de este archivo en vez del SEED_TEXT embebido arriba.")
    args = parser.parse_args()

    load_dotenv(args.env_file)

    seed_text = SEED_TEXT
    if args.seed_file:
        seed_text = args.seed_file.read_text(encoding="utf-8").strip()
        print(f"Usando semilla de: {args.seed_file}")

    ollama_url = args.ollama_url or os.getenv("OLLAMA_URL", QWEN_URL)
    ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", QWEN_MODEL)
    require_qwen(ollama_model)
    ch_host = args.clickhouse_host or os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port = int(args.clickhouse_port or os.getenv("CLICKHOUSE_PORT", "8123"))
    ch_user = args.clickhouse_user or os.getenv("CLICKHOUSE_USER", "colav")
    ch_password = args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD", "")
    ch_db = args.clickhouse_db or os.getenv("CLICKHOUSE_DB", "theme_curation")

    word_count = len(seed_text.split())
    print(f"Semilla: {word_count} palabras (límite recomendado: ~{WORD_LIMIT_SOFT}).")
    if word_count > WORD_LIMIT_SOFT:
        exceso = word_count - WORD_LIMIT_SOFT
        print(
            f"  ADVERTENCIA: te pasas por ~{exceso} palabras. Ollama probablemente va a "
            f"truncar el final del texto en silencio (no te va a avisar por la API). "
            f"Para confirmarlo, corre en otra terminal:\n"
            f"    docker compose logs -f ollama\n"
            f"  y busca 'truncated = 1' mientras este script corre.",
            file=sys.stderr,
        )

    print(f"\nCalculando embedding de la semilla vía Ollama ({ollama_model})...")
    seed_embedding = get_seed_embedding(ollama_url, ollama_model, seed_text)
    print(f"  Embedding recibido, dimensión = {len(seed_embedding)}")

    print(f"\nConectando a ClickHouse ({ch_host}:{ch_port}/{ch_db})...")
    client = clickhouse_connect.get_client(
        host=ch_host, port=ch_port, username=ch_user, password=ch_password, database=ch_db,
    )

    print(f"Buscando los {args.top_n} tópicos más similares...")
    rows = get_top_candidates(client, seed_embedding, args.top_n)

    print(f"\n{'#':>3}  {'score':>6}  {'topic_id':>8}  {'subfield':<28} name")
    print("-" * 100)
    for i, (topic_id, name, subfield_name, field_name, domain_name, keywords, similarity) in enumerate(rows, 1):
        print(f"{i:>3}  {similarity:>6.3f}  {topic_id:>8}  {subfield_name[:28]:<28} {name}")

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "similarity", "topic_id", "name", "subfield_name",
                          "field_name", "domain_name", "keywords"])
        for i, (topic_id, name, subfield_name, field_name, domain_name, keywords, similarity) in enumerate(rows, 1):
            writer.writerow([i, f"{similarity:.4f}", topic_id, name, subfield_name,
                              field_name, domain_name, keywords])

    print(f"\nGuardado también en: {args.csv_out}")


if __name__ == "__main__":
    main()
