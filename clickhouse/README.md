<img src="https://raw.githubusercontent.com/colav/colav.github.io/master/img/Logo.png"/>

# ClickHouse - Analytical Database

Base de datos analítica (OLAP) para Colav. Primer consumidor: la herramienta
de curación de Ejes Temáticos (catálogo de tópicos OpenAlex + embeddings,
ejes curados por expertos). Pensada para crecer, más adelante, como destino
también de indicadores/tableros que hoy son lentos sobre Mongo.

## 📁 Estructura

```
clickhouse/
├── docker-compose.yml
├── .env.example
├── init/                        # Se ejecuta una sola vez, al primer arranque
│   ├── 001_create_database.sql
│   ├── 002_create_topics.sql
│   ├── 003_create_themes.sql
│   └── 004_create_theme_topics.sql
├── scripts/
│   ├── load_topics.py
│   ├── requirements.txt
│   └── data/
│       └── openalex_topics.csv  # catálogo OpenAlex, se commitea (data semilla)
└── README.md
```

## Quick Start

```bash
cp .env.example .env
# editar .env: contraseña real, y CLICKHOUSE_CPUS/CLICKHOUSE_RAM según el servidor

docker compose up -d
docker compose ps          # confirmar "healthy"
```

`.env` está en `.gitignore` — nunca se commitea, a diferencia de `mongodb/.env`
en este mismo repo (que sí tiene una contraseña real versionada; deuda
técnica existente que decidimos no replicar acá).

Al primer arranque (volumen `clickhouse_data` vacío), se ejecutan en orden
alfabético los `.sql` de `init/`, creando la base `theme_curation` y sus 3
tablas: `topics`, `themes`, `theme_topics`.

## Cómo se conecta quyca

quyca corre con `network_mode: host` (igual que ya lo hace para Mongo y
Elasticsearch) y le habla a `localhost:$CLICKHOUSE_PORT`. Por eso el puerto
del contenedor se publica atado a `127.0.0.1` — alcanzable solo desde
procesos de esta misma máquina, no desde la red ni desde internet. No hay
red interna de Docker que configurar aparte; esa línea de `ports:` ya
resuelve el aislamiento.

## Verificar que las tablas quedaron creadas

```bash
docker compose exec clickhouse clickhouse-client \
  --user "$(grep CLICKHOUSE_USER .env | cut -d= -f2)" \
  --password "$(grep CLICKHOUSE_PASSWORD .env | cut -d= -f2)" \
  --query "SHOW TABLES FROM theme_curation"
```

## Cargar el catálogo de tópicos (una sola vez)

```bash
cd scripts
pip install -r requirements.txt

# probar primero con pocas filas
python load_topics.py --csv-path data/openalex_topics.csv --limit 20

# si todo bien, la carga completa (4.516 tópicos)
python load_topics.py --csv-path data/openalex_topics.csv
```

Requiere que `clickhouse` y `ollama` ya estén arriba (`docker compose up -d`
en cada carpeta), y que el modelo esté descargado en Ollama. Desde D-061 el
catálogo `topics` usa **Qwen3-Embedding-4B** (`qwen3-embedding:4b`, 2.560
dimensiones) en un Ollama aparte, `ollama-qwen` (puerto 11436, compose en
`ollama-qwen/`): `docker exec ollama-qwen ollama pull qwen3-embedding:4b`.
Los scripts de tópicos usan ese modelo y esa URL por defecto
(`scripts/embedding_profile.py`; `OLLAMA_URL` y `OLLAMA_MODEL` del `.env` los
sobrescriben). `hs_products` sigue en Nomic (768 dimensiones, Ollama de
`../ollama/`, puerto 11434): los scripts del HS fijan Nomic.

El script lee `../.env` automáticamente para las credenciales de ClickHouse.
Si la tabla `topics` ya tiene datos, se detiene — pasa `--force` si de
verdad quieres truncarla y recargar desde cero.

## Notas importantes

- **`topics` es una carga única, no un pipeline recurrente.** El catálogo de
  OpenAlex (4.516 tópicos + su embedding) no cambia en la operación normal
  del sistema. La carga (leer el CSV, calcular embeddings vía Ollama,
  insertar) es un script que se corre una vez a mano — **no va en Airflow**.
  Ese script es el siguiente paso, todavía no está en esta carpeta.
- **`theme_topics` usa `ReplacingMergeTree`, no hace UPDATE real.** Ver el
  comentario en `init/004_create_theme_topics.sql` — toda lectura desde
  quyca debe usar `FINAL`.
- **Sin backups automatizados por ahora**, igual que Mongo en este mismo
  repo hoy. Deuda técnica conocida, no resuelta en esta entrega.
- **Reusa el Ollama que ya existe en `../ollama/`** para los embeddings —
  no levanta uno propio, salvo `ollama-qwen/` para Qwen. Modelos: `qwen3-embedding:4b`
  (tópicos, en `ollama-qwen`) y `nomic-embed-text-v2-moe` (HS, en `../ollama/`).
