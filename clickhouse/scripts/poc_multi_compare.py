"""
poc_multi_compare.py

Prueba pequeña: ¿una semilla partida en ASPECTOS recupera mejores tópicos de
OpenAlex que la semilla completa en un solo vector?

Compara tres métodos sobre los 4.516 tópicos:
  single   La semilla completa, un solo embedding (lo que hace la herramienta hoy).
  max      Cada aspecto es una consulta; cada tópico toma su MEJOR similitud entre
           aspectos (la semilla completa no entra).
  rrf      Reciprocal Rank Fusion (k=60) de la semilla completa + todos los aspectos.
           Fusiona posiciones, no puntajes, así que no importa que el coseno de una
           consulta corta y el de una larga no sean comparables.

Dos pasos:

  1) run: calcula los tres rankings y arma un "pool" mezclado y SIN etiquetas de
     método para que un juez (un agente de IA, o un experto) califique cada tópico
     sin saber de qué método viene.

     python poc_multi_compare.py run --seed-file ../seed_bioeconomia.txt \\
         --aspects-file data/aspects_bioeconomia_en.txt --top-n 65 --out-dir multi_out

     Genera en --out-dir:
       comparison.csv   todos los tópicos del pool con su posición en cada método
       judge_pool.csv   pool barajado, sin método: judge_id, name, keywords, summary
       judge_key.csv    clave judge_id -> topic_id (NO se le entrega al juez)
       overlap.txt      solapamiento entre métodos

  2) score: con las calificaciones de UNO O MÁS jueces (un CSV por juez, columnas
     judge_id,label con label 0 irrelevante, 1 relacionado, 2 central) calcula, por
     juez, P@N, nDCG@N y los tópicos relevantes que solo encontró cada método. Con
     dos o más jueces agrega el acuerdo entre ellos (% de coincidencia, kappa y kappa
     ponderado) y escribe disagreements.csv con los tópicos en que difieren, para que
     el experto desempate. Con tres o más jueces calcula además un consenso (mediana).

     python poc_multi_compare.py score --out-dir multi_out \\
         --judgments multi_out/opus.csv multi_out/sonnet.csv multi_out/chatgpt.csv \\
         --names Opus Sonnet ChatGPT

Advertencia: el juez no es la verdad. Un agente de IA califica con sus propios
sesgos; el experto temático es quien decide. Esto mide cuál método le entrega al
juez más tópicos útiles, no cuál es "correcto".

Variables de entorno: igual que poc_candidates.py (../.env).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from embedding_profile import QWEN_MODEL, QWEN_URL
from poc_candidates import get_seed_embedding  # same calculation (Qwen query format)

RRF_K = 60
METHODS = ["single", "max", "rrf"]


# --------------------------------------------------------------------------
# Fusión (funciones puras, probables sin base de datos)
# --------------------------------------------------------------------------

def to_ranking(scores: dict[str, float]) -> list[str]:
    """Ids ordenados de mayor a menor similitud."""
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def fuse_max(aspect_scores: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, int]]:
    """Mejor similitud por tópico entre aspectos. Devuelve (puntaje, índice de aspecto 1-based)."""
    best: dict[str, float] = {}
    which: dict[str, int] = {}
    for i, scores in enumerate(aspect_scores, start=1):
        for tid, s in scores.items():
            if tid not in best or s > best[tid]:
                best[tid] = s
                which[tid] = i
    return best, which


def fuse_rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    out: dict[str, float] = {}
    for ranking in rankings:
        for pos, tid in enumerate(ranking, start=1):
            out[tid] = out.get(tid, 0.0) + 1.0 / (k + pos)
    return out


def top_ids(scores: dict[str, float], n: int) -> list[str]:
    return to_ranking(scores)[:n]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def ndcg_at(ranked: list[str], labels: dict[str, int], n: int, ideal_pool: list[int]) -> float:
    dcg = sum(labels.get(t, 0) / math.log2(i + 1) for i, t in enumerate(ranked[:n], start=1))
    ideal = sorted(ideal_pool, reverse=True)[:n]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------
# Acceso a ClickHouse
# --------------------------------------------------------------------------

def fetch_all_scores(client: Any, embedding: list[float]) -> dict[str, float]:
    rows = client.query(
        """
        SELECT topic_id, 1 - cosineDistance(embedding, {e:Array(Float32)}) AS s
        FROM theme_curation.topics
        """,
        parameters={"e": embedding},
    ).result_rows
    return {tid: float(s) for tid, s in rows}


def fetch_topic_info(client: Any, ids: list[str]) -> dict[str, dict[str, str]]:
    rows = client.query(
        """
        SELECT topic_id, name, subfield_name, field_name, domain_name, keywords, summary
        FROM theme_curation.topics WHERE has({ids:Array(String)}, topic_id)
        """,
        parameters={"ids": ids},
    ).result_rows
    return {r[0]: {"name": r[1], "subfield": r[2], "field": r[3], "domain": r[4],
                   "keywords": r[5], "summary": r[6]} for r in rows}


def read_aspects(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


# --------------------------------------------------------------------------
# Paso 1: run
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, client: Any, ollama_url: str, model: str) -> None:
    seed = args.seed_file.read_text(encoding="utf-8").strip()
    aspects = read_aspects(args.aspects_file)
    if len(seed.split()) < 20:
        sys.exit("ERROR: la semilla completa está vacía o es muy corta.")
    if len(aspects) < 2:
        sys.exit("ERROR: el archivo de aspectos necesita al menos 2 líneas.")
    n = args.top_n
    print(f"Semilla: {len(seed.split())} palabras · {len(aspects)} aspectos · top-{n}")

    single = fetch_all_scores(client, get_seed_embedding(ollama_url, model, seed))
    aspect_scores = [fetch_all_scores(client, get_seed_embedding(ollama_url, model, a)) for a in aspects]

    max_scores, which = fuse_max(aspect_scores)
    rrf_scores = fuse_rrf([to_ranking(single)] + [to_ranking(s) for s in aspect_scores])

    tops = {"single": top_ids(single, n), "max": top_ids(max_scores, n), "rrf": top_ids(rrf_scores, n)}
    pool_ids = list(dict.fromkeys(tops["single"] + tops["max"] + tops["rrf"]))
    pos = {m: {t: i for i, t in enumerate(ids, start=1)} for m, ids in tops.items()}
    info = fetch_topic_info(client, pool_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "name", "subfield", "field", "rank_single", "rank_max", "rank_rrf",
                    "sim_single", "best_aspect", "best_aspect_sim"])
        for t in sorted(pool_ids, key=lambda t: min(pos[m].get(t, 10**6) for m in METHODS)):
            a = which.get(t)
            w.writerow([t, info[t]["name"], info[t]["subfield"], info[t]["field"],
                        pos["single"].get(t, ""), pos["max"].get(t, ""), pos["rrf"].get(t, ""),
                        f"{single[t]:.4f}", aspects[a - 1] if a else "", f"{max_scores[t]:.4f}"])

    rnd = random.Random(42)
    shuffled = pool_ids[:]
    rnd.shuffle(shuffled)
    key_rows = []
    with open(args.out_dir / "judge_pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["judge_id", "name", "keywords", "summary"])
        for i, t in enumerate(shuffled, start=1):
            jid = f"J{i:03d}"
            key_rows.append((jid, t))
            w.writerow([jid, info[t]["name"], info[t]["keywords"], info[t]["summary"][:500]])
    with open(args.out_dir / "judge_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["judge_id", "topic_id"])
        w.writerows(key_rows)
    # Rankings completos por método, para calcular métricas en el paso score.
    with open(args.out_dir / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rank", "topic_id"])
        for m in METHODS:
            for i, t in enumerate(tops[m], start=1):
                w.writerow([m, i, t])

    lines = [f"Pool: {len(pool_ids)} tópicos distintos (unión de los tres top-{n})."]
    for a, b in (("single", "max"), ("single", "rrf"), ("max", "rrf")):
        both = len(set(tops[a]) & set(tops[b]))
        lines.append(f"{a} vs {b}: {both} en común, Jaccard {jaccard(tops[a], tops[b]):.2f}")
    for m in METHODS:
        others = set().union(*(set(tops[o]) for o in METHODS if o != m))
        lines.append(f"Solo en {m}: {len(set(tops[m]) - others)} tópicos")
    per_aspect: dict[int, int] = {}
    for t in tops["max"]:
        per_aspect[which[t]] = per_aspect.get(which[t], 0) + 1
    lines.append("Tópicos del top-max por aspecto: " +
                 ", ".join(f"A{i}={per_aspect.get(i, 0)}" for i in range(1, len(aspects) + 1)))
    text = "\n".join(lines)
    (args.out_dir / "overlap.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nArchivos en {args.out_dir}/. Entrega SOLO judge_pool.csv al juez (no judge_key.csv).")


# --------------------------------------------------------------------------
# Paso 2: score
# --------------------------------------------------------------------------

def read_judgments(path: Path, key: dict[str, str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        jid = r["judge_id"].strip()
        if jid not in key:
            print(f"AVISO ({path.name}): judge_id desconocido {jid!r}, se ignora.", file=sys.stderr)
            continue
        label = int(r["label"])
        if label not in (0, 1, 2):
            sys.exit(f"ERROR ({path.name}): label {label} en {jid}; solo se permite 0, 1 o 2.")
        labels[key[jid]] = label
    missing = set(key.values()) - set(labels)
    if missing:
        sys.exit(f"ERROR ({path.name}): faltan calificaciones para {len(missing)} tópicos del pool.")
    return labels


def kappa(a: list[int], b: list[int], weighted: bool) -> float:
    """Kappa de Cohen entre dos jueces (ordinal 0/1/2). weighted=True: pesos cuadráticos."""
    k = 3
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2 / (k - 1) ** 2) if weighted else (0.0 if i == j else 1.0)
            num += w * obs[i][j]
            den += w * row[i] * col[j] / n
    return 1 - num / den if den else float("nan")


def print_method_table(labels: dict[str, int], tops: dict[str, list[str]]) -> None:
    n = max(len(v) for v in tops.values())
    ideal_pool = list(labels.values())
    print(f"Calificados: {len(labels)} tópicos · relevantes (>=1): {sum(v >= 1 for v in labels.values())} "
          f"· centrales (2): {sum(v == 2 for v in labels.values())}\n")
    print(f"{'método':<8} {'P@'+str(n)+' (>=1)':>12} {'P@'+str(n)+' (=2)':>12} {'nDCG@'+str(n):>10} "
          f"{'relev.':>7} {'centr.':>7} {'solo él':>8}")
    for m in METHODS:
        ids = tops[m]
        rel = [t for t in ids if labels[t] >= 1]
        cen = [t for t in ids if labels[t] == 2]
        others = set().union(*(set(tops[o]) for o in METHODS if o != m))
        unique_rel = [t for t in rel if t not in others]
        print(f"{m:<8} {len(rel) / len(ids):>12.2f} {len(cen) / len(ids):>12.2f} "
              f"{ndcg_at(ids, labels, n, ideal_pool):>10.3f} {len(rel):>7} {len(cen):>7} {len(unique_rel):>8}")


def cmd_score(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.judgments]
    for needed, what in ((args.out_dir / "judge_key.csv", "el paso 'run' (genera la clave)"),
                         (args.out_dir / "rankings.csv", "el paso 'run' (genera los rankings)"),
                         *((p, "las calificaciones del juez: un CSV con columnas judge_id,label") for p in paths)):
        if not Path(needed).is_file():
            sys.exit(f"ERROR: falta {needed}. Falta hacer: {what}.")
    names = args.names or [p.stem for p in paths]
    if len(names) != len(paths):
        sys.exit("ERROR: --names debe tener tantos nombres como archivos de --judgments.")

    key = {r["judge_id"]: r["topic_id"] for r in csv.DictReader(open(args.out_dir / "judge_key.csv", encoding="utf-8"))}
    judges = {nm: read_judgments(p, key) for nm, p in zip(names, paths)}
    tops: dict[str, list[str]] = {m: [] for m in METHODS}
    for r in csv.DictReader(open(args.out_dir / "rankings.csv", encoding="utf-8")):
        tops[r["method"]].append(r["topic_id"])

    for nm, labels in judges.items():
        print(f"=== Juez: {nm} ===")
        print_method_table(labels, tops)
        print()

    if len(judges) >= 2:
        order = list(key.values())
        print("=== Acuerdo entre jueces ===")
        print(f"{'par':<28} {'coinciden':>10} {'kappa':>7} {'kappa pond.':>12} {'0 vs 2':>7}")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = [judges[names[i]][t] for t in order]
                b = [judges[names[j]][t] for t in order]
                exact = sum(x == y for x, y in zip(a, b)) / len(order)
                big = sum(abs(x - y) == 2 for x, y in zip(a, b))
                print(f"{names[i] + ' / ' + names[j]:<28} {exact:>10.0%} {kappa(a, b, False):>7.2f} "
                      f"{kappa(a, b, True):>12.2f} {big:>7}")
        print("(kappa: <0,4 débil · 0,4-0,6 moderado · 0,6-0,8 bueno · >0,8 muy bueno)\n")

        info_names = {r["topic_id"]: r for r in csv.DictReader(open(args.out_dir / "comparison.csv", encoding="utf-8"))}
        with open(args.out_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["topic_id", "name"] + names + ["spread"])
            rows = []
            for t in order:
                vals = [judges[nm][t] for nm in names]
                if max(vals) - min(vals) >= 1:
                    rows.append((max(vals) - min(vals), t, vals))
            for spread, t, vals in sorted(rows, key=lambda r: -r[0]):
                w.writerow([t, info_names.get(t, {}).get("name", "")] + vals + [spread])
        print(f"Desacuerdos para que el experto desempate: {args.out_dir}/disagreements.csv "
              f"({len(rows)} tópicos con alguna diferencia)\n")

    if len(judges) >= 3:
        consensus = {t: sorted(judges[nm][t] for nm in names)[len(names) // 2] for t in key.values()}
        print("=== Consenso (mediana de los jueces) ===")
        print_method_table(consensus, tops)
        print()
    print("Nota: la referencia es el propio pool calificado (solo lo que algún método trajo). "
          "No mide tópicos relevantes que ningún método encontró.")


def main() -> None:
    from dotenv import load_dotenv
    import clickhouse_connect

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--seed-file", required=True, type=Path)
    pr.add_argument("--aspects-file", required=True, type=Path)
    pr.add_argument("--top-n", type=int, default=65)
    pr.add_argument("--out-dir", type=Path, default=Path("multi_out"))
    pr.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    ps = sub.add_parser("score")
    ps.add_argument("--out-dir", type=Path, default=Path("multi_out"))
    ps.add_argument("--judgments", required=True, nargs="+",
                    help="Uno o más CSV judge_id,label (uno por juez).")
    ps.add_argument("--names", nargs="+", default=None,
                    help="Nombres de los jueces, en el mismo orden (por defecto, el nombre del archivo).")
    args = parser.parse_args()

    if args.cmd == "score":
        cmd_score(args)
        return
    load_dotenv(args.env_file)
    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "colav"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DB", "theme_curation"),
    )
    cmd_run(args, client, os.getenv("OLLAMA_URL", QWEN_URL),
            os.getenv("OLLAMA_MODEL", QWEN_MODEL))


if __name__ == "__main__":
    main()
