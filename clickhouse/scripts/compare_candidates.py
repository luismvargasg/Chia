"""
compare_candidates.py

Compara dos CSV generados por poc_candidates.py (por ejemplo, dos semillas
distintas para la misma trayectoria) y reporta cuántos tópicos coinciden y
cuáles son exclusivos de cada uno — la misma métrica de "coincidencia" que
ya se usó al comparar developer vs. experto en la prueba real (24-26 de
51-65 tópicos coincidentes).

Uso:
    python compare_candidates.py candidatos_a.csv candidatos_b.csv
"""

import argparse
import csv
from pathlib import Path


def read_topic_ids(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["topic_id"]: row["name"] for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", type=Path)
    parser.add_argument("csv_b", type=Path)
    args = parser.parse_args()

    a = read_topic_ids(args.csv_a)
    b = read_topic_ids(args.csv_b)

    common = set(a) & set(b)
    only_a = set(a) - set(b)
    only_b = set(b) - set(a)

    print(f"{args.csv_a.name}: {len(a)} tópicos")
    print(f"{args.csv_b.name}: {len(b)} tópicos")
    print(f"\nCoinciden: {len(common)} ({len(common) / max(len(a), len(b)) * 100:.0f}%)")

    print(f"\nSolo en {args.csv_a.name} ({len(only_a)}):")
    for tid in sorted(only_a):
        print(f"  {tid}  {a[tid]}")

    print(f"\nSolo en {args.csv_b.name} ({len(only_b)}):")
    for tid in sorted(only_b):
        print(f"  {tid}  {b[tid]}")


if __name__ == "__main__":
    main()
