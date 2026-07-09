"""Manifest'teki tüm görseller için Ideogram referansı çek.
Kullanım: uv run python scripts/fetch_ideogram.py [--limit N]
"""
import argparse
from pathlib import Path

from benchmark.ideogram import fetch_reference
from benchmark.testset import load_manifest

ROOT = Path(__file__).resolve().parent.parent
MAX_PER_RUN = 250  # maliyet koruması (~$2.50 tavan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_PER_RUN)
    n = min(ap.parse_args().limit, MAX_PER_RUN)
    rows = load_manifest(str(ROOT / "data/testset/manifest.jsonl"))[:n]
    for i, row in enumerate(rows, 1):
        out = ROOT / "results/ideogram" / f"{row['id']}.png"
        fetch_reference(str(ROOT / row["image"]), str(out))
        print(f"[{i}/{len(rows)}] {row['id']}")


if __name__ == "__main__":
    main()
