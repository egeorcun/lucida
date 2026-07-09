"""DIS-VD manifest satırlarını dosya adı token'larından yeniden etiketle (thin/complex).

Kullanım:
    uv run python scripts/relabel_disvd.py

Sebep: sample_disvd_multi() ilk halde DIS-VD havuzunu thin/complex/general'e RASTGELE
dağıtıyordu. Gerçek DIS5K sınıfı id'nin içine kodlu (ör.
disvd_thin_20_Sports_8_Racket_4827171149_3140bffe12_o -> sınıf 'Racket'). Bu script
YALNIZCA 'category' alanını, classify_disvd() ile yeniden hesaplayarak düzeltir;
id/dosya adları DEĞİŞMEZ (önbelleklenmiş model çıktıları geçerliliğini korur).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_testset import classify_disvd, parse_disvd_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/testset/manifest.jsonl"


def main() -> None:
    lines = MANIFEST.read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    changed = 0
    for row in rows:
        if not row["id"].startswith("disvd_"):
            continue
        old_cat = row["category"]
        cls = parse_disvd_class(row["id"])
        new_cat = classify_disvd(row["id"])
        print(f"{row['id']}: sınıf={cls!r} kategori {old_cat!r} -> {new_cat!r}")
        if new_cat != old_cat:
            changed += 1
        row["category"] = new_cat

    MANIFEST.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"\n{changed} DIS-VD satırı yeniden etiketlendi.")
    print("Nihai manifest kategori dağılımı:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
