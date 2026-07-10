"""bgr-v1 (fine-tune) fine-tune sonuçlarını baseline modellerle karşılaştıran Markdown tablo.

`benchmark/run.py`'nin ürettiği `metrics.json`'ı okur (per_category + overall),
`bgr-v1`/`bgr-v1+refine`'i varsayılan baseline'larla (`rmbg-2.0`, `birefnet-hr`,
`rmbg-2.0+refine`) karşılaştırır. Ideogram, `bgr/registry.py` üzerinden
çalıştırılan bir segmenter olmadığı (GT karşılaştırmalı metriği hesaplanmıyor,
yalnızca galeride görsel referans) için metrics.json'da yer almaz; bu betik
ideogram'ı yalnızca metrics.json'da fiilen bulunuyorsa listeye ekler.

Kullanım:
    uv run python scripts/compare_v1.py --metrics results/baseline/metrics.json
    uv run python scripts/compare_v1.py --metrics results/baseline/metrics.json \
        --v1 bgr-v1,bgr-v1+refine --baselines rmbg-2.0,birefnet-hr,rmbg-2.0+refine
"""
import argparse
import json
from pathlib import Path

METRIC_ORDER = ["mae", "sad", "mse", "grad", "conn"]  # tümü: düşük = iyi


def _delta_cell(v1_value: float, baseline_value: float) -> str:
    """Baseline'ın kendi değeri + v1'in ona göre ok/yüzde farkı (düşük=iyi).

    Ok, v1'in bu baseline'a göre nasıl olduğunu gösterir: v1 < baseline ise v1
    daha iyi (↓ iyi), v1 > baseline ise v1 daha kötü (↑ kötü).
    """
    if baseline_value == 0:
        return f"{baseline_value:.4f}"
    delta_pct = (v1_value - baseline_value) / abs(baseline_value) * 100
    if v1_value < baseline_value:
        arrow = "↓ v1 iyi"
    elif v1_value > baseline_value:
        arrow = "↑ v1 kötü"
    else:
        arrow = "="
    return f"{baseline_value:.4f} ({arrow} {delta_pct:+.1f}%)"


def build_table(metrics: dict, v1_models: list[str], baseline_models: list[str]) -> str:
    per_category = metrics.get("per_category", {})
    overall = metrics.get("overall", {})

    # yalnız metrics.json'da gerçekten bulunan baseline'lar (örn. ideogram GT'siz koştuysa dahil edilmez)
    present_baselines = [b for b in baseline_models if b in overall]
    present_v1 = [v for v in v1_models if v in overall]
    missing_v1 = [v for v in v1_models if v not in overall]

    lines: list[str] = []
    lines.append("# bgr-v1 karşılaştırma raporu")
    lines.append("")
    if missing_v1:
        lines.append(
            f"> UYARI: metrics.json'da bulunamayan v1 modeli/modelleri: {', '.join(missing_v1)} "
            "— önce `benchmark.run --models " + ",".join(missing_v1) + "` koşusu gerekli."
        )
        lines.append("")
    if not present_v1:
        lines.append("Karşılaştırılacak v1 modeli bulunamadı, tablo üretilmedi.")
        return "\n".join(lines)

    all_categories = sorted({c for m in present_v1 + present_baselines for c in per_category.get(m, {})})

    for v1_name in present_v1:
        lines.append(f"## {v1_name} vs baseline'lar")
        lines.append("")
        header = ["kategori", "metrik", v1_name] + present_baselines
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for cat in all_categories:
            cat_metrics = per_category.get(v1_name, {}).get(cat)
            if cat_metrics is None:
                continue
            for metric in METRIC_ORDER:
                if metric not in cat_metrics:
                    continue
                v1_value = cat_metrics[metric]
                row = [cat, metric, f"{v1_value:.4f}"]
                for b in present_baselines:
                    b_value = per_category.get(b, {}).get(cat, {}).get(metric)
                    row.append(_delta_cell(v1_value, b_value) if b_value is not None else "n/a")
                lines.append("| " + " | ".join(row) + " |")

        # genel (overall)
        lines.append("")
        lines.append(f"**Genel (overall) — {v1_name}**")
        lines.append("")
        header = ["metrik", v1_name] + present_baselines
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for metric in METRIC_ORDER:
            if metric not in overall.get(v1_name, {}):
                continue
            v1_value = overall[v1_name][metric]
            row = [metric, f"{v1_value:.4f}"]
            for b in present_baselines:
                b_value = overall.get(b, {}).get(metric)
                row.append(_delta_cell(v1_value, b_value) if b_value is not None else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="results/baseline/metrics.json")
    ap.add_argument("--v1", default="bgr-v1,bgr-v1+refine")
    ap.add_argument("--baselines", default="rmbg-2.0,birefnet-hr,rmbg-2.0+refine,ideogram")
    a = ap.parse_args()

    metrics = json.loads(Path(a.metrics).read_text())
    table = build_table(
        metrics,
        v1_models=a.v1.split(","),
        baseline_models=a.baselines.split(","),
    )
    print(table)


if __name__ == "__main__":
    main()
