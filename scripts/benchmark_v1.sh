#!/usr/bin/env bash
# Fine-tune checkpoint (data/checkpoints/epoch_1.pth) geldiğinde çalıştırılacak
# tam benchmark + galeri + karşılaştırma dizisi.
#
# Önkoşul: data/checkpoints/epoch_1.pth mevcut olmalı (bgr-v1 registry girdisi
# bunu bekler, bkz. bgr/registry.py MODEL_SPECS["bgr-v1"]).
#
# Kullanım:
#   bash scripts/benchmark_v1.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT="data/checkpoints/epoch_1.pth"
MANIFEST="data/testset/manifest.jsonl"
OUT="results/baseline"
ALL_MODELS="birefnet-hr,rmbg-2.0,rmbg-2.0+refine,bgr-v1,bgr-v1+refine"

if [ ! -f "$CKPT" ]; then
  echo "HATA: $CKPT bulunamadı. Checkpoint henüz gelmedi." >&2
  exit 1
fi

echo "=== 1/4: bgr-v1 + bgr-v1+refine benchmark koşusu (metrics.json baseline'ları korunarak birleşir) ==="
uv run python -m benchmark.run \
  --models bgr-v1,bgr-v1+refine \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --rgba

echo "=== 2/4: galeri yenileme (5 model + ideogram) ==="
uv run python -m benchmark.gallery \
  --manifest "$MANIFEST" \
  --results "$OUT" \
  --models "$ALL_MODELS" \
  --out "$OUT/gallery.html"

echo "=== 3/4: karşılaştırma tablosu (Markdown) ==="
uv run python scripts/compare_v1.py --metrics "$OUT/metrics.json" \
  | tee "$OUT/bgr-v1-comparison.md"

echo "=== 4/4: tamamlandı ==="
echo "metrics.json : $OUT/metrics.json"
echo "gallery.html : $OUT/gallery.html"
echo "karşılaştırma: $OUT/bgr-v1-comparison.md"
