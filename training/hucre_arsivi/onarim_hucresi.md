# Onarım Hücresi — Eksik gt Dosyalarını Atlayan Toleranslı Kopya

Eğitim notebook'unda **hata veren hücrenin ÜSTÜNE** yeni bir kod hücresi ekle,
aşağıdaki bloğun tamamını yapıştır ve çalıştır. Sonra hata veren hücreyi
yeniden çalıştır ve "Çalışma zamanı → Sonrasını çalıştır" ile devam et.

```python
# Teşhis + toleranslı kopya: eksik gt'si olan çiftleri atla (az sayıda ise zararsız)
from pathlib import Path
import shutil

d = Path("/content/drive/MyDrive/bg-remover-data/TRAIN")
im_stems = {p.stem for p in (d/"im").iterdir()}
gt_stems = {p.stem for p in (d/"gt").iterdir()}
eksik_gt = im_stems - gt_stems
eksik_im = gt_stems - im_stems
tam = im_stems & gt_stems
print(f"im: {len(im_stems)}  gt: {len(gt_stems)}  tam çift: {len(tam)}")
print(f"gt'si eksik: {len(eksik_gt)}  im'i eksik: {len(eksik_im)}")

assert len(tam) >= 27000, (
    f"Tam çift sayısı çok düşük ({len(tam)}) — atlamak yerine veri onarımı gerekir, Claude'a bu çıktıyı gönder."
)

# copy_pairs'i eksikleri atlayan versiyonla değiştir
import training.train_colab_lib as tcl

def _tolerant_copy_pairs(stems, src_im_dir, src_gt_dir, dst_im_dir, dst_gt_dir, im_ext=".jpg", gt_ext=".png"):
    copied = skipped = 0
    for stem in stems:
        src_im, src_gt = src_im_dir / f"{stem}{im_ext}", src_gt_dir / f"{stem}{gt_ext}"
        dst_im, dst_gt = dst_im_dir / f"{stem}{im_ext}", dst_gt_dir / f"{stem}{gt_ext}"
        if not (src_im.exists() and src_gt.exists()):
            skipped += 1
            continue
        if dst_im.exists() and dst_gt.exists() and dst_im.stat().st_size == src_im.stat().st_size and dst_gt.stat().st_size == src_gt.stat().st_size:
            continue
        shutil.copy2(src_im, dst_im); shutil.copy2(src_gt, dst_gt)
        copied += 1
    print(f"kopyalandı: {copied}, eksik olduğu için atlandı: {skipped}")
    return copied

tcl.copy_pairs = _tolerant_copy_pairs
print("Toleranslı kopya aktif — şimdi hata veren hücreyi yeniden çalıştır.")
```

Beklenen sonuç: "gt'si eksik" sayısı birkaç yüzse sorun yok, devam.
Binlerceyse hücre kendini durdurur — çıktıyı Claude'a gönder.
