# Paralel Kopya Hücresi — Drive → Yerel Disk (16 iş parçacığı)

Yavaş giden kopya hücresini **durdur** (■), sonra onun ÜSTÜNE yeni bir kod
hücresi ekleyip aşağıdaki bloğu yapıştır ve çalıştır. Bittiğinde, durdurduğun
orijinal kopya hücresini yeniden çalıştır (var olanları görüp saniyeler içinde
geçer) ve "Çalışma zamanı → Sonrasını çalıştır" ile devam et.

Not: Bu hücre, bölünme hücresinin tanımladığı değişkenleri kullanır
(train_stems, drive_train_im, local_train_im...). Bölünme hücresi bu oturumda
çalışmış olmalı (çalıştı — çıktısında TRAIN=27715 gördün).

```python
# Drive -> yerel paralel kopya: tek tek okuma yerine 16 paralel okuyucu.
# İdempotent: boyutu doğru olan var olan dosyalar atlanır.
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

IS_PARCACIGI = 16

def _kopyala_cift(stem, src_im_dir, src_gt_dir, dst_im_dir, dst_gt_dir):
    src_im, src_gt = src_im_dir / f"{stem}.jpg", src_gt_dir / f"{stem}.png"
    dst_im, dst_gt = dst_im_dir / f"{stem}.jpg", dst_gt_dir / f"{stem}.png"
    try:
        im_tam = dst_im.exists() and dst_im.stat().st_size == src_im.stat().st_size
        gt_tam = dst_gt.exists() and dst_gt.stat().st_size == src_gt.stat().st_size
        if im_tam and gt_tam:
            return "atlandi"
        if not im_tam:
            shutil.copy2(src_im, dst_im)
        if not gt_tam:
            shutil.copy2(src_gt, dst_gt)
        return "kopyalandi"
    except Exception as e:
        return f"hata:{stem}:{e}"

def paralel_kopya(stems, src_im_dir, src_gt_dir, dst_im_dir, dst_gt_dir, etiket):
    dst_im_dir.mkdir(parents=True, exist_ok=True)
    dst_gt_dir.mkdir(parents=True, exist_ok=True)
    sayac = {"kopyalandi": 0, "atlandi": 0, "hata": 0}
    hatalar = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=IS_PARCACIGI) as ex:
        futs = [ex.submit(_kopyala_cift, s, src_im_dir, src_gt_dir, dst_im_dir, dst_gt_dir) for s in stems]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.startswith("hata"):
                sayac["hata"] += 1
                hatalar.append(r)
            else:
                sayac[r] += 1
            if i % 2000 == 0:
                hiz = i / (time.time() - t0)
                kalan_dk = (len(stems) - i) / hiz / 60
                print(f"[{etiket}] {i}/{len(stems)}  ~{hiz:.0f} çift/sn  tahmini kalan: {kalan_dk:.0f} dk")
    print(f"[{etiket}] bitti: {sayac}  süre: {(time.time()-t0)/60:.1f} dk")
    if hatalar:
        print("İlk 5 hata:", hatalar[:5])
    return sayac

paralel_kopya(train_stems, drive_train_im, drive_train_gt, local_train_im, local_train_gt, "TRAIN")
paralel_kopya(val_stems, drive_train_im, drive_train_gt, local_val_im, local_val_gt, "VAL")
print("Paralel kopya tamam — şimdi orijinal kopya hücresini yeniden çalıştır ve 'Sonrasını çalıştır' de.")
```

Beklenen süre: ~20-45 dk (2000 çiftte bir ilerleme + tahmini kalan süre yazar).
"hata" sayısı 0 olmalı; değilse ilk 5 hatayı Claude'a gönder.
