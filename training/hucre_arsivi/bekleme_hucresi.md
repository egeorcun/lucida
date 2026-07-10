# Bekleme Hücresi — gt Yüklemesi Tamamlanana Kadar Bekle

Eğitim notebook'unda **hata veren kopya hücresinin ÜSTÜNE** yeni bir kod hücresi
ekle, aşağıdaki bloğu yapıştır. Sonra bu hücre seçiliyken
**Çalışma zamanı → Sonrasını çalıştır** de — bekleme bitince kalan hücreler
otomatik sırayla çalışır (eğitim dahil, başında durman gerekmez).

```python
# Drive'daki gt yüklemesi 28281'e ulaşana kadar 5 dk'da bir kontrol et.
# 30 dk boyunca hiç artış olmazsa durur (askıda kalmaya karşı emniyet).
import time
from pathlib import Path

HEDEF = 28281
d = Path("/content/drive/MyDrive/bg-remover-data/TRAIN/gt")

onceki, duragan_tur = -1, 0
while True:
    n = sum(1 for _ in d.iterdir())
    print(f"[{time.strftime('%H:%M:%S')}] gt: {n}/{HEDEF}")
    if n >= HEDEF:
        print("Tamamlandı — sonraki hücrelere geçiliyor.")
        break
    duragan_tur = duragan_tur + 1 if n == onceki else 0
    assert duragan_tur < 6, (
        f"30 dakikadır artış yok (gt={n}) — yükleme durmuş olabilir, Claude'a bu sayıyı gönder."
    )
    onceki = n
    time.sleep(300)  # 5 dakika
```

Not: Kopya hücresi zaten kısmen kopyalanan dosyaları atlar (boyut kontrolü var),
yani bekleme sonrası kaldığı yerden hızla tamamlar.
