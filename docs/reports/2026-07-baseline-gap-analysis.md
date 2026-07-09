# Baseline Gap Analizi (Ideogram vs açık modeller)

**Tarih:** 2026-07-09 · **Koşu:** `results/baseline/` (130 görsel, BiRefNet_HR + RMBG-2.0, MPS fp32) · **Referans:** Ideogram (fal.ai, 130 çıktı)

## Sayısal sonuçlar (GT'li kategoriler)

Genel (130 görsel ortalaması — düşük = iyi):

| Model | SAD | MAE | MSE | Grad | Conn |
|---|---|---|---|---|---|
| BiRefNet_HR | 361.7 | 0.0287 | 0.0216 | **2.461** | 366.7 |
| RMBG-2.0 | **342.5** | **0.0260** | **0.0194** | 2.754 | **350.7** |

Kategori bazında MAE:

| Kategori | BiRefNet_HR | RMBG-2.0 | Durum |
|---|---|---|---|
| hair (40) | 0.0048 | **0.0045** | İkisi de güçlü |
| thin (20) | **0.0135** | 0.0176 | BiRefNet'in 2048px avantajı |
| complex (30) | 0.0317 | **0.0221** | RMBG'nin veri avantajı |
| general (15) | 0.0400 | **0.0220** | RMBG'nin veri avantajı |
| **transparent (25)** | **0.0687** | 0.0741 | **İkisi de çok kötü — ana gap** |

Okuma: RMBG-2.0 "özneyi doğru bulma"da (alan hataları: SAD/MAE/MSE/Conn) önde; BiRefNet_HR "kenarı keskin kesme"de (Grad) önde. Fine-tune hedefimiz ikisini birleştirmek: BiRefNet tabanı + RMBG kalitesinde veri.

## Kategori bazlı gözlemler (galeri + örnek incelemesi)

### transparent — ANA GAP (10-15× daha yüksek hata)
- Ders kitabı örneği `trans460_pexels-pixabay-34487` (yusufçuk kanadı, MAE 0.312): RMBG-2.0 saydam kanat zarını **komple opak beyaz blob** olarak işaretliyor — damar/hücre yapısı ve kısmi saydamlık tamamen kayıp. Ideogram aynı görselde kanat zarını hücre düzeyinde yarı saydam alpha ile koruyor (damarlar opak, zar bölmeleri kısmi geçirgen).
- Kök neden: açık modeller binary-ish maskeye eğilimli; cam/zar gibi "pikselin kendisi kısmen arka plan" durumlarında ara alpha üretmiyorlar.
- İkinci en kötü: `trans460_pexels-tuca-bianca-360177` (MAE 0.183) — aynı desen.

### thin — orta gap, BiRefNet daha iyi
- En kötüler raket ağı (`disvd_thin_20_Sports_8_Racket`, 0.077) ve kamera kayışı/kablo örnekleri: ağ örgüsü kısmen dolduruluyor (delikler opak) veya ince tel kopuyor.

### complex / general — RMBG belirgin önde
- BiRefNet_HR karmaşık sahnelerde (dişçi koltuğu 0.078, balon halatları 0.058) parça atlama/yanlış obje dahil etme yapıyor. RMBG'nin küratörlü verisi "özne hangisi" sorusunda daha isabetli.

### hair — gap küçük
- P3M portrelerinde iki model de 0.005 MAE civarı; en kötü örnekler bile 0.01. Saç, sanılanın aksine bu iki model için artık zayıf nokta değil (ama galeri incelemesinde Ideogram'ın kenar renk temizliği hâlâ daha "temiz" duruyor — bu metrikte değil kompozitte görünür; Faz 1 decontamination tam bunu hedefliyor).

### product / illustration — GT yok, galeri-karşılaştırmalı
- Test setinde henüz kullanıcı görseli yok (`data/testset/incoming/` bekliyor). Galeri karşılaştırması eklendiğinde güncellenecek.

## Sonuç: Faz 2 veri öncelikleri

1. **Saydamlık (en yüksek öncelik):** Transparent-460'ın train split'i + cam/zar/duman içeren sentetik kompozitler; eğitimde transparent kategorisine yüksek örnekleme ağırlığı. Hedef: ara-alpha üretmeyi öğretmek (binarizasyonu kırmak).
2. **Karmaşık sahne/özne seçimi:** DIS5K-TR tamamı + çok nesneli kompozitler (BiRefNet tabanının complex/general'de RMBG'ye yetişmesi için).
3. **İnce yapı:** DIS5K zaten güçlü; raket ağı/örgü türü delikli yapılar için hedefli örnekler.
4. Saç: mevcut veriler yeterli görünüyor; P3M-10k standart karışımda kalsın (ağırlık artırmaya gerek yok).

## Sonuç: Faz 1 post-processing beklentileri

- **Decontamination (kesin kazanç):** GT metriklerine girmez ama kompozit kalitesinde Ideogram'la aradaki "temizlik" farkının ana kaynağı; tüm kategorilerde geçerli.
- **Edge Refiner:** thin ve hair kenarlarında Grad iyileşmesi beklenir; transparent'ta tek başına yetmez (oradaki sorun kenar değil, alan içi alpha).
- **Router:** thin → BiRefNet_HR, complex/general → RMBG-2.0 yönlendirmesi baseline'da bile ölçülebilir kazanç verir (kategori şampiyonları farklı). Bu, Router'ı V1'e almanın doğrulaması.

## Ham çıktılar

- Metrikler: `results/baseline/metrics.json`
- Galeri: `results/baseline/gallery.html` (orijinal | GT | BiRefNet_HR | RMBG-2.0 | Ideogram)
- Koşu logu: `results/baseline_run.log`, `results/ideogram_fetch.log`
