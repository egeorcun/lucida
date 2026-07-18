# Docker ile Lucida

Lucida'yı tek komutla ayağa kaldırmak için Docker imajı ve gömülü web arayüzü.

## Hızlı başlangıç

```bash
# İmajı derle (repo kökünde)
docker build -t lucida .

# Çalıştır — HF cache'i volume'a bağla ki model bir kez insin
docker run -p 8000:8000 -v hf-cache:/root/.cache/huggingface lucida
```

Tarayıcıda **http://localhost:8000** adresini aç:

1. Bir görseli sürükle-bırak (veya tıklayıp seç).
2. Model açılır menüsünden seç — varsayılan **lucida** (`egeorcun/lucida`, HuggingFace'ten indirilir).
3. Sonuç şeffaflık damalı zemin üzerinde önizlenir; **Download PNG** ile indir.

## Model ağırlıkları hakkında

Ağırlıklar imaja **gömülü değildir**. Seçilen model ilk istekte
HuggingFace'ten indirilir ve `HF_HOME` (`/root/.cache/huggingface`) altına
yazılır. Yukarıdaki `-v hf-cache:...` volume'u sayesinde konteyner yeniden
başlatıldığında tekrar indirme yapılmaz. Volume vermezseniz her yeni
konteynerde model baştan iner (lucida ~1 GB).

Not: `bgr-v1`...`lucida-v6` girdileri lokal checkpoint (`data/checkpoints/*.pth`)
ister; bu dosyalar imajda yoktur. Konteynerde HF tabanlı modelleri kullanın:
`lucida`, `birefnet-hr`, `rmbg-2.0`, `inspyrenet`.

## Beklenen süre (CPU)

İmaj CPU-only torch wheel'i ile derlenir. BiRefNet 1024 px çözünürlükte
CPU'da **görsel başına yaklaşık 5-15 saniye** sürer (çekirdek sayısına ve
görsel boyutuna göre değişir). İlk istek buna ek olarak model indirme +
yükleme süresini içerir.

## GPU ile çalıştırma

NVIDIA GPU'da çalıştırmak için:

1. `Dockerfile`'daki torch kurulum satırını CUDA wheel'ine çevirin:

   ```dockerfile
   RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. İmajı yeniden derleyip `--gpus all` ile çalıştırın
   ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) gerekir):

   ```bash
   docker build -t lucida-gpu .
   docker run --gpus all -p 8000:8000 -v hf-cache:/root/.cache/huggingface lucida-gpu
   ```

## API

Web arayüzü olmadan doğrudan API de kullanılabilir:

```bash
# Arka planı kaldır (PNG döner)
curl -F "file=@foto.jpg" "http://localhost:8000/remove?model=lucida" -o out.png

# Kullanılabilir modeller
curl http://localhost:8000/models

# Sağlık kontrolü
curl http://localhost:8000/health
```

`/remove` parametreleri: `model` (varsayılan `rmbg-2.0`), `refine` (bool),
`decontaminate` (bool, varsayılan `true`).
