# Lucida background remover — CPU imajı.
# Model ağırlıkları imaja gömülmez; ilk istekte HuggingFace'ten indirilir.
# Kalıcı cache için: -v hf-cache:/root/.cache/huggingface
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# opencv (transparent-background bağımlılığı) çalışma zamanında libGL ister.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch'u önce CPU wheel'inden kur — varsayılan CUDA wheel'leri imajı
# gereksizce (~GB'larca) büyütür. GPU imajı için bu satırı kaldırıp
# `--index-url https://download.pytorch.org/whl/cu124` kullanın.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Paket kurulumu (wheel yalnızca bgr/ ve benchmark/ içerir).
COPY pyproject.toml ./
COPY bgr/ bgr/
COPY benchmark/ benchmark/
RUN pip install .

# Servis kodu pakete dahil değil; workdir'den import edilir.
COPY serving/ serving/

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
