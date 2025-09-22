# ---------- базовый образ ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ---------- обновляем pip и устанавливаем пакеты ----------
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

# ---------- создаем юзера ----------
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# ---------- зависимости ----------
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- код ----------
COPY --chown=appuser:appuser . .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
