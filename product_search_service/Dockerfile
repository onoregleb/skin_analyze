# ---------- базовый образ ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip curl wget procps && \
    rm -rf /var/lib/apt/lists/*

# ---------- создаем юзера ----------
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# ---------- обновляем pip и устанавливаем пакеты ----------
RUN python3 -m pip install --user --upgrade --no-cache-dir pip setuptools wheel

# ---------- зависимости ----------
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# ---------- добавляем пользовательский bin в PATH ----------
ENV PATH="/home/appuser/.local/bin:${PATH}"

# ---------- код ----------
COPY --chown=appuser:appuser . .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
