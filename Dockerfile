FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    huggingface_hub

RUN huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --exclude "*.bin" "*.pt"

COPY pipeline.py /app/pipeline.py

ENTRYPOINT ["python", "/app/pipeline.py"]
