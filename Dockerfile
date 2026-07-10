FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install packages including huggingface_hub
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    huggingface_hub

# Safely download only the config and safetensors blocks without loading them into memory
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_files_only=False, ignore_patterns=['*.bin', '*.pt'])"

COPY pipeline.py /app/pipeline.py

ENTRYPOINT ["python", "/app/pipeline.py"]
