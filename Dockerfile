FROM python:3.10-slim

# Eliminate buffer delays and isolate caching trees
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install lightweight system essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install optimized CPU-only PyTorch and Transformers footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers \
    accelerate

# Pull the model parameters cleanly using Python native scripts instead of CLI
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)"

# Copy execution script and configs into place
COPY pipeline.py /app/pipeline.py

ENTRYPOINT ["python", "/app/pipeline.py"]
