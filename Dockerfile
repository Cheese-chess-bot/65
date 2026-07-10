FROM python:3.10-slim

# Force completely isolated system configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements cleanly
RUN pip install --upgrade pip && \
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers accelerate huggingface_hub

# Safely pre-download the parameters directly into /app/model
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='/app/model', ignore_patterns=['*.bin', '*.pt'])"

# Copy the pipeline script over
COPY pipeline.py /app/pipeline.py

# Force permissions so the script is universally readable and executable
RUN chmod +x /app/pipeline.py

# Double-layer entrypoint structure to prevent runtime launching parsing bugs
ENTRYPOINT ["python"]
CMD ["/app/pipeline.py"]
