FROM python:3.10-slim

# Force completely isolated system configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements with the fast pre-compiled llama-cpp wheel
RUN pip install --upgrade pip && \
    pip install huggingface_hub && \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Ensure model directory exists and fetch the exact Gemma 2B GGUF file reliably
RUN mkdir -p /app/model && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Bartowski/gemma-2-2b-it-GGUF', filename='gemma-2-2b-it-Q4_K_M.gguf', local_dir='/app/model')"

# Copy the pipeline script over
COPY pipeline.py /app/pipeline.py

# Force permissions so the script is universally readable and executable
RUN chmod +x /app/pipeline.py

# Double-layer entrypoint structure to prevent runtime launching parsing bugs
ENTRYPOINT ["python"]
CMD ["/app/pipeline.py"]
