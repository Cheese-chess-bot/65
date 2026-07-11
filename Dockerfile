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

# Ensure model directory exists and fetch the exact 1.5B GGUF file
RUN mkdir -p /app/model && \
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir /app/model --local-dir-use-symlinks False

# Copy the pipeline script over
COPY pipeline.py /app/pipeline.py

# Force permissions so the script is universally readable and executable
RUN chmod +x /app/pipeline.py

# Double-layer entrypoint structure to prevent runtime launching parsing bugs
ENTRYPOINT ["python"]
CMD ["/app/pipeline.py"]
