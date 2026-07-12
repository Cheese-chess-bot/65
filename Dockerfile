FROM node:24-slim

# Force completely isolated system configurations for Python and Node
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production

WORKDIR /app

# Install lightweight system dependencies, Python 3, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink so "python" calls "python3" automatically
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install requirements with the fast pre-compiled llama-cpp wheel
RUN pip install --upgrade pip --break-system-packages && \
    pip install huggingface_hub --break-system-packages && \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --break-system-packages

# Ensure model directory exists and fetch the faster 3-bit K-quant Gemma file reliably
RUN mkdir -p /app/model && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Bartowski/gemma-2-2b-it-GGUF', filename='gemma-2-2b-it-Q3_K_L.gguf', local_dir='/app/model')"

# Copy the pipeline script over
COPY pipeline.py /app/pipeline.py

# Force permissions so the script is universally readable and executable
RUN chmod +x /app/pipeline.py

# Double-layer entrypoint structure to prevent runtime launching parsing bugs
ENTRYPOINT ["python"]
CMD ["/app/pipeline.py"]
