FROM node:24-slim

# Force completely isolated system configurations for Python and Node
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production

WORKDIR /app

# Install lightweight system dependencies, Python 3, and core build tools
# Added python3-venv to guarantee pip3 binary availability in minimal slim distros
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use the robust update-alternatives system instead of manual symlinking
# This prevents path collision bugs on modern minimal Debian base layers
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip3 and install requirements with the fast pre-compiled llama-cpp wheel.
RUN pip3 install --upgrade pip --break-system-packages && \
    pip3 install huggingface_hub --break-system-packages && \
    pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --break-system-packages

# Provision model path and fetch the lightweight 3-bit K-quant Gemma file securely
RUN mkdir -p /app/model && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Bartowski/gemma-2-2b-it-GGUF', filename='gemma-2-2b-it-Q3_K_L.gguf', local_dir='/app/model')"

# Copy the execution layer over
COPY pipeline.py /app/pipeline.py

# Force full execution permissions across evaluation runtime files
RUN chmod +x /app/pipeline.py

# Double-layer entrypoint structure to eliminate runtime shell escaping bugs
ENTRYPOINT ["python"]
CMD ["/app/pipeline.py"]
