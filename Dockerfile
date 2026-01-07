# LTX-2 Training Dockerfile for RunPod
# Based on NVIDIA CUDA image with cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Clone the repository
ARG GITHUB_TOKEN
RUN if [ -z "$GITHUB_TOKEN" ]; then \
        echo "Error: GITHUB_TOKEN build arg is required for private repo" && exit 1; \
    fi && \
    git clone https://${GITHUB_TOKEN}@github.com/aliabougazia/ltx-2.git /workspace/ltx-2

WORKDIR /workspace/ltx-2

# Install Python dependencies using uv
RUN uv sync

# Install runpod for serverless support
RUN /workspace/ltx-2/.venv/bin/pip install runpod requests

# Create directories for checkpoints and datasets
RUN mkdir -p /workspace/checkpoints /workspace/datasets

# Set up environment variables
ENV TOKENIZERS_PARALLELISM=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create entrypoint script
RUN echo '#!/bin/bash' > /workspace/entrypoint.sh && \
    echo 'set -e' >> /workspace/entrypoint.sh && \
    echo '' >> /workspace/entrypoint.sh && \
    echo 'echo "=================================================="' >> /workspace/entrypoint.sh && \
    echo 'echo "LTX-2 Training Container"' >> /workspace/entrypoint.sh && \
    echo 'echo "=================================================="' >> /workspace/entrypoint.sh && \
    echo 'echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"' >> /workspace/entrypoint.sh && \
    echo 'echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"' >> /workspace/entrypoint.sh && \
    echo 'echo "Python: $(python --version)"' >> /workspace/entrypoint.sh && \
    echo 'echo "Working Directory: $(pwd)"' >> /workspace/entrypoint.sh && \
    echo 'echo "=================================================="' >> /workspace/entrypoint.sh && \
    echo '' >> /workspace/entrypoint.sh && \
    echo '# If a command was provided, run it' >> /workspace/entrypoint.sh && \
    echo 'if [ $# -gt 0 ]; then' >> /workspace/entrypoint.sh && \
    echo '    exec "$@"' >> /workspace/entrypoint.sh && \
    echo 'else' >> /workspace/entrypoint.sh && \
    echo '    # Default: start bash' >> /workspace/entrypoint.sh && \
    echo '    exec /bin/bash' >> /workspace/entrypoint.sh && \
    echo 'fi' >> /workspace/entrypoint.sh && \
    chmod +x /workspace/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
