# Use NVIDIA CUDA runtime image with cuDNN support.
# Required for GPU-enabled PyTorch inference in production clusters.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Ensure Python outputs logs immediately (useful for Kubernetes logging)
ENV PYTHONUNBUFFERED=1

# Install system dependencies:
# - python3 + pip for runtime
# - git (sometimes required by pip for certain dependencies)
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching.
# If requirements.txt doesn't change, Docker skips reinstalling deps.
COPY requirements.txt .

# Install Python dependencies without caching to keep image smaller.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI application using Uvicorn ASGI server
# Host 0.0.0.0 allows external container access (required in Docker/K8s)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]