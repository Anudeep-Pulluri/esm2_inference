# Use a lightweight ARM-compatible base image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies (CPU/MPS version of PyTorch)
RUN pip install --no-cache-dir torch torchvision torchaudio transformers fastapi uvicorn pytest

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]