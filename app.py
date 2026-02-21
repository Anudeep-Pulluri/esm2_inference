# FastAPI framework for building the inference API
from fastapi import FastAPI, HTTPException

# Pydantic for request validation and schema enforcement
from pydantic import BaseModel

# Typing support for batch inputs
from typing import List

# Core inference engine (handles multi-GPU logic)
from inference import ESM2Inference

# Initialize FastAPI application with metadata
app = FastAPI(
    title="ESM-2 Multi-GPU Inference API",
    description="Production-ready multi-GPU inference service for ESM-2",
    version="1.0.0",
)


# Initialize the inference model once at startup.
# This ensures model replicas are loaded into GPU memory only once,
# avoiding repeated initialization per request.
model = ESM2Inference()


# Schema for single-sequence inference request
class SequenceInput(BaseModel):
    # Amino acid sequence string
    sequence: str


# Schema for batch inference request
class BatchInput(BaseModel):
    # List of amino acid sequences
    sequences: List[str]


# Health endpoint for Kubernetes probes and monitoring systems
@app.get("/health")
def health():
    """
    Reports service readiness and GPU availability.

    Used by:
    - Kubernetes startupProbe
    - readinessProbe
    - livenessProbe
    - Monitoring systems
    """

    # If model is still loading, report service unavailable
    if not model.ready:
        raise HTTPException(status_code=503, detail="Model loading")

    return {
        "status": "ok",
        # Whether physical CUDA devices are present
        "physical_cuda_available": model.physical_cuda_available,
        # Number of logical devices detected (0, 1, 4, 8)
        "logical_devices": model.logical_device_count,
        # List of device identifiers used for model replicas
        "devices": model.devices,
    }


# Single sequence inference endpoint
@app.post("/predict")
def predict_single(data: SequenceInput):
    """
    Performs inference on a single protein sequence.
    Automatically routed to appropriate GPU replica.
    """
    return model.predict(data.sequence)


# Batch inference endpoint (max 64 sequences)
@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    """
    Performs distributed batch inference across available GPUs.

    The batch is automatically split across detected GPU replicas
    for parallel processing.
    """

    # Enforce batch size limit to prevent OOM and ensure predictable latency
    if len(data.sequences) > 64:
        raise HTTPException(status_code=400, detail="Batch limit is 64")

    return model.predict(data.sequences)
