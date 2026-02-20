# Import FastAPI for building the web API
from fastapi import FastAPI, HTTPException

# Import BaseModel from Pydantic for request validation
from pydantic import BaseModel

# Import the ESm2Inference class that handles model loading and prediction
from inference import ESM2Inference

from typing import List

# Intialize the FastAPI app
app = FastAPI()

# Create a single instance of the ESM2Inference model when the app starts

# This ensures the model is loaded only once and reused for all requests

model = ESM2Inference()

# Define a Pydantic model for validating incoming JSON requests


class SequenceInput(BaseModel):

    sequence: str  # The input protein sequence string


class BatchInput(BaseModel):

    sequences: List[str]  # List of protein sequences


# Health check endpoint

# Returns the number of GPUs detected (mocked if none exist)


@app.get("/health")
def health():
    return {"status": "ok", "gpus": model.device_count}


# Endpoint for single sequence inference

# Accepts one sequence and returns its model embedding or result


@app.post("/predict")
def predict_single(data: SequenceInput):
    result = model.predict(data.sequence)
    return {"embedding": result}


# Endpoint for batch inference

# Accepts a list of sequences (up to 64) and returns results for all


@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    # Enforce 64-sequence limit
    if len(data.sequences) > 64:
        raise HTTPException(status_code=400, detail="Batch size limit is 64")

    result = model.predict(data.sequences)
    return {"embeddings": result}
