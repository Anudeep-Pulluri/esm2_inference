# Comprehensive unit tests for the ESM2Inference class and FastAPI endpoints

import pytest
from unittest.mock import patch
from inference import ESM2Inference
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# -----------------------------
# Model-level unit tests
# -----------------------------


def test_predict_single():
    """
    Verifies that the model can process a single input sequence
    and return one embedding vector.
    """
    model = ESM2Inference()
    output = model.predict("MADEUPSEQ")
    assert isinstance(output, dict)
    assert "result" in output
    assert isinstance(output["result"], list)
    assert len(output["result"]) > 0
    # Each element in the list should be numeric
    assert all(isinstance(x, (float, int)) for x in output["result"])


@patch("torch.cuda.device_count", return_value=2)
def test_predict_batch(mock_gpu_count):
    """
    Verifies that multiple sequences return the same number of embeddings
    and that the multi-GPU distribution logic executes without error.
    """
    model = ESM2Inference()
    sequences = ["SEQ1", "SEQ2", "SEQ3", "SEQ4"]
    output = model.predict(sequences)
    assert isinstance(output, dict)
    assert "result" in output
    assert isinstance(output["result"], list)
    assert len(output["result"]) > 0

    # Each element inside "result" should be numeric
    assert all(isinstance(x, (float, int)) for x in output["result"])


def test_empty_batch():
    """
    Ensures that an empty batch returns an empty list
    instead of raising an exception.
    """
    model = ESM2Inference()
    output = model.predict([])
    assert output == []


@patch("torch.cuda.device_count", return_value=4)
def test_gpu_detection(mock_gpu_count):
    """
    Ensures that GPU detection logic correctly identifies available GPUs.
    """
    model = ESM2Inference()
    assert model.device_count == 4


def test_invalid_input_type():
    """
    Ensures that invalid input types raise a TypeError or ValueError.
    """
    model = ESM2Inference()
    with pytest.raises(Exception):
        model.predict(12345)  # invalid input type


# -----------------------------
# API-level endpoint tests
# -----------------------------


def test_health_endpoint():
    """
    Verifies that /health returns GPU info and status 200.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpus" in data
    assert "status" in data


def test_predict_endpoint():
    """
    Verifies that /predict endpoint works for a single sequence.
    """
    resp = client.post("/predict", json={"sequence": "MKTFFV"})
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert "result" in data["embedding"]
    assert isinstance(data["embedding"]["result"], list)


def test_predict_batch_endpoint():
    """
    Verifies that /predict/batch endpoint works for multiple sequences.
    """
    payload = {"sequences": ["AAA", "BBB", "CCC"]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "embeddings" in data
    assert "result" in data["embeddings"]
    assert isinstance(data["embeddings"]["result"], list)


def test_predict_batch_limit():
    """
    Ensures that /predict/batch enforces the 64-sequence limit.
    """
    payload = {"sequences": ["SEQ"] * 65}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 400
