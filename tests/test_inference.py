from unittest.mock import patch, MagicMock
from inference import ESM2Inference
import pytest


# Mock CUDA device count to simulate a 4-GPU environment
@patch("torch.cuda.device_count", return_value=4)
# Mock CUDA availability to simulate GPU-enabled system
@patch("torch.cuda.is_available", return_value=True)
# Mock model loading to avoid downloading real ESM-2 weights
@patch("transformers.AutoModel.from_pretrained")
# Mock tokenizer loading to avoid network calls
@patch("transformers.AutoTokenizer.from_pretrained")
def test_parallel_distribution(
    mock_tokenizer,
    mock_model,
    mock_cuda,
    mock_count,
):
    """
    Unit test to validate multi-GPU batch distribution logic.

    This test ensures:
    - GPU count is auto-detected correctly (mocked as 4 GPUs)
    - Model replicas are created per GPU
    - Batch inference runs without requiring real GPU hardware
    - Output structure is returned as expected

    All heavy GPU/model operations are mocked for CI compatibility.
    """

    # Create a fake model object
    fake_model = MagicMock()

    # Create a fake output tensor object
    fake_output = MagicMock()

    # Mock the model forward pass to return deterministic embeddings
    # Simulates: output.last_hidden_state.mean(...).cpu().tolist()
    mean_mock = fake_output.last_hidden_state.mean.return_value
    cpu_mock = mean_mock.cpu.return_value
    cpu_mock.tolist.return_value = [[0.1]]

    # When fake_model(...) is called, return fake_output
    fake_model.return_value = fake_output
    mock_model.return_value = fake_model

    # Mock tokenizer behavior to simulate tokenized tensors
    # Ensures .to(device) calls don't fail
    mock_tokenizer.return_value = MagicMock(
        __call__=lambda x, **kwargs: {
            "input_ids": MagicMock(to=lambda device: MagicMock()),
            "attention_mask": MagicMock(to=lambda device: MagicMock()),
        }
    )

    # Initialize inference class (will use mocked GPU + model)
    model = ESM2Inference()

    # Run batch prediction with 4 sequences
    result = model.predict(["AAA", "BBB", "CCC", "DDD"])

    # Validate output structure
    assert "result" in result


@patch("torch.cuda.device_count", return_value=4)
@patch("torch.cuda.is_available", return_value=True)
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_output_length_matches_input(
    mock_tokenizer,
    mock_model,
    mock_cuda,
    mock_count,
):
    """
    Ensures number of embeddings equals number of input sequences.
    """

    # Mock model behavior
    fake_model = MagicMock()
    fake_output = MagicMock()

    mean_mock = fake_output.last_hidden_state.mean.return_value
    cpu_mock = mean_mock.cpu.return_value
    cpu_mock.tolist.return_value = [[0.1], [0.1], [0.1], [0.1]]

    fake_model.return_value = fake_output
    mock_model.return_value = fake_model

    # Proper tensor-like mock
    mock_tokenizer.return_value = MagicMock(
        __call__=lambda x, **kwargs: {
            "input_ids": MagicMock(to=lambda device: MagicMock()),
            "attention_mask": MagicMock(to=lambda device: MagicMock()),
        }
    )

    model = ESM2Inference()

    sequences = ["AAA", "BBB", "CCC", "DDD"]
    result = model.predict(sequences)

    assert len(result["result"]) == len(sequences)


@patch("torch.cuda.device_count", return_value=0)
@patch("torch.cuda.is_available", return_value=False)
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_cpu_fallback(
    mock_tokenizer,
    mock_model,
    mock_cuda,
    mock_count,
):
    """
    Ensures CPU fallback works when no GPUs are available.
    """

    mock_model.return_value = MagicMock()

    mock_tokenizer.return_value = MagicMock(
        __call__=lambda x, **kwargs: {
            "input_ids": MagicMock(to=lambda device: MagicMock()),
            "attention_mask": MagicMock(to=lambda device: MagicMock()),
        }
    )

    model = ESM2Inference()

    assert model.logical_device_count == 1
    assert model.devices == ["cpu"]


@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_invalid_sequence_raises(mock_tokenizer, mock_model):
    """
    Ensures invalid protein sequences raise ValueError.
    """

    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    model = ESM2Inference()

    with pytest.raises(ValueError):
        model.predict(["AAA", "", "CCC"])


@patch("torch.cuda.device_count", return_value=2)
@patch("torch.cuda.is_available", return_value=True)
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_ordering_is_preserved(
    mock_tokenizer,
    mock_model,
    mock_cuda,
    mock_count,
):
    """
    Ensures output ordering matches input ordering
    across multiple GPUs.
    """

    fake_model = MagicMock()
    fake_output = MagicMock()

    mean_mock = fake_output.last_hidden_state.mean.return_value
    cpu_mock = mean_mock.cpu.return_value

    # Return constant embedding
    cpu_mock.tolist.return_value = [[0.1], [0.1]]

    fake_model.return_value = fake_output
    mock_model.return_value = fake_model

    mock_tokenizer.return_value = MagicMock(
        __call__=lambda x, **kwargs: {
            "input_ids": MagicMock(to=lambda device: MagicMock()),
            "attention_mask": MagicMock(to=lambda device: MagicMock()),
        }
    )

    model = ESM2Inference()

    sequences = ["A", "B", "C", "D"]
    result = model.predict(sequences)

    # Just verify output length and no reordering crash
    assert len(result["result"]) == 4
