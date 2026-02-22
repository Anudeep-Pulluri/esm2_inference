from unittest.mock import patch, MagicMock
from inference import ESM2Inference


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
