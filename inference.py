"""
inference.py

Multi-GPU ESM2 inference engine.

Capabilities:
- Automatic GPU detection (0, 1, 4, 8, etc.)
- CPU fallback when CUDA is unavailable
- One model replica per device
- Parallel batch distribution across devices
- Mock-friendly GPU simulation for CI
"""

import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import time


class ESM2Inference:
    """
    Core inference engine responsible for:
    - Loading model replicas per available device
    - Distributing batch inputs across devices
    - Executing inference in parallel
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        simulate_latency: float = 0.0,  # used for benchmark simulation
    ):
        self.model_name = model_name

        # Artificial latency for benchmark simulation (no real GPU required)
        self.simulate_latency = simulate_latency

        # Detect whether CUDA hardware exists
        self.physical_cuda_available = torch.cuda.is_available()

        # Detect number of visible CUDA devices
        self.logical_device_count = torch.cuda.device_count()

        # If no GPU detected, fallback to single CPU device
        if self.logical_device_count == 0:
            self.logical_device_count = 1

        # Store device identifiers and model replicas
        self.devices = []
        self.models = []

        # Readiness flag used by /health endpoint
        self.ready = False

        # Load tokenizer once (shared across replicas)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model replicas per device
        self._load_models()

        self.ready = True

    def _load_models(self):
        """
        Load one model replica per detected device.

        This enables parallel inference without inter-device locking.
        Each GPU receives its own independent model instance.
        """

        for i in range(self.logical_device_count):

            # Assign CUDA device if available, otherwise CPU
            if self.physical_cuda_available:
                device = f"cuda:{i}"
            else:
                device = "cpu"

            # Load pretrained ESM-2 model
            model = AutoModel.from_pretrained(self.model_name)

            # Move model to assigned device
            model.to(device)

            # Set to evaluation mode (disable dropout, training ops)
            model.eval()

            self.devices.append(device)
            self.models.append(model)

    def _infer_on_device(self, model, device, indexed_sequences):
        """
        Runs inference on a specific device (GPU or CPU).

        Parameters:
        - model: The ESM2 model replica assigned to this device
        - device: The device string (e.g., "cuda:0" or "cpu")
        - indexed_sequences: List of tuples (original_index, sequence)

        Returns:
        - List of tuples (original_index, embedding)
        """
        if self.simulate_latency > 0:
            time.sleep(self.simulate_latency)

        # Separate original indices and sequences
        indices = [item[0] for item in indexed_sequences]
        sequences = [item[1] for item in indexed_sequences]

        # Tokenize batch
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)

        # Move tensors to correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run forward pass without gradient tracking
        with torch.no_grad():
            outputs = model(**inputs)

        # Compute per-sequence embedding (mean pooling over tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # Move embeddings to CPU and convert to Python list
        embeddings = embeddings.cpu().tolist()

        # Return (index, embedding) pairs
        return list(zip(indices, embeddings))

    def predict(self, sequences: Union[str, List[str]]):
        """
        Perform distributed inference.

        Supports:
        - Single sequence (string)
        - Batch of sequences (list[str])

        Batch is distributed across devices using round-robin strategy.
        """

        # Normalize single input to list
        if isinstance(sequences, str):
            sequences = [sequences]

        if not isinstance(sequences, list):
            raise ValueError("Input must be a string or list of strings.")

        # Handle empty input gracefully
        if len(sequences) == 0:
            return {"result": []}

        num_devices = len(self.models)

        # Create one chunk per device
        # Round-robin ensures balanced distribution across GPUs
        chunks = [[] for _ in range(num_devices)]

        for idx, seq in enumerate(sequences):

            # Basic validation of protein sequence
            if not isinstance(seq, str) or len(seq.strip()) == 0:
                raise ValueError("Invalid protein sequence.")

            chunks[idx % num_devices].append((idx, seq))

        results = []

        # Execute inference in parallel across devices
        with ThreadPoolExecutor(max_workers=num_devices) as executor:

            futures = []

            for i in range(num_devices):
                if chunks[i]:  # Only submit work if device has data
                    futures.append(
                        executor.submit(
                            self._infer_on_device,
                            self.models[i],
                            self.devices[i],
                            chunks[i],
                        )
                    )

            # Collect results as they complete
            for future in as_completed(futures):
                results.extend(future.result())

        results.sort(key=lambda x: x[0])

        # Extract embeddings only
        ordered_embeddings = [embedding for _, embedding in results]

        return {"result": ordered_embeddings}
