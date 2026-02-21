"""
Simulated benchmark demonstrating expected throughput scaling
from 1 to 8 GPUs.

This benchmark does NOT require real GPUs.
GPU count is mocked to simulate different hardware configurations.
"""

import time
from unittest.mock import patch
from inference import ESM2Inference


def benchmark():
    """
    Simulates inference throughput under different GPU counts.

    Strategy:
    - Mock torch.cuda.device_count() to simulate 1, 2, 4, and 8 GPUs
    - Use artificial latency per device to emulate compute cost
    - Measure throughput (sequences per second)
    """

    # Simulated batch of 64 protein sequences (max supported size)
    batch = ["MADEUPSEQ"] * 64

    # Number of repeated batch requests for measurement
    num_requests = 10

    print("\nSimulated Multi-GPU Scaling:\n")

    # Test scaling behavior across different GPU counts
    for gpus in [1, 2, 4, 8]:

        # Mock CUDA device detection to simulate hardware scaling
        with patch("torch.cuda.device_count", return_value=gpus):

            # simulate_latency adds artificial per-GPU compute delay
            # This allows demonstration of scaling behavior without real GPUs
            model = ESM2Inference(simulate_latency=0.05)

            start = time.time()

            # Run repeated batch inference calls
            for _ in range(num_requests):
                model.predict(batch)

            end = time.time()

            # Compute throughput: total sequences processed per second
            throughput = (num_requests * 64) / (end - start)

            print(f"{gpus} GPUs (simulated) -> {throughput:.2f} seq/sec")


if __name__ == "__main__":
    benchmark()
