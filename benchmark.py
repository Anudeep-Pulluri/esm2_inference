import time

# Import the ESM2Inference class from inference.py

from inference import ESM2Inference


def benchmark():
    """
    Benchmark the simulated multi-GPU inference performance.

    This function measures how inference time scales when the number of

    available GPUs increases from 1 to 8. Since this environment has no

    real GPUs, the GPU count is mocked to simulate scaling behavior.
    """

    # Initialize the inference model

    model = ESM2Inference()

    # Create a batch of 64 identical sequences for testing

    sequences = ["MADEUPSEQ"] * 64  # batch of 64 sequences

    print("\nBenchmarking simulated multi-GPU inference:\n")

    # Loop through different GPU counts (mocked)

    for gpus in [1, 2, 4, 8]:

        model.device_count = gpus  # mock GPU count

        # Record start time

        start = time.time()

        # Run inference on the batch

        model.predict(sequences)

        # Record end time

        end = time.time()

        # Print the time taken for this GPU configuration

        print(f"{gpus} GPUs -> {end - start:.2f} seconds")


if __name__ == "__main__":
    benchmark()
