# This class handles model loading and prediction
from inference import ESM2Inference

def test_predict():
    """
    Unit test for the ESM2Inference.predict() function.

    This test verifies that the model can successfully process
    a single input sequence and return one output embedding.
    """

    # Initialize the inference model

    model = ESM2Inference()

     # Run prediction on a mock protein sequence

    output = model.predict("MADEUPSEQ")

    # Assert that the output list contains exactly one result
    
    assert len(output) == 1