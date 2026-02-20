import torch
from transformers import AutoTokenizer, AutoModel

# For handling model loading and inference


class ESM2Inference:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        # Loads the tokenizer
        # prepares model replicas based on GPU availability.

        # Detect how many GPUs are available (returns 0 on CPU-only systems)

        self.device_count = torch.cuda.device_count()

        # Store the model name

        self.model_name = model_name

        # Placeholder for model replicas

        self.models = []

        # Load the tokenizer for the ESM2 model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model(s) into memory

        self._load_models()

    def _load_models(self):
        """
        Load the model onto the appropriate device.
        Uses MPS (Apple GPU) if available, otherwise CPU.
        """
        # Choose device: MPS for Apple Silicon, else CPU

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # Load the model and move it to the selected device

        model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Store the model in a list (mocked single device for now)

        self.models = [model]

    def predict(self, sequences):
        """
        Perform inference on one or more input sequences.
        Handles both single-sequence and batch inputs.
        """

        # Convert a single string input into a list

        if isinstance(sequences, str):

            sequences = [sequences]

        if not sequences:
            return []

        # Tokenize the input sequences and prepare tensors

        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)

        # Move all tensors to the selected device

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Placeholder for output embeddings

        outputs = []

        # Use the first (or only) model replica

        model = self.models[0]

        # Disable gradient computation for inference

        with torch.no_grad():
            out = model(**inputs)

        # Convert model output to a NumPy array and store it

        outputs.append(out.last_hidden_state.mean().cpu().numpy())

        # Convert NumPy arrays to Python lists for JSON serialization
        result = [out.tolist() for out in outputs]
        return {"result": result}
