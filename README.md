# CLIP Embedder

* To run this, you need to put the onnx file for [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) at `data/clip_text_encoder.onnx`.
* You can generate this onnx file with the following Python script:

Run the script below with `uv run create-onnx.py`, [uv](https://docs.astral.sh/uv/getting-started/installation/) is required for this.

### `create-onnx.py:`

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers",
#   "onnx",
#   "pillow",
# ]
# ///

import torch
from transformers import CLIPModel, CLIPProcessor

# Load the CLIP model and processor
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Extract the text encoder
text_encoder = model.text_model

# Example input for the text encoder
dummy_input = processor(text=["A sample text"], return_tensors="pt", padding=True)

# Export the text encoder to ONNX
torch.onnx.export(
    text_encoder,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "clip_text_encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=20,
)
```