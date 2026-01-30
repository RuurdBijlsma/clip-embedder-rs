# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "huggingface-hub==0.36.0",
#    "onnxscript==0.5.7",
#    "open-clip-torch==3.2.0",
#    "torch==2.10.0",
#    "torchvision==0.25.0",
# ]
# ///
# Run with `uv run export_onnx.py`

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained
from huggingface_hub import hf_hub_download
import shutil

# 1. Setup Model
model_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
print(f"Loading {model_name}...")
model, preprocess = create_model_from_pretrained(model_name)

model.eval()
for param in model.parameters():
    param.requires_grad = False


# 2. Define Wrappers
class VisualWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_image(x, normalize=True)


class TextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_text(x, normalize=True)


# 3. Dummy Inputs
dummy_image = torch.randn(1, 3, 384, 384)
dummy_text = torch.randint(0, 1000, (1, 64), dtype=torch.long)

# 4. Instantiate Wrappers
visual_wrapper = VisualWrapper(model).eval()
text_wrapper = TextWrapper(model).eval()

# 5. Export Visual Tower
print("Exporting Visual Tower to visual.onnx...")
torch.onnx.export(
    visual_wrapper,
    dummy_image,
    "visual.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeddings": {0: "batch_size"}},
    opset_version=18,
    do_constant_folding=True
)

# 6. Export Text Tower
print("Exporting Text Tower to text.onnx...")
torch.onnx.export(
    text_wrapper,
    dummy_text,
    "text.onnx",
    input_names=["input_ids"],
    output_names=["text_embeddings"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "text_embeddings": {0: "batch_size"}},
    opset_version=18,
    do_constant_folding=True
)

# 7. Save Tokenizer
print("Downloading tokenizer.json...")
config_path = hf_hub_download(repo_id="timm/ViT-SO400M-14-SigLIP-384", filename="tokenizer.json")
shutil.copy(config_path, "tokenizer.json")

print("Done! Files created successfully.")
