# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "huggingface-hub==0.36.0",
#    "onnxruntime-gpu==1.23.2",
#    "onnxscript==0.5.7",
#    "open-clip-torch==3.2.0",
#    "pillow==12.1.0",
#    "torch==2.10.0",
#    "torchvision==0.25.0",
#    "transformers==4.57.6",
# ]
# ///

import torch
import torch.nn as nn
import json
import os
import shutil
import open_clip
from huggingface_hub import hf_hub_download

# --- CONFIG ---
MODEL_ID = 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384'
HF_REPO_ID = "timm/ViT-SO400M-16-SigLIP2-384"
OUTPUT_DIR = "assets/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model
print(f"Loading {MODEL_ID}...")
model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
model.eval()

logit_scale = model.logit_scale.exp().item()
logit_bias = model.logit_bias.item()

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

# 2. Dummy Inputs - CRITICAL: Use batch size > 1 to avoid hardcoding batch=1 in the graph
BATCH_SIZE = 2
dummy_image = torch.randn(BATCH_SIZE, 3, 384, 384)
dummy_text = torch.randint(0, 256000, (BATCH_SIZE, 64), dtype=torch.long)

visual_wrapper = VisualWrapper(model).eval()
text_wrapper = TextWrapper(model).eval()

# 3. Export Visual Tower
print("Exporting Visual Tower (with batch size awareness)...")
torch.onnx.export(
    visual_wrapper,
    dummy_image,
    os.path.join(OUTPUT_DIR, "visual.onnx"),
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "image_embeddings": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=True
)

# 4. Export Text Tower
print("Exporting Text Tower...")
torch.onnx.export(
    text_wrapper,
    dummy_text,
    os.path.join(OUTPUT_DIR, "text.onnx"),
    input_names=["input_ids"],
    output_names=["text_embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "text_embeddings": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=True
)

# 5. Save Config
config = {
    "model_name": MODEL_ID,
    "logit_scale": logit_scale,
    "logit_bias": logit_bias,
    "image_size": 384,
    "context_length": 64,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5]
}
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# 6. Tokenizer
print("Downloading tokenizer files...")
for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        shutil.copy(path, os.path.join(OUTPUT_DIR, filename))
    except: pass

print(f"\nDone! Re-exported models to {OUTPUT_DIR}")