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

# 1. Download HF Files first (we need these to build our model_config.json)
print("Downloading configuration files...")
config_files = ["open_clip_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
downloaded_paths = {}

for filename in config_files:
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        dest = os.path.join(OUTPUT_DIR, filename)
        shutil.copy(path, dest)
        downloaded_paths[filename] = dest
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename} (Optional or Missing)")

# 2. Load Model & Extract Live Weights
print(f"Loading {MODEL_ID}...")
model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
model.eval()

# Extract trained parameters
# Note: we .exp() the scale because Open-CLIP stores it in log-space
logit_scale = model.logit_scale.exp().item()
logit_bias = model.logit_bias.item() if hasattr(model, 'logit_bias') and model.logit_bias is not None else 0.0

# 3. Build the flattened model_config.json
print("Building flattened model_config.json...")

# Load the raw OpenCLIP config to extract metadata
with open(downloaded_paths["open_clip_config.json"], "r") as f:
    raw_config = json.load(f)

model_cfg = raw_config.get("model_cfg", {})
preprocess_cfg = raw_config.get("preprocess_cfg", {})

# Determine model type (SigLIP models have an initial logit bias)
is_siglip = "init_logit_bias" in model_cfg or "siglip" in MODEL_ID.lower()

config = {
    "model_type": "siglip" if is_siglip else "clip",
    "embed_dim": model_cfg.get("embed_dim"),
    "image_size": model_cfg.get("vision_cfg", {}).get("image_size"),
    "context_length": model_cfg.get("text_cfg", {}).get("context_length"),
    "vocab_size": model_cfg.get("text_cfg", {}).get("vocab_size"),

    # These are the LIVE trained values from the model object
    "logit_scale": logit_scale,
    "logit_bias": logit_bias,

    # Preprocessing
    "mean": preprocess_cfg.get("mean"),
    "std": preprocess_cfg.get("std"),
    "interpolation": preprocess_cfg.get("interpolation"),
    "resize_mode": preprocess_cfg.get("resize_mode")
}

with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# 4. Wrap for ONNX
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

# 5. Export
BATCH_SIZE = 2
img_size = config["image_size"]
ctx_len = config["context_length"]

dummy_image = torch.randn(BATCH_SIZE, 3, img_size, img_size)
dummy_text = torch.randint(0, config["vocab_size"], (BATCH_SIZE, ctx_len), dtype=torch.long)

print("Exporting Visual Tower...")
torch.onnx.export(
    VisualWrapper(model),
    dummy_image,
    os.path.join(OUTPUT_DIR, "visual.onnx"),
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeddings": {0: "batch_size"}},
    opset_version=18,
    do_constant_folding=True
)

print("Exporting Text Tower...")
torch.onnx.export(
    TextWrapper(model),
    dummy_text,
    os.path.join(OUTPUT_DIR, "text.onnx"),
    input_names=["input_ids"],
    output_names=["text_embeddings"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "text_embeddings": {0: "batch_size"}},
    opset_version=18,
    do_constant_folding=True
)

print(f"\nDone! Files created in {OUTPUT_DIR}")