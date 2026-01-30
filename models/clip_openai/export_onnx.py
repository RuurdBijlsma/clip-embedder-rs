# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "huggingface-hub==0.36.0",
#    "onnxruntime-gpu==1.23.2",
#    "onnxscript==0.5.7",
#    "pillow==12.1.0",
#    "torch==2.10.0",
#    "torchvision==0.25.0",
#    "transformers==4.57.6",
# ]
# ///
# Run with `uv run export_onnx.py`
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import os

# 1. Setup Model
model_id = "openai/clip-vit-base-patch32"
output_dir = "assets/model"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading {model_id}...")
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

logit_scale = model.logit_scale.exp().item()
with open(os.path.join(output_dir, "model_config.json"), "w") as f:
    json.dump({"logit_scale": logit_scale}, f)
print(f"Logit scale extracted: {logit_scale:.4f}")

model.eval()
for param in model.parameters():
    param.requires_grad = False


# 2. Define Wrappers
# Note: We explicitly normalize the output to match the behavior
# of the open_clip script (normalize=True), making it ready for cosine similarity.

class VisualWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        # HF implementation details to get embeddings
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        # Normalize
        return F.normalize(image_embeds, p=2, dim=1)


class TextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask):
        # HF implementation details to get embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        # Normalize
        return F.normalize(text_embeds, p=2, dim=1)


# 3. Dummy Inputs
# ViT-B/32 uses 224x224 images
dummy_image = torch.randn(1, 3, 224, 224)
# Standard text inputs
dummy_input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)
dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)

# 4. Instantiate Wrappers
visual_wrapper = VisualWrapper(model).eval()
text_wrapper = TextWrapper(model).eval()

# 5. Export Visual Tower
print("Exporting Visual Tower to visual.onnx...")
torch.onnx.export(
    visual_wrapper,
    dummy_image,
    f"{output_dir}/visual.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "image_embeddings": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=True
)

# 6. Export Text Tower
print("Exporting Text Tower to text.onnx...")
torch.onnx.export(
    text_wrapper,
    (dummy_input_ids, dummy_attention_mask),
    f"{output_dir}/text.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["text_embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "text_embeddings": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=True
)

# 7. Save Processor (Tokenizer + Feature Extractor config)
print("Saving processor files...")
processor.save_pretrained(output_dir)

print(f"Done! Files created in {output_dir}")
