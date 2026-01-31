# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "huggingface_hub[hf_xet]==0.36.0",
#    "onnxruntime-gpu==1.23.2",
#    "onnxscript==0.5.7",
#    "open-clip-torch==3.2.0",
#    "pillow==12.1.0",
#    "torch==2.10.0",
#    "torchvision==0.25.0",
#    "transformers==4.57.6",
#    "timm==1.0.24",
# ]
# ///

import torch
import torch.nn as nn
import json
import os
import shutil
import open_clip
from huggingface_hub import hf_hub_download
from timm.utils import reparameterize_model


def export_model(repo_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download HF Files
    print(f"--- Processing Repo: {repo_id} ---")
    print("Downloading configuration files...")
    config_files = ["open_clip_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    downloaded_paths = {}

    for filename in config_files:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            dest = os.path.join(output_dir, filename)
            shutil.copy(path, dest)
            downloaded_paths[filename] = dest
            print(f"  ✓ {filename}")
        except Exception:
            print(f"  ✗ {filename} (Optional or Missing)")

    # 2. Load Model
    # We use hf-hub: prefix to ensure open_clip pulls from the correct location
    model_id = f"hf-hub:{repo_id}"
    print(f"Loading model {model_id}...")

    # create_model_and_transforms is more robust for various CLIP flavors
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model.eval()

    # CRITICAL: Reparameterize (required for MobileCLIP/FastViT to merge branches)
    try:
        model = reparameterize_model(model)
        model.eval()
        print("  ✓ Model reparameterized successfully.")
    except Exception:
        print("  (i) Model does not support/require reparameterization.")

    # 3. Extract Weights and Configs
    logit_scale = model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item() if hasattr(model, 'logit_bias') and model.logit_bias is not None else 0.0

    print("Building model_config.json...")
    with open(downloaded_paths["open_clip_config.json"], "r") as f:
        raw_config = json.load(f)

    model_cfg = raw_config.get("model_cfg", {})

    # Detection logic: SigLIP uses Sigmoid and usually has an initial logit bias
    is_siglip = "siglip" in repo_id.lower() or "init_logit_bias" in model_cfg
    default_pad_id = 1 if (is_siglip and "siglip2" not in repo_id.lower()) else 0

    config = {
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
        "activation_function": 'sigmoid' if is_siglip else 'softmax',
        "tokenizer_needs_lowercase": True if is_siglip else False,
        "pad_id": default_pad_id,
    }

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 4. Wrap for ONNX
    class VisualWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # normalize=True is standard for CLIP inference
            return self.model.encode_image(x, normalize=True)

    class TextWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.encode_text(x, normalize=True)

    # 5. Export Execution
    # Batch size has to be >1 to enable dynamic axes
    BATCH_SIZE = 2
    img_size = model_cfg.get("vision_cfg", {}).get("image_size")
    ctx_len = model_cfg.get("text_cfg", {}).get("context_length")
    vocab_size = model_cfg.get("text_cfg", {}).get("vocab_size")
    dummy_image = torch.randn(BATCH_SIZE, 3, img_size, img_size)
    dummy_text = torch.randint(0, vocab_size, (BATCH_SIZE, ctx_len), dtype=torch.long)

    print(f"Exporting Visual Tower (Size: {img_size})...")
    visual_encoder = VisualWrapper(model)
    visual_encoder.eval()
    torch.onnx.export(
        visual_encoder,
        dummy_image,
        os.path.join(output_dir, "visual.onnx"),
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True
    )

    print(f"Exporting Text Tower (Ctx: {ctx_len})...")
    text_encoder = TextWrapper(model)
    text_encoder.eval()
    torch.onnx.export(
        text_encoder,
        dummy_text,
        os.path.join(output_dir, "text.onnx"),
        input_names=["input_ids"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True
    )

    print(f"Done! Files created in {output_dir}\n")


if __name__ == "__main__":
    mobileclip_hf_id = "timm/MobileCLIP2-S4-OpenCLIP"
    siglip2_hf_id = "timm/ViT-SO400M-16-SigLIP2-384"
    openai_hf_id = "timm/vit_base_patch32_clip_224.openai"
    siglip1_hf_id = "timm/ViT-SO400M-14-SigLIP-384"
    big_g_hf_id = "timm/PE-Core-bigG-14-448"
    laion_hf_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    export_model(laion_hf_id, "assets/" + laion_hf_id)
