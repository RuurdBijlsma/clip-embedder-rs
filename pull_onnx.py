# ruff: noqa: G004, BLE001, EM101, TRY003, EM102
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "huggingface_hub[hf_xet]==0.36.0",
#    "onnxscript==0.6.0",
#    "open-clip-torch==3.2.0",
#    "rich==14.3.1",
# ]
# ///
import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import open_clip
import torch
from huggingface_hub import hf_hub_download
from rich.logging import RichHandler
from timm.utils import reparameterize_model
from torch import nn

# Setup Logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
)
logger = logging.getLogger("export_script")
logger.setLevel(logging.INFO)


@dataclass
class ExportConfig:
    hf_username: str = "RuteNL"
    batch_size: int = 2
    opset_version: int = 18
    config_files: tuple[str, ...] = (
        "open_clip_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "README.md",
        ".gitattributes",
    )


class VisualWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x, normalize=True)


class TextWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(x, normalize=True)


class HuggingFaceClient:
    """Handles interaction with Hugging Face Hub."""

    def __init__(self, repo_id: str, output_dir: Path) -> None:
        self.repo_id = repo_id
        self.output_dir = output_dir

    def download_configs(self, files: tuple[str, ...]) -> dict[str, Path]:
        downloaded = {}
        for filename in files:
            try:
                path = hf_hub_download(repo_id=self.repo_id, filename=filename)
                dest = self.output_dir / filename
                shutil.copy(path, dest)
                downloaded[filename] = dest
                logger.info(f"✓ {filename}")
            except Exception:
                logger.warning(f"✗ {filename} (Missing)")
        return downloaded


class ModelManager:
    """Handles model loading, reparameterization, and metadata extraction."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        model_id = f"hf-hub:{repo_id}"
        logger.info(f"Loading model: {model_id}")

        try:
            self.model, _, _ = open_clip.create_model_and_transforms(model_id)
        except Exception as e:
            raise ValueError(
                f"Failed to fetch model '{model_id}'. Does it exist?",
            ) from e
        self.model.eval()
        self._reparameterize()

    def _reparameterize(self) -> None:
        try:
            self.model = reparameterize_model(self.model)
            self.model.eval()
            logger.info("✓ Model reparameterized successfully.")
        except Exception:
            logger.info("Model does not require reparameterization.")

    def get_vocab_size(self) -> int:
        if hasattr(self.model, "vocab_size"):
            return self.model.vocab_size
        if hasattr(self.model, "token_embedding"):
            return self.model.token_embedding.weight.shape[0]
        try:
            return self.model.transformer.config.vocab_size
        except AttributeError as e:
            raise ValueError("Could not determine vocab_size.") from e

    def get_model_config(
            self,
            raw_config: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        model_cfg = raw_config.get("model_cfg", {})
        is_siglip = "siglip" in self.repo_id.lower() or "init_logit_bias" in model_cfg
        is_siglip2 = "siglip2" in self.repo_id.lower()

        logit_scale = self.model.logit_scale.exp().item()
        logit_bias = (
            self.model.logit_bias.item()
            if hasattr(self.model, "logit_bias") and self.model.logit_bias is not None
            else 0.0
        )

        return {
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
            "activation_function": "sigmoid" if is_siglip else "softmax",
            "tokenizer_needs_lowercase": is_siglip,
            "pad_id": 1 if (is_siglip and not is_siglip2) else 0,
            "vocab_size": self.get_vocab_size(),
        }


class ONNXExporter:
    """Handles the conversion to ONNX format."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def export(
            self,
            model: nn.Module,
            dummy_input: torch.Tensor,
            output_path: Path,
            input_name: str,
            output_name: str,
    ) -> None:
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"},
            },
            opset_version=self.config.opset_version,
            do_constant_folding=True,
        )
        logger.info(f"✓ Exported ONNX '{output_path}'")


def _modify_readme(readme_path: Path, repo_id: str) -> None:
    parts = repo_id.split("/", 1)
    if len(parts) == 2:
        model_name = parts[1]
    else:
        model_name = repo_id
    new_model_id = f"{ExportConfig.hf_username}/{model_name}-ONNX"

    rust_code = f"""
```rust
use open_clip_inference::Clip;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let model_id = "{new_model_id}";
    let mut clip = Clip::from_hf(model_id).build().await?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let results = clip.classify(&img, texts)?;

    for (text, prob) in results {{
        println!("{{}}: {{:.4}}%", text, prob * 100.0);
    }}

    Ok(())
}}
```
"""

    header = (
        f"# ONNX export of {repo_id}\n\n"
        f"This model is an export of [{repo_id}](https://huggingface.co/{repo_id}). "
        "It can be used with the [`open_clip_inference`](https://crates.io/crates/open_clip_inference) rust crate, or any other ONNX Runtime based implementation.\n\n"
        "## Usage with `open_clip_inference` in Rust: \n"
        f"{rust_code}\n"
        "---\n"
    )
    with readme_path.open("r", encoding="utf-8") as f:
        content = f.read()

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            # Remove any library_name lines
            frontmatter_lines = [
                line for line in frontmatter.splitlines()
                if not line.strip().startswith("library_name:")
            ]
            frontmatter = "\n".join(frontmatter_lines).strip("\n")
            new_content = (
                f"---\n{frontmatter}\n---\n\n"
                f"{header}\n"
                f"{parts[2].lstrip()}"
            )
        else:
            new_content = header + content
    else:
        new_content = header + content

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(new_content)


def run_export(repo_id: str, base_output_dir: str) -> None:
    output_dir = Path(base_output_dir) / repo_id

    config = ExportConfig()
    hf_client = HuggingFaceClient(repo_id, output_dir)
    model_mgr = ModelManager(repo_id)
    exporter = ONNXExporter(config)

    # Configs
    paths = hf_client.download_configs(config.config_files)

    if "README.md" in paths:
        _modify_readme(paths["README.md"], repo_id)
    if "open_clip_config.json" not in paths:
        raise FileNotFoundError("Could not find open_clip_config.json for metadata.")
    with Path.open(paths["open_clip_config.json"]) as f:
        open_clip_config = json.load(f)
    model_config = model_mgr.get_model_config(open_clip_config)
    clip_model_cfg = open_clip_config.get("model_cfg", {})
    img_size = int(clip_model_cfg.get("vision_cfg", {}).get("image_size"))
    ctx_len = int(clip_model_cfg.get("text_cfg", {}).get("context_length"))

    # Write `model_config.json`
    with Path.open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Export Visual ONNX
    logger.info(f"Exporting Visual Embedder (Size: {img_size})...")
    dummy_img = torch.randn(config.batch_size, 3, img_size, img_size)
    exporter.export(
        VisualWrapper(model_mgr.model),
        dummy_img,
        output_dir / "visual.onnx",
        "pixel_values",
        "image_embeddings",
    )

    # Export Text ONNX
    logger.info(f"Exporting Text Embedder (Ctx: {ctx_len})...")
    dummy_txt = torch.randint(
        0,
        model_config["vocab_size"],
        (config.batch_size, ctx_len),
        dtype=torch.long,
    )
    exporter.export(
        TextWrapper(model_mgr.model),
        dummy_txt,
        output_dir / "text.onnx",
        "input_ids",
        "text_embeddings",
    )
    logger.info(f"Done! Saved to: '{output_dir}'")


def main() -> None:
    default_cache_path = Path.home() / ".cache/open_clip_rs"
    parser = argparse.ArgumentParser(description="Export OpenCLIP models to ONNX.")
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'timm/ViT-SO400M-16-SigLIP2-384'). "
             "Must be open_clip compatible",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_cache_path,
        help=f"Base output directory (default: '{default_cache_path}')",
    )

    args = parser.parse_args()

    try:
        run_export(args.id, args.output)
    except Exception:
        logger.exception("Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
