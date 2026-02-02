# OpenCLIP embedding in Rust

Easily run pre-trained [open_clip](https://github.com/mlfoundations/open_clip) compatible embedding models in Rust via
ONNX Runtime.

## Features

- Run CLIP models in Rust via ONNX.
- Should support [any model compatible with
  `open_clip`](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending) (
  Python).
- Python is only needed once to download and export the model weights.

## Prerequisites

1. [Rust & Cargo](https://rust-lang.org/).
2. [uv](https://docs.astral.sh/uv/) - to generate ONNX files from HuggingFace models.
3. [onnxruntime](https://github.com/microsoft/onnxruntime) - It's linked dynamically as I had issues with static
   linking.

## Usage: Export Model to ONNX

Use the provided `pull_onnx.py` script to download and export an OpenCLIP model from Hugging Face.

```shell
# Run the export script - uv will handle the dependencies
# Example: Export mobileclip 2
uv run pull_onnx.py --id "timm/MobileCLIP2-S2-OpenCLIP"
```

## Usage: Inference in Rust

### Option 1: `Clip` struct

The `Clip` struct is built for ease of use, handling both vision and text together, with convenience functions for
similarity rankings.

```rust
use open_clip_inference::Clip;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
    let mut clip = Clip::from_model_id(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg"))?;
    let labels = &["cat", "dog", "beignet"];

    let results = clip.classify(&img, labels)?;

    for (label, prob) in results {
        println!("{}: {:.2}%", label, prob * 100.0);
    }

    Ok(())
}
```

### Option 2: Individual vision & text embedders

Use `VisionEmbedder` or `TextEmbedder` standalone to just produce embeddings from images & text.

```rust
use open_clip_inference::{VisionEmbedder, TextEmbedder, Clip};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
    let mut vision = VisionEmbedder::from_model_id(model_id)?;
    let mut text = TextEmbedder::from_model_id(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg"))?;
    let img_emb = vision.embed_image(&img)?;
    // Now you may put the embeddings in a database like Postgres with PgVector to set up semantic image search.

    let text_embs = text.embed_text("a cat")?;
    // You can search with the text embedding through images using cosine similarity.
    // All embeddings produced are already l2 normalized.

    Ok(())
}
```

## Examples

Run the included examples (ensure you have exported the relevant model first):

```shell
# Simple generic example
cargo run --example basic

# Semantic image search demo
cargo run --example search
```

## Tested Models

The following models have been tested to work with `pull_onnx.py` & this Rust crate. I picked these models as they are
highest performing in benchmarks or most popular on HuggingFace.

* `timm/MobileCLIP2-S4-OpenCLIP`
* `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
* `timm/ViT-SO400M-16-SigLIP2-384`
* `timm/vit_base_patch32_clip_224.openai`
* `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
* `imageomics/bioclip`
* `Marqo/marqo-fashionSigLIP`
* `timm/PE-Core-bigG-14-448`
* `timm/ViT-SO400M-14-SigLIP-384`

### Verified Embeddings

The following models have been verified to produce embeddings in Rust that match the Python reference implementation:

Python implementations here: https://github.com/RuurdBijlsma/clip-model-research

* `timm/ViT-SO400M-16-SigLIP2-384`
* `timm/MobileCLIP2-S4-OpenCLIP`
* `timm/vit_base_patch32_clip_224.openai`
* `timm/ViT-SO400M-14-SigLIP-384`
* `Marqo/marqo-fashionSigLIP`

## Troubleshooting

### If it doesn't build on Windows due to onnxruntime problems

Try using the feature `load-dynamic` and point to the onnxruntime dll as described below.

### [When using `load-dynamic` feature] ONNX Runtime Library Not Found

Onnxruntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
from [GitHub Releases](http://github.com/microsoft/onnxruntime/releases).

Then put the dll/so/dylib location in your `PATH`, or point the `ORT_DYLIB_PATH` env var to it.

**PowerShell example:**

* Adjust path to where the dll is.

```powershell
$env:ORT_DYLIB_PATH="C:/Apps/onnxruntime/lib/onnxruntime.dll"
```

**Shell example:**

```shell
export ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so"
```
