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

#### From `examples/basic.rs`

First, add `open_clip_inference` to your `Cargo.toml`.

```rust
use open_clip_inference::{VisionEmbedder, TextEmbedder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
    let mut vision_embedder = VisionEmbedder::from_model_id(model_id)?;
    let mut text_embedder = TextEmbedder::from_model_id(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg"))?;
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let img_emb = vision_embedder.embed_image(&img)?;
    let text_embs = text_embedder.embed_texts(texts)?;

    let similarities = text_embs.dot(&img_emb);

    // compute softmax on output similarities
    let scale = text_embedder.model_config.logit_scale.unwrap_or(1.0);
    let bias = text_embedder.model_config.logit_bias.unwrap_or(0.0);
    let logits: Vec<f32> = similarities
        .iter()
        .map(|&s| s.mul_add(scale, bias))
        .collect();

    for (text, prob) in texts.iter().zip(softmax(&logits)) {
        println!("{}: {:.2}%", text, prob * 100.0);
    }

    Ok(())
}
```

### Output (cosine similarity scores)

These values are pre-softmax similarity logits. They are not probabilities and appear less confident.

```
A photo of a cat: 0.38
A photo of a dog: 0.32
A photo of a beignet: 0.30
```

After applying softmax to the output, it looks better:

```
A photo of a cat: 99.99%
A photo of a dog: 0.01%
A photo of a beignet: 0.00%
```

## Examples

Run the included examples (ensure you have exported the relevant model first, usually `timm/ViT-SO400M-16-SigLIP2-384`
for the examples):

```shell
# Simple generic example
cargo run --example basic

# Semantic image search demo
cargo run --example search
```

## Tested Models

The following models have been tested to work with `pull_onnx.py` & this Rust crate. I picked these models as they are
highest performing in benchmarks or most popular on HuggingFace.

* `imageomics/bioclip`
* `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
* `Marqo/marqo-fashionSigLIP`
* `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
* `timm/MobileCLIP2-S4-OpenCLIP`
* `timm/PE-Core-bigG-14-448`
* `timm/ViT-SO400M-14-SigLIP-384`
* `timm/ViT-SO400M-16-SigLIP2-384`
* `timm/vit_base_patch32_clip_224.openai`

### Verified Embeddings

The following models have been verified to produce embeddings in Rust that match the Python reference implementation:

Python implementations here: https://github.com/RuurdBijlsma/clip-model-research

* `Marqo/marqo-fashionSigLIP`
* `timm/MobileCLIP2-S4-OpenCLIP`
* `timm/ViT-SO400M-14-SigLIP-384`
* `timm/ViT-SO400M-16-SigLIP2-384`
* `timm/vit_base_patch32_clip_224.openai`

## Troubleshooting

### ONNX Runtime Library Not Found

Onnxruntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
from [GitHub Releases](http://github.com/microsoft/onnxruntime/releases).

Then put the dll/so/dylib location in your `PATH`, or point the `ORT_DYLIB_PATH` env var to it.

**PowerShell example:**

* Adjust path to where the dll is.

```powershell
$env:ORT_DYLIB_PATH = "C:/Apps/onnxruntime/lib/onnxruntime.dll"
```

**Shell example:**

```shell
export ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so"
```
