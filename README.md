# OpenCLIP embedding in Rust

Easily run pre-trained [open_clip](https://github.com/mlfoundations/open_clip) compatible embedding models in Rust via
ONNX Runtime.

## Features

- Run CLIP models in Rust via ONNX.
- Should support [any model compatible with
  `open_clip`](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending) (
  Python).
- Automatic model downloading: Just provide the Hugging Face model ID (has to point to HuggingFace repo with ONNX
  files & `open_clip_config.json`).
- Python is only needed if you want to convert new models yourself.

## Prerequisites

1. [Rust & Cargo](https://rust-lang.org/).
2. (Optional) [uv](https://docs.astral.sh/uv/) - Only if you want to convert models from HuggingFace to ONNX.
3. (Optional) If you have to link dynamically (on Windows) - [onnxruntime](https://github.com/microsoft/onnxruntime).

## Usage: Embedding text & image

### Option 1: `Clip` struct

The `Clip` struct is built for ease of use, handling both vision and text together, with convenience functions for
similarity rankings.

```rust
use open_clip_inference::Clip;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `model_id` from Hugging Face -> This `model_id` is a pre-converted model.
    // Use `from_local_id` or `from_local_dir` to supply a locally stored model, not on HuggingFace.
    let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
    let mut clip = Clip::from_hf(model_id).build().await?;

    let img = image::open(Path::new("assets/img/cat_face.jpg"))?;
    let labels = &["cat", "dog", "beignet"];

    let results = clip.classify(&img, labels)?;

    for (label, prob) in results {
        println!("{}: {:.2}%", label, prob * 100.0);
    }

    Ok(())
}
```

Input image: `Poekie`

<img alt="Poeker" src="assets/img/cat_face.jpg" width="150" title="Poekie">

Outputs:

```
A photo of a cat: 99.99%
A photo of a dog: 0.01%
A photo of a beignet: 0.00%
```

### Option 2: Individual vision & text embedders

Use `VisionEmbedder` or `TextEmbedder` standalone to just produce embeddings from images & text.

```rust
use open_clip_inference::{VisionEmbedder, TextEmbedder, Clip};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "timm/ViT-SO400M-16-SigLIP2-384";
    let mut vision = VisionEmbedder::from_model_id(model_id).build()?;
    let mut text = TextEmbedder::from_model_id(model_id).build()?;

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

## Model support

This crate is implemented with [`ort`](https://crates.io/crates/ort), it runs ONNX models. I've uploaded the following
ONNX Clip Embedding models to HuggingFace:

* [RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX](https://huggingface.co/RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX)
* [RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX](https://huggingface.co/RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX)

If you need a model that hasn't been converted to ONNX on HuggingFace yet, you can easily convert [any open_clip
compatible model](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending)
yourself, using `pull_onnx.py` from this repo.

1. Make sure you have [uv](https://docs.astral.sh/uv/).
2. Run `pull_onnx.py --id timm/vit_base_patch32_clip_224.openai`
3. After the Python script is done, you can the following in your Rust code:

```
let clip = Clip::from_local_id("timm/vit_base_patch32_clip_224.openai").build()?
```

I've tested the following models to work with `pull_onnx.py` & this crate:

* [timm/MobileCLIP2-S4-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S4-OpenCLIP) *
* [timm/ViT-SO400M-16-SigLIP2-384](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384) *
* [timm/ViT-SO400M-14-SigLIP-384](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) *
* [timm/vit_base_patch32_clip_224.openai](https://huggingface.co/timm/vit_base_patch32_clip_224.openai) *
* [Marqo/marqo-fashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) *
* [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
* [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
* [imageomics/bioclip](https://huggingface.co/imageomics/bioclip)
* [timm/PE-Core-bigG-14-448](https://huggingface.co/timm/PE-Core-bigG-14-448)

`*` Verified equal embedding outputs compared
to [reference Python implemenation](https://github.com/RuurdBijlsma/clip-model-research)

## Execution Providers (Nvidia, AMD, Intel, Mac, Arm, etc.)

Since this is implemented with `ort`, many execution providers are available to enable hardware acceleration. You can
enable an execution provider in this crate with cargo features. A full list of execution providers is
available [here](https://ort.pyke.io/perf/execution-providers). 

To enable `cuda`, add the "cuda" feature,
and pass the CUDA execution provider when creating the embedder:

```rust
use open_clip_inference::Clip;
use ort::ep::{CUDA, CoreML, DirectML, TensorRT};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
    // Execution providers can be passed to the Clip, TextEmbedder, and VisionEmbedder constructor builders.
    // By default, an empty list is passed, which results in CPU inference.
    // When multiple are passed, each execution provider is tried in order, if one doesn't work, the next one is tried, 
    // until falling back to CPU with no options left.
    let mut clip = Clip::from_hf(model_id)
        .with_execution_providers(&[
            TensorRT::default().build(),
            CUDA::default().build(),
            DirectML::default().build(),
            CoreML::default().build(),
        ])
        .build()
        .await?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let results = clip.classify(&img, texts)?;

    for (text, prob) in results {
        println!("{}: {:.2}", text, prob * 100.0);
    }

    Ok(())
}
```

## Troubleshooting

### If it doesn't build on Windows due to onnxruntime problems

Try using the feature `load-dynamic` and point to the onnxruntime dll as described below.

### [When using `load-dynamic` feature] ONNX Runtime Library Not Found

OnnxRuntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
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
