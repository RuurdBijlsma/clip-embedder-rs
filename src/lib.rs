#![allow(clippy::missing_errors_doc)]
//! # `OpenCLIP` in Rust
//!
//! Easily run pre-trained [open_clip](https://github.com/mlfoundations/open_clip) compatible models in Rust via ONNX Runtime.
//!
//! ## Features
//!
//! - Run CLIP models in Rust via ONNX.
//! - Should support [any model compatible with `open_clip`](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending).
//! - Automatic model downloading: Just provide the Hugging Face model ID (has to point to `HuggingFace` repo with ONNX
//!   files & `open_clip_config.json`).
//!   - *Note*: This is enabled by default via the `hf-hub` feature. Disable it to remove `tokio` & `hf-hub` dependencies in case you don't need internet model loading.
//!
//! ## Usage
//!
//! Add `open_clip_inference` to your `Cargo.toml`.
//!
//! ### Option 1: Combined vision & text `Clip` API
//!
//! Use the `Clip` struct to perform classification or image ranking.
//!
//! ```rust
//! use open_clip_inference::Clip;
//! use std::path::Path;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
//! let mut clip = Clip::from_hf(model_id).build().await?;
//!
//! let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
//! let labels = &["cat", "dog"];
//!
//! let results = clip.classify(&img, labels)?;
//! for (label, prob) in results {
//!     println!("{}: {:.2}%", label, prob * 100.0);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Option 2: Individual text & vision embedders
//!
//! Use `VisionEmbedder` or `TextEmbedder` standalone to reduce memory usage if you only need one or the other.
//!
//! ```rust
//! use open_clip_inference::{VisionEmbedder, TextEmbedder, Clip};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
//! let mut vision = VisionEmbedder::from_local_id(model_id).build()?;
//! let mut text = TextEmbedder::from_local_id(model_id).build()?;
//!
//! let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
//! let img_emb = vision.embed_image(&img)?;
//! let text_embs = text.embed_texts(&["cat", "dog"])?;
//!
//! // Raw dot product
//! let similarities = text_embs.dot(&img_emb);
//!
//! // Apply model scale and bias
//! let scale = text.model_config.logit_scale.unwrap_or(1.0);
//! let bias = text.model_config.logit_bias.unwrap_or(0.0);
//! let logits: Vec<f32> = similarities.iter().map(|&s| s.mul_add(scale, bias)).collect();
//!
//! // Convert to probabilities
//! let probs = Clip::softmax(&logits);
//! # Ok(())
//! # }
//! ```
//!
//! ## Execution Providers (Nvidia, AMD, Intel, Mac, Arm, etc.)
//!
//! Since this is implemented with `ort`, many execution providers are available to enable hardware acceleration. You can
//! enable an execution provider in this crate with cargo features. A full list of execution providers is
//! available [here](https://ort.pyke.io/perf/execution-providers).
//!
//! To enable `cuda`, add the "cuda" feature,
//! and pass the CUDA execution provider when creating the embedder:
//!
//! ```rust
//! use open_clip_inference::Clip;
//! use ort::ep::{CUDA, CoreML, DirectML, TensorRT};
//! use std::path::Path;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
//! // Execution providers can be passed to the Clip, TextEmbedder, and VisionEmbedder builders.
//! // By default, an empty list is passed, which results in CPU inference.
//! // When multiple are passed, each execution provider is tried in order, if one doesn't work,
//! // the next one is tried, until falling back to CPU with no options left.
//! let mut clip = Clip::from_hf(model_id)
//!     .with_execution_providers(&[
//!         TensorRT::default().build(),
//!         CUDA::default().build(),
//!         DirectML::default().build(),
//!         CoreML::default().build(),
//!     ])
//!     .build()
//!     .await?;
//! #
//! #     let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
//! #     let texts = &[
//! #         "A photo of a cat",
//! #         "A photo of a dog",
//! #         "A photo of a beignet",
//! #     ];
//! #
//! #     let results = clip.classify(&img, texts)?;
//! #
//! #     for (text, prob) in results {
//! #         println!("{}: {:.2}", text, prob * 100.0);
//! #     }
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Model support
//!
//! This crate is implemented with [`ort`](https://crates.io/crates/ort), it runs ONNX models. I've uploaded the following
//! ONNX Clip Embedding models to HuggingFace:
//!
//! * [RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX](https://huggingface.co/RuteNL/ViT-SO400M-16-SigLIP2-384-ONNX)
//! * [RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX](https://huggingface.co/RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX)
//!
//! If you need a model that hasn't been converted to ONNX on HuggingFace yet, you can easily convert [any open_clip
//! compatible model](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending)
//! yourself, using `pull_onnx.py` from this repo.
//!
//! 1. Make sure you have [uv](https://docs.astral.sh/uv/).
//! 2. Run `pull_onnx.py --id timm/vit_base_patch32_clip_224.openai`
//! 3. After the Python script is done, you can the following in your Rust code:
//!
//! ```
//! let clip = Clip::from_local_id("timm/vit_base_patch32_clip_224.openai").build()?
//! ```
//!
//! I've tested the following models to work with `pull_onnx.py` & this crate:
//!
//! * [timm/MobileCLIP2-S4-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S4-OpenCLIP) *
//! * [timm/ViT-SO400M-16-SigLIP2-384](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384) *
//! * [timm/ViT-SO400M-14-SigLIP-384](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) *
//! * [timm/vit_base_patch32_clip_224.openai](https://huggingface.co/timm/vit_base_patch32_clip_224.openai) *
//! * [Marqo/marqo-fashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) *
//! * [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
//! * [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
//! * [imageomics/bioclip](https://huggingface.co/imageomics/bioclip)
//! * [timm/PE-Core-bigG-14-448](https://huggingface.co/timm/PE-Core-bigG-14-448)
//!
//! `*` Verified equal embedding outputs compared
//! to [reference Python implemenation](https://github.com/RuurdBijlsma/clip-model-research)
//!
//! See the `examples/` directory for detailed usage.

pub mod clip;
pub mod config;
pub mod error;
pub mod model_manager;
pub mod onnx;
pub mod text;
pub mod vision;

pub use clip::Clip;
pub use error::ClipError;
pub use text::TextEmbedder;
pub use vision::VisionEmbedder;
