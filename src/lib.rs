#![allow(clippy::missing_errors_doc)]
//! # `OpenCLIP` in Rust
//!
//! Easily run pre-trained [open_clip](https://github.com/mlfoundations/open_clip) compatible models in Rust via ONNX Runtime.
//!
//! ## Features
//!
//! - Run CLIP models in Rust via ONNX.
//! - Should support [any model compatible with `open_clip`](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending) (Python).
//! - Python is only needed once to download and export the model weights.
//!
//! ## Prerequisites
//!
//! 1. [Rust & Cargo](https://rust-lang.org/).
//! 2. [uv](https://docs.astral.sh/uv/) - to generate ONNX files from `HuggingFace` models.
//! 3. [onnxruntime](https://github.com/microsoft/onnxruntime) - Linked dynamically.
//!
//! ## Usage
//!
//! ### Step 1: Export Model to ONNX
//!
//! Use the provided `pull_onnx.py` script to download and export an `OpenCLIP` model from Hugging Face.
//!
//! ```shell
//! # Run the export script - uv will handle the dependencies
//! # Example: Export mobileclip 2
//! uv run pull_onnx.py --id "timm/MobileCLIP2-S2-OpenCLIP"
//! ```
//!
//! ### Step 2: Inference in Rust
//!
//! Add `open_clip` to your `Cargo.toml`.
//!
//! ### Option 1: High-Level API (Convenience)
//!
//! Use the `Clip` struct to perform classification or image ranking.
//!
//! ```rust
//! use open_clip_inference::Clip;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
//! let mut clip = Clip::from_hf(model_id).build()?;
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
//! Use `VisionEmbedder` or `TextEmbedder` standalone for custom workflows.
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
