#![allow(clippy::missing_errors_doc)]
//! # `OpenCLIP` in Rust
//!
//! Easily run pre-trained [open_clip](https://github.com/mlfoundations/open_clip) compatible models in Rust via ONNX Runtime.
//!
//! ## Features
//!
//! - Run CLIP models in Rust via ONNX.
//! - Should support [any model compatible with `open_clip`](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&library=open_clip&sort=trending).
//! - Automatic model downloading: Just provide the Hugging Face model ID (has to point to HuggingFace repo with ONNX
//!   files & `open_clip_config.json`).
//!   - *Note*: This is enabled by default via the `hf-hub` feature. Disable it to remove `tokio` & `hf-hub` dependencies if you only need local model loading.
//!
//! ## Usage
//!
//! ### Embedding in Rust
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
