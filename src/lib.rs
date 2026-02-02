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
//! 3. [onnxruntime](https://github.com/microsoft/onnxruntime) - It's linked dynamically.
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
//! ```rust,no_run
//! use open_clip_inference::{VisionEmbedder, TextEmbedder};
//! use image::ImageReader;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
//! let mut vision_embedder = VisionEmbedder::from_model_id(model_id)?;
//! let mut text_embedder = TextEmbedder::from_model_id(model_id)?;
//!
//! // In a real app, load an image from path
//! // let img = image::open(Path::new("assets/img/cat_face.jpg"))?;
//! // For this doc test, we'll create a dummy image
//! let img = image::DynamicImage::new_rgb8(224, 224);
//!
//! let texts = &[
//!     "A photo of a cat",
//!     "A photo of a dog",
//! ];
//!
//! let img_emb = vision_embedder.embed_image(&img)?;
//! let text_embs = text_embedder.embed_texts(texts)?;
//!
//! let similarities = text_embs.dot(&img_emb);
//! println!("Similarities: {:?}", similarities);
//! # Ok(())
//! # }
//! ```
//!
//! See the `examples/` directory for detailed usage.

pub mod config;
pub mod error;
pub mod onnx;
pub mod text;
pub mod vision;
pub mod clip;

pub use error::ClipError;
pub use text::TextEmbedder;
pub use vision::VisionEmbedder;
pub use clip::Clip;
