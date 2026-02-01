#![allow(clippy::missing_errors_doc)]
pub mod config;
pub mod error;
pub mod onnx;
pub mod text;
pub mod vision;

pub use error::{ClipError};
pub use text::TextEmbedder;
pub use vision::VisionEmbedder;
