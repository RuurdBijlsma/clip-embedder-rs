use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClipError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("ONNX error: {0}")]
    Onnx(#[from] ort::Error),
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Tokenization error: {0}")]
    Tokenizer(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, ClipError>;
