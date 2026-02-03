use hf_hub::api::tokio::ApiError;
use std::path::PathBuf;
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
    #[error("Model folder not found, generate it with `uv run pull_onnx.py -h`. '{0}'")]
    ModelFolderNotFound(PathBuf),
    #[error("Hugging Face Hub error: {0}")]
    HfHub(String),
    #[error("Missing model file '{file}' in folder '{model_dir}'")]
    MissingModelFile { model_dir: PathBuf, file: String },
}

impl From<ApiError> for ClipError {
    fn from(value: ApiError) -> Self {
        Self::HfHub(value.to_string())
    }
}
