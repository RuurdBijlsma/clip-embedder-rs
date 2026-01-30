use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClipError {
    #[error("Inference engine error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Shape/Tensor error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config parsing error: {0}")]
    Config(#[from] serde_json::Error),
}