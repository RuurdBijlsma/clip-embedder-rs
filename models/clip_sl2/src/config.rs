use crate::ClipError;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub logit_scale: f32,
    pub logit_bias: f32,
    pub image_size: u32,
    pub context_length: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ModelConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ClipError> {
        let content = fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }
}