use crate::error::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LocalConfig {
    #[serde(default)]
    pub tokenizer_needs_lowercase: bool,
    pub activation_function: Option<String>,
    pub logit_scale: Option<f32>,
    pub logit_bias: Option<f32>,
    pub pad_id: Option<u32>,
}

impl LocalConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenClipConfig {
    pub model_cfg: ModelCfg,
    pub preprocess_cfg: PreprocessCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCfg {
    pub embed_dim: usize,
    pub vision_cfg: VisionCfg,
    pub text_cfg: TextCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionCfg {
    pub image_size: u32,
    pub layers: Option<usize>,
    pub width: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextCfg {
    pub context_length: usize,
    pub vocab_size: usize,
    pub hf_tokenizer_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessCfg {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub interpolation: String,
    pub resize_mode: String,
}

impl OpenClipConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}
