use serde::Deserialize;
use std::path::Path;
use std::{fs, io};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Siglip,
    Clip,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    // Identity
    pub model_type: ModelType,

    // Dimensions
    pub embed_dim: usize,
    pub image_size: u32,
    pub context_length: usize,
    pub vocab_size: usize,

    // Math Parameters
    // We expect these to be the *final* multipliers (e.g. already exp() if needed)
    pub logit_scale: f32,
    #[serde(default)]
    pub logit_bias: f32, // Defaults to 0.0 if missing (good for CLIP)

    // Preprocessing
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub interpolation: String,
    pub resize_mode: String,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let content = fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }
}

// -----------------------------------------------------------------------------
// Keep SpecialTokensMap (it works well)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SpecialTokensMap {
    pub pad_token: Option<SpecialToken>,
    pub eos_token: Option<SpecialToken>,
    pub bos_token: Option<SpecialToken>,
    pub unk_token: Option<SpecialToken>,
}

impl SpecialTokensMap {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let content = fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "SpecialTokenRaw")]
pub struct SpecialToken {
    pub content: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SpecialTokenRaw {
    Simple(String),
    Complex { content: String },
}

impl From<SpecialTokenRaw> for SpecialToken {
    fn from(raw: SpecialTokenRaw) -> Self {
        match raw {
            SpecialTokenRaw::Simple(s) => SpecialToken { content: s },
            SpecialTokenRaw::Complex { content } => SpecialToken { content },
        }
    }
}