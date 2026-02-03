use crate::error::ClipError;
#[cfg(feature = "hf-hub")]
use hf_hub::api::tokio::Api;
use std::env;
use std::path::{Path, PathBuf};

/// Files to download from the Hugging Face repository.
pub const MODEL_FILES: &[&str] = &[
    "model_config.json",
    "open_clip_config.json",
    "special_tokens_map.json",
    "text.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "visual.onnx",
    "text.onnx.data",
    "visual.onnx.data",
];

/// Ensures that the model files are present locally.
#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model_id: &str) -> Result<PathBuf, ClipError> {
    // Try Hugging Face Hub
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let mut model_dir = None;
    for file in MODEL_FILES {
        let downloaded_file = repo.get(file).await?;
        if model_dir.is_none() {
            model_dir = downloaded_file.parent().map(ToOwned::to_owned);
        }
    }

    model_dir.ok_or_else(|| {
        ClipError::HfHub(format!(
            "Could not determine model directory for '{model_id}'"
        ))
    })
}

/// Get default model base folder (where `pull_onnx.py` also exports to by default).
#[must_use]
pub fn get_default_base_folder() -> PathBuf {
    env::home_dir().map_or_else(
        || Path::new(".open_clip_cache").to_owned(),
        |p| p.join(".cache/open_clip_rs"),
    )
}

/// Verify that a model directory is valid, and contains the right files.
pub fn verify_model_dir(model_dir: &Path) -> Result<(), ClipError> {
    if !model_dir.exists() {
        return Err(ClipError::ModelFolderNotFound(model_dir.to_owned()));
    }

    for file in MODEL_FILES {
        let path = model_dir.join(file);
        if !path.is_file() {
            return Err(ClipError::MissingModelFile {
                model_dir: model_dir.to_owned(),
                file: file.to_string(),
            });
        }
    }

    Ok(())
}
