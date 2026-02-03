use crate::error::ClipError;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Files to download from the Hugging Face repository.
const MODEL_FILES: &[&str] = &[
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
///
/// todo: this should work with custom base folders
pub fn ensure_model(model_id: &str) -> Result<PathBuf, ClipError> {
    // Check local cache
    let local_dir = crate::onnx::OnnxSession::get_model_dir(model_id);
    if local_dir.exists() && local_dir.join("model_config.json").exists() {
        return Ok(local_dir);
    }

    // Try Hugging Face Hub
    let api = Api::new().map_err(|e| ClipError::HfHub(e.to_string()))?;
    let repo = api.model(model_id.to_string());

    let mut model_dir = None;

    for file in MODEL_FILES {
        match repo.get(file) {
            Ok(path) => {
                if model_dir.is_none() {
                    let parent = path.parent().ok_or_else(|| {
                        ClipError::HfHub(format!("File '{file}' has no parent directory"))
                    })?;
                    model_dir = Some(parent.to_path_buf());
                }
            }
            Err(e) => {
                return Err(ClipError::HfHub(format!(
                    "Failed to download required file '{file}': {e}"
                )));
            }
        }
    }

    model_dir.ok_or_else(|| {
        ClipError::HfHub(format!(
            "Could not determine model directory for '{model_id}'"
        ))
    })
}
