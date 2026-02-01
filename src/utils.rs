use std::env;
use std::path::{Path, PathBuf};

/// Get the directory where model files are stored for a given model ID.
#[must_use]
pub fn get_model_dir(model_id: &str) -> PathBuf {
    let base_folder = env::home_dir().map_or_else(
        || Path::new(".open_clip_cache").to_owned(),
        |p| p.join(".cache/open_clip_rs"),
    );
    base_folder.join(model_id)
}
