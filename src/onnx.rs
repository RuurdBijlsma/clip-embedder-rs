use crate::ClipError;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::env;
use std::path::{Path, PathBuf};

pub struct OnnxSession {
    pub session: Session,
}

impl OnnxSession {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ClipError> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Helper to check if the model expects a specific input name
    #[must_use]
    pub fn has_input(&self, name: &str) -> bool {
        self.session.inputs().iter().any(|i| i.name() == name)
    }

    /// Helper to find the first likely input name for a specific role
    #[must_use]
    pub fn find_input(&self, possibilities: &[&str]) -> Option<String> {
        for &p in possibilities {
            if self.has_input(p) {
                return Some(p.to_string());
            }
        }
        None
    }

    /// Get model directory by `model_id`
    #[must_use]
    pub fn get_model_dir(model_id: &str) -> PathBuf {
        let base_folder = env::home_dir().map_or_else(
            || Path::new(".open_clip_cache").to_owned(),
            |p| p.join(".cache/open_clip_rs"),
        );
        base_folder.join(model_id)
    }
}
