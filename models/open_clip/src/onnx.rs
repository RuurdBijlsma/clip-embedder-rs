use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use crate::error::Result;

pub struct OnnxSession {
    pub session: Session,
}

impl OnnxSession {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Helper to check if the model expects a specific input name
    pub fn has_input(&self, name: &str) -> bool {
        self.session.inputs().iter().any(|i| i.name() == name)
    }

    /// Helper to find the first likely input name for a specific role
    pub fn find_input(&self, possibilities: &[&str]) -> Option<String> {
        for &p in possibilities {
            if self.has_input(p) { return Some(p.to_string()); }
        }
        None
    }
}