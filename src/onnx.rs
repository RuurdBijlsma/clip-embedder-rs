use crate::ClipError;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use std::sync::RwLock;

pub struct OnnxSession {
    pub session: RwLock<Session>,
}

impl OnnxSession {
    pub fn new(
        path: impl AsRef<Path>,
        execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, ClipError> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)?;

        Ok(Self {
            session: RwLock::new(session),
        })
    }

    /// Helper to check if the model expects a specific input name
    ///
    /// # Panics
    /// Panics if the session lock is poisoned.
    #[must_use]
    pub fn has_input(&self, name: &str) -> bool {
        let session = self.session.read().unwrap();
        session.inputs().iter().any(|i| i.name() == name)
    }

    /// Helper to find the first likely input name for a specific role
    ///
    /// # Panics
    /// Panics if the session lock is poisoned.
    #[must_use]
    pub fn find_input(&self, possibilities: &[&str]) -> Option<String> {
        let session = self.session.read().unwrap();
        for &p in possibilities {
            if session.inputs().iter().any(|i| i.name() == p) {
                return Some(p.to_string());
            }
        }
        None
    }
}
