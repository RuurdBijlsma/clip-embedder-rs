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
    pub fn has_input(&self, name: &str) -> Result<bool, ClipError> {
        let session = self.session.read()?;
        Ok(session.inputs().iter().any(|i| i.name() == name))
    }

    /// Helper to find the first likely input name for a specific role
    pub fn find_input(&self, possibilities: &[&str]) -> Result<Option<String>, ClipError> {
        let session = self.session.read()?;
        for &p in possibilities {
            if session.inputs().iter().any(|i| i.name() == p) {
                return Ok(Some(p.to_string()));
            }
        }
        Ok(None)
    }
}
