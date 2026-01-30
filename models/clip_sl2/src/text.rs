use crate::{ClipError, ModelConfig};
use ndarray::{Array2, ArrayView, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub struct SigLipTextModel {
    session: Session,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: ModelConfig,
}

impl SigLipTextModel {
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<Self, ClipError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        // Configure padding/truncation based on model config
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(config.context_length),
            pad_id: 0,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.context_length,
                ..Default::default()
            }))
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            config,
        })
    }

    /// Tokenizes text into input IDs.
    /// Returns (1, ContextLen)
    pub fn tokenize(&self, text: &str) -> Result<Array2<i64>, ClipError> {
        let encoding = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        Ok(Array2::from_shape_vec((1, ids.len()), ids)?)
    }

    /// Batch inference. Input shape: (Batch, ContextLen)
    pub fn inference(&mut self, ids_array: Array2<i64>) -> Result<Array2<f32>, ClipError> {
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Value::from_array(ids_array)?
        ])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }

    /// Helper to get raw IDs as u32 (useful for debugging)
    pub fn get_ids(&self, text: &str) -> Result<Vec<u32>, ClipError> {
        let encoding = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}