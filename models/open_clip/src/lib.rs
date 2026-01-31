pub mod config;
pub mod onnx;
pub mod text;
pub mod vision;

use color_eyre::eyre::{Context, Result};
use config::{ModelConfig, ModelType};
use image::DynamicImage;
use ndarray::{Array2, Axis};
use onnx::OnnxRunner;
use std::path::Path;
use text::TextProcessor;
use vision::VisionProcessor;

pub struct ClipEmbedder {
    config: ModelConfig,
    vision: VisionProcessor,
    text: TextProcessor,
    vision_ort: OnnxRunner,
    text_ort: OnnxRunner,
}

impl ClipEmbedder {
    /// Load from directory containing `model_config.json`, `tokenizer.json`, etc.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();

        // 1. Load Clean Config
        let config_path = dir.join("model_config.json");
        let config = ModelConfig::from_file(&config_path)
            .wrap_err_with(|| format!("Missing model_config.json at {:?}", config_path))?;

        let tokens_map_path = dir.join("special_tokens_map.json");
        let tokenizer_path = dir.join("tokenizer.json");

        // 2. Init Processors
        let vision = VisionProcessor::new(&config);
        let text = TextProcessor::new(tokenizer_path, tokens_map_path, &config)?;

        // 3. Init ORT
        let vision_ort = OnnxRunner::new(dir.join("visual.onnx"))?;
        let text_ort = OnnxRunner::new(dir.join("text.onnx"))?;

        Ok(Self {
            config,
            vision,
            text,
            vision_ort,
            text_ort,
        })
    }

    // --- EMBEDDING (Same as before) ---

    pub fn embed_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let input = self.vision.process(image)?;
        Ok(self.vision_ort.run_vision(input)?.into_raw_vec())
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let (ids, mask) = self.text.process(text)?;
        Ok(self.text_ort.run_text(ids, mask)?.into_raw_vec())
    }

    pub fn embed_images(&self, images: &[DynamicImage]) -> Result<Array2<f32>> {
        let input = self.vision.process_batch(images)?;
        self.vision_ort.run_vision(input)
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Array2<f32>> {
        let (ids, mask) = self.text.process_batch(texts)?;
        self.text_ort.run_text(ids, mask)
    }

    // --- VERSATILE SCORING ---

    /// Calculates Raw Logits (dot_product * scale + bias).
    /// Returns a similarity matrix (M images x N texts).
    ///
    /// Use this if you want to apply your own Softmax or analysis.
    pub fn compute_logits(&self, img_embs: &Array2<f32>, text_embs: &Array2<f32>) -> Array2<f32> {
        // Matrix Multiplication: (M x D) dot (N x D)^T -> (M x N)
        // We transpose text_embs to align dimensions
        let dots = img_embs.dot(&text_embs.t());

        // Apply Scale & Bias
        // formula: dot * scale + bias
        dots.mapv(|x| x * self.config.logit_scale + self.config.logit_bias)
    }

    /// Calculates Probabilities (0.0 - 1.0).
    ///
    /// - **SigLIP**: Applies Sigmoid to every element independently.
    /// - **CLIP**: Applies Softmax.
    ///   *Note*: Standard CLIP Softmax is usually applied row-wise (Image vs All Candidate Texts).
    ///   If you are doing Image-to-Image or single comparisons, Softmax might not be what you want.
    ///   This function assumes the standard use case: "Classify this image against these texts".
    pub fn compute_probs(&self, img_embs: &Array2<f32>, text_embs: &Array2<f32>) -> Array2<f32> {
        let logits = self.compute_logits(img_embs, text_embs);

        match self.config.model_type {
            ModelType::Siglip => {
                // SigLIP: Independent Sigmoid for every pair
                logits.mapv(sigmoid)
            }
            ModelType::Clip => {
                // CLIP: Softmax across the "Text" dimension (columns)
                // For each image (row), which text is it?
                let mut probs = logits.clone();
                for mut row in probs.axis_iter_mut(Axis(0)) {
                    softmax_inplace(&mut row);
                }
                probs
            }
        }
    }
}

// --- MATH HELPERS ---

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Applies Softmax to a 1D array view in place
fn softmax_inplace(x: &mut ndarray::ArrayViewMut1<f32>) {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}