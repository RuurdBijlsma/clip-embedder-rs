use crate::config::ModelConfig;
use crate::error::ClipError;
use crate::text::TextEmbedder;
use crate::vision::VisionEmbedder;
use image::DynamicImage;
use std::path::Path;

/// A convenience wrapper that holds both a `VisionEmbedder` and a `TextEmbedder`.
pub struct Clip {
    pub vision: VisionEmbedder,
    pub text: TextEmbedder,
}

impl Clip {
    /// Load both Vision and Text embedders from a model ID in the default cache location.
    pub fn from_model_id(model_id: &str) -> Result<Self, ClipError> {
        let vision = VisionEmbedder::from_model_id(model_id)?;
        let text = TextEmbedder::from_model_id(model_id)?;
        Ok(Self { vision, text })
    }

    /// Load both Vision and Text embedders from a specific directory.
    pub fn new(model_dir: &Path) -> Result<Self, ClipError> {
        let vision = VisionEmbedder::new(model_dir)?;
        let text = TextEmbedder::new(model_dir)?;
        Ok(Self { vision, text })
    }

    pub fn get_model_config(&self) -> ModelConfig {
        self.text.model_config.clone()
    }

    /// Compare an image to a piece of text and return the raw logit.
    /// This handles embedding, dot-product, and applying logit scale/bias.
    pub fn compare(&mut self, image: &DynamicImage, text: &str) -> Result<f32, ClipError> {
        let vision_emb = self.vision.embed_image(image)?;
        let text_emb = self.text.embed_text(text)?;

        let sim = vision_emb.dot(&text_emb);
        let scale = self.text.model_config.logit_scale.unwrap_or(1.0);
        let bias = self.text.model_config.logit_bias.unwrap_or(0.0);

        Ok(sim.mul_add(scale, bias))
    }

    /// Classify an image against a list of text labels.
    /// Returns a list of (label, probability) pairs sorted by highest probability.
    pub fn classify<T: AsRef<str>>(
        &mut self,
        image: &DynamicImage,
        labels: &[T],
    ) -> Result<Vec<(String, f32)>, ClipError> {
        let vision_emb = self.vision.embed_image(image)?;
        let text_embs = self.text.embed_texts(labels)?;

        let similarities = text_embs.dot(&vision_emb);
        let scale = self.text.model_config.logit_scale.unwrap_or(1.0);
        let bias = self.text.model_config.logit_bias.unwrap_or(0.0);

        let logits: Vec<f32> = similarities
            .iter()
            .map(|&s| s.mul_add(scale, bias))
            .collect();
        let activation = self
            .text
            .model_config
            .activation_function
            .as_deref()
            .unwrap_or("softmax");

        let probs = if activation == "sigmoid" {
            logits.iter().map(|&l| Self::sigmoid(l)).collect()
        } else {
            Self::softmax(&logits)
        };

        let mut results: Vec<(String, f32)> = labels
            .iter()
            .zip(probs.into_iter())
            .map(|(l, p)| (l.as_ref().to_string(), p))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Rank a batch of images against a single text query.
    /// Returns a list of (image_index, probability) pairs sorted by highest probability.
    pub fn rank_images(
        &mut self,
        images: &[DynamicImage],
        text: &str,
    ) -> Result<Vec<(usize, f32)>, ClipError> {
        let img_embs = self.vision.embed_images(images)?;
        let text_emb = self.text.embed_text(text)?;

        let similarities = img_embs.dot(&text_emb);
        let scale = self.text.model_config.logit_scale.unwrap_or(1.0);
        let bias = self.text.model_config.logit_bias.unwrap_or(0.0);

        let logits: Vec<f32> = similarities
            .iter()
            .map(|&s| s.mul_add(scale, bias))
            .collect();
        let activation = self
            .text
            .model_config
            .activation_function
            .as_deref()
            .unwrap_or("softmax");

        let probs = if activation == "sigmoid" {
            logits.iter().map(|&l| Self::sigmoid(l)).collect()
        } else {
            Self::softmax(&logits)
        };

        let mut results: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Compute softmax probabilities for an array of logits.
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Compute sigmoid probabilities for a single logit.
    pub fn sigmoid(logit: f32) -> f32 {
        1.0 / (1.0 + (-logit).exp())
    }
}
