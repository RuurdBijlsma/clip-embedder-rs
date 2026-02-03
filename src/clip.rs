use crate::config::ModelConfig;
use crate::error::ClipError;
use crate::text::TextEmbedder;
use crate::vision::VisionEmbedder;
use bon::bon;
use image::DynamicImage;
use ort::ep::ExecutionProviderDispatch;
use std::path::Path;

/// A convenience wrapper that holds both a `VisionEmbedder` and a `TextEmbedder`.
pub struct Clip {
    pub vision: VisionEmbedder,
    pub text: TextEmbedder,
}

#[bon]
impl Clip {
    /// Load both Vision and Text embedders from a model ID
    ///
    /// todo: we need new constructor method, something like from_preconverted(, and from_local_converted? idk sounds dumb
    #[builder(finish_fn = build)]
    pub fn from_model_id(
        #[builder(start_fn)] model_id: &str,
        with_execution_providers: Option<&[ExecutionProviderDispatch]>,
    ) -> Result<Self, ClipError> {
        let model_dir = crate::download::ensure_model(model_id)?;
        dbg!(&model_dir);
        Self::from_model_dir(&model_dir)
            .maybe_with_execution_providers(with_execution_providers)
            .build()
    }

    /// Load both Vision and Text embedders from a specific directory
    #[builder(finish_fn = build)]
    pub fn from_model_dir(
        #[builder(start_fn)] model_dir: &Path,
        with_execution_providers: Option<&[ExecutionProviderDispatch]>,
    ) -> Result<Self, ClipError> {
        let vision = VisionEmbedder::from_model_dir(model_dir)
            .maybe_with_execution_providers(with_execution_providers)
            .build()?;
        let text = TextEmbedder::from_model_dir(model_dir)
            .maybe_with_execution_providers(with_execution_providers)
            .build()?;
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
            .zip(probs)
            .map(|(l, p)| (l.as_ref().to_string(), p))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Rank a batch of images against a single text query.
    /// Returns a list of (`image_index`, `probability`) pairs sorted by highest probability.
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

    /// Compute softmax probabilities for an array of logits
    #[must_use]
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Compute sigmoid probabilities for a single logit
    #[must_use]
    pub fn sigmoid(logit: f32) -> f32 {
        1.0 / (1.0 + (-logit).exp())
    }
}
