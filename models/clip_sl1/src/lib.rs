#![allow(clippy::missing_errors_doc)]

use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

#[derive(thiserror::Error, Debug)]
pub enum SigLipError {
    #[error("Inference engine error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

pub struct SigLipVisionModel {
    pub session: Session,
    pub image_size: u32,
}

impl SigLipVisionModel {
    pub fn new(visual_path: impl AsRef<Path>, image_size: u32) -> Result<Self, SigLipError> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(visual_path)?;

        Ok(Self {
            session,
            image_size,
        })
    }

    /// Exposes the preprocessing logic for debugging
    pub fn preprocess(&self, img: &DynamicImage) -> Result<Array4<f32>, SigLipError> {
        let resized = img.resize_exact(
            self.image_size,
            self.image_size,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = resized.to_rgb8();
        let mut pixels: Vec<f32> =
            vec![0.0; 3 * self.image_size as usize * self.image_size as usize];
        let raw_samples = rgb.as_flat_samples().samples;
        let channel_step = (self.image_size * self.image_size) as usize;

        pixels
            .par_chunks_exact_mut(channel_step)
            .enumerate()
            .for_each(|(c, channel_slice)| {
                for i in 0..channel_step {
                    // SigLIP standard: (x / 127.5) - 1.0
                    channel_slice[i] = (f32::from(raw_samples[i * 3 + c]) / 127.5) - 1.0;
                }
            });

        let array = Array4::from_shape_vec(
            (1, 3, self.image_size as usize, self.image_size as usize),
            pixels,
        )?;
        Ok(array)
    }

    pub fn embed(&mut self, img: &DynamicImage) -> Result<Vec<f32>, SigLipError> {
        let array = self.preprocess(img)?;
        let input_tensor = Value::from_array(array)?;
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input_tensor])?;

        let extract = outputs["image_embeddings"].try_extract_tensor::<f32>()?;
        Ok(extract.1.to_vec())
    }
}

pub struct SigLipTextModel {
    pub session: Session,
    pub tokenizer: Tokenizer,
}

impl SigLipTextModel {
    pub fn new(
        text_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        context_length: usize,
    ) -> Result<Self, SigLipError> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(text_path)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            pad_id: 1, // Standard SigLIP pad_id
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: context_length,
                ..Default::default()
            }))
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    /// Exposes tokenization logic for debugging
    pub fn tokenize(&self, text: &str) -> Result<Array2<i64>, SigLipError> {
        let text = text.to_lowercase();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| i64::from(x)).collect();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, SigLipError> {
        let array = self.tokenize(&text)?;
        let input_tensor = Value::from_array(array)?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => input_tensor])?;

        let extract = outputs["text_embeddings"].try_extract_tensor::<f32>()?;
        Ok(extract.1.to_vec())
    }
}