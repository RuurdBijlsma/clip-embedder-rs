use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

// todo:
// think about separating text & vision model
// * in ruurd photos they will be in separate places (ingest & api[search])
// find last discrepency between image preprocessing :( (check pixel difference with python)
// Make "search-engine" usecase and make benchmark, also make it in python, compare performance.
// * Maybe use imagenet dataset or something

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

pub struct SigLipModel {
    visual_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
    image_size: u32,
}

impl SigLipModel {
    pub fn new(
        visual_path: impl AsRef<Path>,
        text_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        image_size: u32,
        context_length: usize,
    ) -> Result<Self, SigLipError> {
        let threads = num_cpus::get();
        let visual_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(visual_path)?;
        let text_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(text_path)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            pad_id: 1, // Standard SigLIP/ViT pad ID
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: context_length,
                ..Default::default()
            }))
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;
        Ok(Self {
            visual_session,
            text_session,
            tokenizer,
            image_size,
        })
    }

    pub fn embed_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>, SigLipError> {
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
                    channel_slice[i] = (raw_samples[i * 3 + c] as f32 / 127.5) - 1.0;
                }
            });
        let array = Array4::from_shape_vec(
            (1, 3, self.image_size as usize, self.image_size as usize),
            pixels,
        )?;
        let input_tensor = Value::from_array(array)?;
        let outputs = self
            .visual_session
            .run(ort::inputs!["pixel_values" => input_tensor])?;
        let extract = outputs["image_embeddings"].try_extract_tensor::<f32>()?;
        let embeddings = extract.1;

        Ok(embeddings.to_vec())
    }

    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, SigLipError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SigLipError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();
        let array = Array2::from_shape_vec((1, seq_len), ids)?;

        let input_tensor = Value::from_array(array)?;
        let outputs = self
            .text_session
            .run(ort::inputs!["input_ids" => input_tensor])?;

        let extract = outputs["text_embeddings"].try_extract_tensor::<f32>()?;
        let embeddings = extract.1;

        Ok(embeddings.to_vec())
    }
}
