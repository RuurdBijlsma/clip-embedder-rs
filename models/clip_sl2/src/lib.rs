use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array4, ArrayView, IxDyn};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

#[derive(thiserror::Error, Debug)]
pub enum ClipError {
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
    session: Session,
    pub image_size: u32,
}

impl SigLipVisionModel {
    pub fn new(visual_path: impl AsRef<Path>, image_size: u32) -> Result<Self, ClipError> {
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

    pub fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        // SigLIP 2 uses "squash" resize (non-aspect-ratio preserving)
        let resized = img.resize_exact(
            self.image_size,
            self.image_size,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = resized.to_rgb8();

        // SigLIP 2 normalization: (val - 0.5) / 0.5
        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];

        let mut pixels = vec![0.0f32; 3 * (self.image_size * self.image_size) as usize];
        let channel_step = (self.image_size * self.image_size) as usize;
        let raw_samples = rgb.as_flat_samples().samples;

        pixels
            .par_chunks_exact_mut(channel_step)
            .enumerate()
            .for_each(|(c, channel_slice)| {
                for i in 0..channel_step {
                    let val = raw_samples[i * 3 + c] as f32 / 255.0;
                    channel_slice[i] = (val - mean[c]) / std[c];
                }
            });

        Array4::from_shape_vec(
            (1, 3, self.image_size as usize, self.image_size as usize),
            pixels,
        )
        .unwrap()
    }

    pub fn embed_batch(&mut self, images: &[DynamicImage]) -> Result<Array2<f32>, ClipError> {
        let mut batch_array = Array4::zeros((
            images.len(),
            3,
            self.image_size as usize,
            self.image_size as usize,
        ));

        for (i, img) in images.iter().enumerate() {
            let processed = self.preprocess(img);
            batch_array
                .slice_mut(ndarray::s![i, .., .., ..])
                .assign(&processed.slice(ndarray::s![0, .., .., ..]));
        }

        let input_tensor = Value::from_array(batch_array)?;
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input_tensor])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }
}

pub struct SigLipTextModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl SigLipTextModel {
    pub fn new(
        text_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        context_length: usize,
    ) -> Result<Self, ClipError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(text_path)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        // SigLIP / Gemma uses pad_id 0
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            pad_id: 0,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: context_length,
                ..Default::default()
            }))
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, text: &str) -> Result<Array2<f32>, ClipError> {
        // lowercase to match python .lower()
        let encoding = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let ids_array = Array2::from_shape_vec((1, ids.len()), ids)?;

        // SigLIP export only takes input_ids
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Value::from_array(ids_array)?
        ])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }

    pub fn get_ids(&self, text: &str) -> Result<Vec<u32>, ClipError> {
        let encoding = self
            .tokenizer
            .encode(text.to_lowercase(), true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
