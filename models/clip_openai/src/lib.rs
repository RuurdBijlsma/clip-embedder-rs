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

pub struct ClipVisionModel {
    session: Session,
    pub image_size: u32,
}

impl ClipVisionModel {
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
        let (w, h) = img.dimensions();

        let (new_w, new_h) = if w < h {
            (
                self.image_size,
                (h as f32 * (self.image_size as f32 / w as f32)) as u32,
            )
        } else {
            (
                (w as f32 * (self.image_size as f32 / h as f32)) as u32,
                self.image_size,
            )
        };

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);

        let left = (new_w - self.image_size) / 2;
        let top = (new_h - self.image_size) / 2;
        let cropped = resized.crop_imm(left, top, self.image_size, self.image_size);
        let rgb = cropped.to_rgb8();

        let mean = [0.481_454_66, 0.457_827_5, 0.408_210_73];
        let std = [0.268_629_54, 0.261_302_6, 0.275_777_1];

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

        // FIX: Clone the string name so we don't hold an immutable borrow of self.session
        let input_name = self.session.inputs()[0].name().to_string();

        // Now self.session can be borrowed mutably because input_name is an owned String
        let outputs = self
            .session
            .run(ort::inputs![input_name.as_str() => input_tensor])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }
}

pub struct ClipTextModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl ClipTextModel {
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

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            pad_id: 49407,
            pad_token: "<|endoftext|>".to_string(),
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
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let masks: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let ids_array = Array2::from_shape_vec((1, ids.len()), ids)?;
        let masks_array = Array2::from_shape_vec((1, masks.len()), masks)?;

        // Note: Hardcoded keys like "input_ids" are &'static str, so they don't borrow self.session.
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Value::from_array(ids_array)?,
            "attention_mask" => Value::from_array(masks_array)?
        ])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }

    pub fn get_ids(&self, text: &str) -> Result<Vec<u32>, ClipError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

pub fn softmax(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps = x.mapv(|v| (v - max_val).exp());
    let sum = exps.sum();
    exps / sum
}
