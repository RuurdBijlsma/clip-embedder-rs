use crate::{ClipError, ModelConfig};
use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array4, ArrayView, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;

pub struct SigLipVisionModel {
    session: Session,
    config: ModelConfig,
}

impl SigLipVisionModel {
    pub fn new(model_path: impl AsRef<Path>, config: ModelConfig) -> Result<Self, ClipError> {
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;

        Ok(Self { session, config })
    }

    /// Preprocesses a single image into a tensor (1, 3, H, W).
    /// Exposed for debug purposes.
    pub fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let size = self.config.image_size;
        // resize_exact is used to match the specific pipeline logic provided previously.
        // Standard CLIP usually does Resize -> CenterCrop.
        let resized = img.resize_exact(
            size,
            size,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = resized.to_rgb8();

        let mean = self.config.mean;
        let std = self.config.std;

        let mut pixels = vec![0.0f32; 3 * (size * size) as usize];
        let channel_step = (size * size) as usize;
        let raw_samples = rgb.as_flat_samples().samples;

        // Parallel normalization
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
            (1, 3, size as usize, size as usize),
            pixels,
        )
            .expect("Shape logic is guaranteed by construction")
    }

    /// Run inference on a batch of preprocessed tensors.
    /// Input shape: (Batch, 3, H, W)
    pub fn inference(&mut self, input: Array4<f32>) -> Result<Array2<f32>, ClipError> {
        let input_tensor = Value::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input_tensor])?;

        let (shape_ort, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape_ort.iter().map(|&x| x as usize).collect();

        // Convert dynamic shape to 2D (Batch, EmbedDim)
        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;
        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }

    /// High-level helper: Takes a list of images, preprocesses them in parallel,
    /// and runs inference.
    pub fn embed_batch(&mut self, images: &[DynamicImage]) -> Result<Array2<f32>, ClipError> {
        if images.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let size = self.config.image_size as usize;
        let mut batch_array = Array4::zeros((images.len(), 3, size, size));

        // Preprocess images in parallel
        let processed_images: Vec<Array4<f32>> = images
            .par_iter()
            .map(|img| self.preprocess(img))
            .collect();

        // Stack into batch array
        for (i, p_img) in processed_images.iter().enumerate() {
            batch_array
                .slice_mut(ndarray::s![i, .., .., ..])
                .assign(&p_img.slice(ndarray::s![0, .., .., ..]));
        }

        self.inference(batch_array)
    }
}