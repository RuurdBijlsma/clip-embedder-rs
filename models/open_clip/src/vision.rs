use crate::config::{ModelConfig, OpenClipConfig};
use color_eyre::eyre::{Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, ArrayBase, OwnedRepr, Dim};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
enum ResizeStrategy {
    /// Forces the image to (size, size), ignoring aspect ratio.
    /// Common in SigLIP models.
    Squash,
    /// Resizes the shortest edge to 'size' while maintaining aspect ratio,
    /// then center crops.
    /// Common in original OpenAI CLIP.
    ShortestEdge,
}

#[derive(Debug, Clone)]
pub struct VisionProcessor {
    image_size: u32,
    mean: [f32; 3],
    std: [f32; 3],
    interpolation: FilterType,
    resize_strategy: ResizeStrategy,
}

impl VisionProcessor {
    pub fn new(config: &ModelConfig) -> Self {
        // Map "squash" / "shortest" directly from flat config
        let resize_strategy = match config.resize_mode.as_str() {
            "squash" => ResizeStrategy::Squash,
            "shortest" => ResizeStrategy::ShortestEdge,
            _ => ResizeStrategy::ShortestEdge,
        };

        // 2. Determine Interpolation
        // Map string to image::FilterType
        let interpolation = match config.interpolation.as_str() {
            "bicubic" => FilterType::CatmullRom, // Best approx for bicubic in `image` crate
            "bilinear" => FilterType::Triangle,
            "nearest" => FilterType::Nearest,
            _ => FilterType::CatmullRom, // Default high quality
        };

        Self {
            image_size: config.image_size,
            mean: config.mean,
            std: config.std,
            interpolation,
            resize_strategy,
        }
    }

    /// Preprocesses an image into a (1, 3, H, W) normalized tensor.
    pub fn process(&self, image: &DynamicImage) -> Result<Array4<f32>> {
        // 1. Resize & Crop
        let processed_img = match self.resize_strategy {
            ResizeStrategy::Squash => image.resize_exact(
                self.image_size,
                self.image_size,
                self.interpolation,
            ),
            ResizeStrategy::ShortestEdge => {
                // Resize preserving aspect ratio such that the smallest dimension becomes image_size
                let (w, h) = image.dimensions();
                let (nw, nh) = if w < h {
                    (self.image_size, (h as f32 * (self.image_size as f32 / w as f32)) as u32)
                } else {
                    ((w as f32 * (self.image_size as f32 / h as f32)) as u32, self.image_size)
                };

                let resized = image.resize_exact(nw, nh, self.interpolation);

                // Center Crop
                let left = (nw - self.image_size) / 2;
                let top = (nh - self.image_size) / 2;

                resized.crop_imm(left, top, self.image_size, self.image_size)
            }
        };

        // 2. Convert to RGB8 (strips alpha if present)
        let rgb = processed_img.to_rgb8();

        // 3. Normalize & Tensorize (Parallelized)
        // ONNX expects NCHW: (Batch, Channel, Height, Width)
        let height = self.image_size as usize;
        let width = self.image_size as usize;
        let channel_step = height * width;

        // Flattened vector: RRR...GGG...BBB...
        let mut flat_pixels = vec![0.0f32; 3 * channel_step];
        let raw_samples = rgb.as_flat_samples();
        let raw_slice = raw_samples.samples; // [R, G, B, R, G, B, ...]

        // We process each channel (R, G, B) in parallel chunks
        flat_pixels
            .par_chunks_exact_mut(channel_step)
            .enumerate()
            .for_each(|(c, channel_out)| {
                // c=0 is R, c=1 is G, c=2 is B
                let mean = self.mean[c];
                let std = self.std[c];

                for i in 0..channel_step {
                    // map output index `i` back to interleaved input index
                    let input_idx = i * 3 + c;
                    let val = f32::from(raw_slice[input_idx]) / 255.0;
                    channel_out[i] = (val - mean) / std;
                }
            });

        // Create Array4 (1, 3, H, W)
        let tensor = Array4::from_shape_vec(
            (1, 3, height, width),
            flat_pixels,
        ).wrap_err("Failed to create generic vision tensor")?;

        Ok(tensor)
    }

    /// Process a batch of images.
    /// Returns (Batch, 3, H, W)
    pub fn process_batch(&self, images: &[DynamicImage]) -> Result<Array4<f32>> {
        if images.is_empty() {
            return Ok(Array4::zeros((0, 3, self.image_size as usize, self.image_size as usize)));
        }

        // Process all images in parallel
        let tensors: Result<Vec<Array4<f32>>> = images
            .par_iter()
            .map(|img| self.process(img))
            .collect();

        let tensors = tensors?;

        // Stack them
        // Note: This is a bit allocation-heavy (vec of arrays -> one big array),
        // but safe and clean. For massive batches, we could pre-allocate one big vector.
        let views: Vec<_> = tensors.iter().map(|a| a.view()).collect();
        let batch = ndarray::stack(ndarray::Axis(0), &views)
            .wrap_err("Failed to stack image batch")?;

        Ok(batch)
    }
}