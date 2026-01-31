use crate::config::OpenClipConfig;
use crate::error::{ClipError, Result};
use crate::onnx::OnnxSession;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array2, Array4, ArrayView, Axis, IxDyn};
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;

pub struct VisionTower {
    pub session: OnnxSession,
    pub config: OpenClipConfig,
    pub input_name: String,
}

impl VisionTower {
    pub fn new(model_path: impl AsRef<Path>, config_path: impl AsRef<Path>) -> Result<Self> {
        let session = OnnxSession::new(model_path)?;
        let config = OpenClipConfig::from_file(config_path)?;

        let input_name = session
            .find_input(&["pixel_values", "input"])
            .ok_or_else(|| ClipError::Config("Could not find vision input node".to_string()))?;

        Ok(Self {
            session,
            config,
            input_name,
        })
    }

    pub fn embed_images(&mut self, images: &[DynamicImage]) -> Result<Array2<f32>> {
        if images.is_empty() {
            return Err(ClipError::Inference("Empty batch".to_string()));
        }

        // Parallel preprocessing
        let processed: Vec<Array4<f32>> = images
            .par_iter()
            .map(|img| self.preprocess(img))
            .collect::<Result<Vec<_>>>()?;

        let views: Vec<_> = processed.iter().map(|a| a.view()).collect();
        let batch_tensor = ndarray::concatenate(Axis(0), &views)
            .map_err(|e| ClipError::Inference(e.to_string()))?;

        let input_tensor = Value::from_array(batch_tensor)?;
        let outputs = self
            .session
            .session
            .run(ort::inputs![&self.input_name => input_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)
            .map_err(|e| ClipError::Inference(e.to_string()))?;

        Ok(view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ClipError::Inference(e.to_string()))?
            .to_owned())
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<Array4<f32>> {
        let size = self.config.model_cfg.vision_cfg.image_size;
        let interp = match self.config.preprocess_cfg.interpolation.as_str() {
            "bicubic" => FilterType::CatmullRom,
            "bilinear" => FilterType::Triangle,
            _ => FilterType::Nearest,
        };

        let resized = match self.config.preprocess_cfg.resize_mode.as_str() {
            "squash" => image.resize_exact(size, size, interp),
            _ => {
                let (w, h) = image.dimensions();
                let scale = size as f32 / w.min(h) as f32;
                let (nw, nh) = ((w as f32 * scale) as u32, (h as f32 * scale) as u32);
                let r = image.resize_exact(nw, nh, interp);
                r.crop_imm((nw - size) / 2, (nh - size) / 2, size, size)
            }
        };

        let rgb = resized.to_rgb8();
        let (mean, std) = (
            self.config.preprocess_cfg.mean,
            self.config.preprocess_cfg.std,
        );
        let mut flat = vec![0.0f32; 3 * (size as usize).pow(2)];

        let channel_len = (size as usize).pow(2);
        for c in 0..3 {
            for i in 0..channel_len {
                let val = f32::from(rgb.as_raw()[i * 3 + c]) / 255.0;
                flat[c * channel_len + i] = (val - mean[c]) / std[c];
            }
        }

        Array4::from_shape_vec((1, 3, size as usize, size as usize), flat)
            .map_err(|e| ClipError::Inference(e.to_string()))
    }
}
