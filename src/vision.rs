use crate::config::{OnnxModelConfig, OpenClipConfig};
use crate::error::ClipError;
use crate::onnx::OnnxSession;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array2, Array4, ArrayView, Axis, IxDyn};
use ort::value::Value;
use rayon::prelude::*;

pub struct VisionEmbedder {
    pub session: OnnxSession,
    pub config: OpenClipConfig,
    pub local_config: OnnxModelConfig,
    pub input_name: String,
}

impl VisionEmbedder {
    // todo: use bon and let user set cache folder+model_id, or model folder directly
    pub fn new(model_id: &str) -> Result<Self, ClipError> {
        let model_dir = OnnxSession::get_model_dir(model_id);
        let model_path = model_dir.join("visual.onnx");
        let config_path = model_dir.join("open_clip_config.json");
        let local_config_path = model_dir.join("model_config.json");

        let session = OnnxSession::new(model_path)?;
        let config = OpenClipConfig::from_file(config_path)?;
        let local_config = OnnxModelConfig::from_file(local_config_path)?;

        let input_name = session
            .find_input(&["pixel_values", "input"])
            .ok_or_else(|| ClipError::Config("Could not find vision input node".to_string()))?;

        Ok(Self {
            session,
            config,
            local_config,
            input_name,
        })
    }

    /// Embed a single image
    pub fn embed_image(
        &mut self,
        image: &DynamicImage,
    ) -> Result<ndarray::Array1<f32>, ClipError> {
        let embs = self.embed_images(std::slice::from_ref(image))?;
        let len = embs.len();
        embs.into_shape_with_order(len)
            .map_err(|e| ClipError::Inference(e.to_string()))
    }

    /// Embed a batch of images
    pub fn embed_images(&mut self, images: &[DynamicImage]) -> Result<Array2<f32>, ClipError> {
        let batch_tensor = self.preprocess_batch(images)?;

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

    /// Preprocess batch of images
    pub fn preprocess_batch(&self, images: &[DynamicImage]) -> Result<Array4<f32>, ClipError> {
        if images.is_empty() {
            return Err(ClipError::Inference("Empty batch".to_string()));
        }

        let batch_size = images.len();
        let size = self.config.model_cfg.vision_cfg.image_size as usize;
        let mut batch_tensor = Array4::<f32>::zeros((batch_size, 3, size, size));
        batch_tensor
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(images.par_iter())
            .try_for_each(|(mut slot, img)| self.preprocess_into(img, &mut slot))?;

        Ok(batch_tensor)
    }

    /// Preprocess single image
    pub fn preprocess(&self, image: &DynamicImage) -> Result<Array4<f32>, ClipError> {
        self.preprocess_batch(std::slice::from_ref(image))
    }

    fn preprocess_into(
        &self,
        image: &DynamicImage,
        out_view: &mut ndarray::ArrayViewMut3<f32>,
    ) -> Result<(), ClipError> {
        let size = self.config.model_cfg.vision_cfg.image_size;
        let interp = match self.config.preprocess_cfg.interpolation.as_str() {
            "bicubic" => FilterType::CatmullRom,
            "bilinear" => FilterType::Triangle,
            _ => FilterType::Nearest,
        };

        #[allow(
            clippy::single_match_else,
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let resized = match self.config.preprocess_cfg.resize_mode.as_str() {
            "squash" => image.resize_exact(size, size, interp),
            _ => {
                let (width, height) = image.dimensions();
                let scale = size as f32 / width.min(height) as f32;
                let scaled_width = (width as f32 * scale).round() as u32;
                let scaled_height = (height as f32 * scale).round() as u32;
                let resized = image.resize_exact(scaled_width, scaled_height, interp);
                let x = ((scaled_width as f32 - size as f32) / 2.0).round() as u32;
                let y = ((scaled_height as f32 - size as f32) / 2.0).round() as u32;
                resized.crop_imm(x, y, size, size)
            }
        };

        let rgb = resized.to_rgb8();
        let (mean, std) = (
            self.config.preprocess_cfg.mean,
            self.config.preprocess_cfg.std,
        );

        let pixels = rgb.as_raw();
        let channel_len = (size as usize).pow(2);
        for c in 0..3 {
            let channel_slice = out_view.index_axis_mut(Axis(0), c);
            let flat_channel = channel_slice
                .into_slice()
                .ok_or_else(|| ClipError::Inference("Layout mismatch".into()))?;
            for i in 0..channel_len {
                let val = f32::from(pixels[i * 3 + c]) / 255.0;
                flat_channel[i] = (val - mean[c]) / std[c];
            }
        }

        Ok(())
    }
}
