use crate::utils::load_images;
use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage};
use ndarray::{s, Array2, Array3, Axis};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    value::Tensor,
};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

pub struct ImageEncoder {
    session: Session,
    image_size: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImageEncoder {
    pub fn new(
        model_path: impl AsRef<Path>,
        image_size: Option<u32>,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
    ) -> Result<Self> {
        let image_size = image_size.unwrap_or(224);
        let mean = mean.unwrap_or([0.48145466, 0.4578275, 0.40821073]);
        let std = std.unwrap_or([0.26862954, 0.2613026, 0.2757771]);

        let session = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        Ok(Self {
            session,
            image_size,
            mean,
            std,
        })
    }

    fn preprocess(&self, img: &DynamicImage) -> Result<Array3<f32>> {
        let target_size = self.image_size;
        let (w, h) = (img.width(), img.height());
        let (new_w, new_h) = if w < h {
            (
                target_size,
                (h as f32 * (target_size as f32 / w as f32)) as u32,
            )
        } else {
            (
                (w as f32 * (target_size as f32 / h as f32)) as u32,
                target_size,
            )
        };

        let resized = img.resize_exact(new_w, new_h, FilterType::CatmullRom);
        let left = (new_w - target_size) / 2;
        let top = (new_h - target_size) / 2;
        let cropped = resized.crop_imm(left, top, target_size, target_size);

        let rgb = cropped.to_rgb8();
        let mut array = ndarray::Array::from_shape_vec(
            (target_size as usize, target_size as usize, 3),
            rgb.into_raw(),
        )?
        .mapv(|v| v as f32 / 255.0);

        for c in 0..3 {
            let mut channel = array.slice_mut(s![.., .., c]);
            channel -= self.mean[c];
            channel /= self.std[c];
        }

        Ok(array.permuted_axes([2, 0, 1]))
    }

    pub fn encode(&self, images: &[DynamicImage]) -> Result<Array2<f32>> {
        let preprocessed = images
            .par_iter()
            .map(|img| self.preprocess(img))
            .collect::<Result<Vec<_>>>()?;

        let batch = ndarray::stack(
            Axis(0),
            &preprocessed.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?;

        let input_tensor = Tensor::from_array(batch.into_dyn())?;
        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input_tensor]?)?;

        let mut embeddings = outputs["image_embeddings"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?
            .to_owned();

        let norms = embeddings.map_axis(Axis(1), |row| row.dot(&row).sqrt());
        embeddings /= &norms.insert_axis(Axis(1));

        Ok(embeddings)
    }
}

pub fn run_image_encoding() -> Result<()> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let image_names = [
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];
    let images =
        load_images(&data_dir.join("images"), image_names).context("Failed to load images")?;

    let image_encoder = ImageEncoder::new(data_dir.join("clip_vision.onnx"), None, None, None)?;

    let start = Instant::now();
    let embeddings = image_encoder.encode(&images)?;
    println!("Encoding images took: {:?}", start.elapsed());

    // Calculate similarities.
    let query_embedding = embeddings.row(0);
    println!("\nQuery: {}", image_names[0]);

    for (i, text) in image_names.iter().enumerate().skip(1) {
        let target_embedding = embeddings.row(i);
        let similarity = query_embedding.dot(&target_embedding);
        println!("\tSimilarity to '{}': {:.2}", text, similarity);
    }

    Ok(())
}
