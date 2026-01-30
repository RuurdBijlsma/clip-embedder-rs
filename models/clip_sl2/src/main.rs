use clip_sl2::{SigLipTextModel, SigLipVisionModel, sigmoid};
use color_eyre::eyre::Result;
use ndarray::{Axis, ArrayView1};
use serde::Deserialize;
use std::fs;

const ASSETS_FOLDER: &str = "models/clip_sl2/assets";

fn get_asset(name: &str) -> String {
    format!("{ASSETS_FOLDER}/{name}")
}

#[derive(Deserialize)]
struct ModelConfig {
    logit_scale: f32,
    logit_bias: f32,
    image_size: u32,
    context_length: usize,
}

/// Helper to calculate mean and standard deviation to match Python's np.mean/np.std
fn get_stats(data: ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| acc + (x - mean).powi(2)) / data.len() as f32;
    (mean, var.sqrt())
}

use image::{ImageBuffer, Rgb};

/// Saves the preprocessed tensor to a file for visual inspection
fn save_debug_image(pix: &ndarray::Array4<f32>, path: &str) -> Result<()> {
    // pix is [1, 3, H, W]
    let height = pix.shape()[2];
    let width = pix.shape()[3];

    // Create an ImageBuffer to hold the RGB data
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // SigLIP 2 De-normalization: (val * 0.5) + 0.5
            // Then scale to 255
            let r = ((pix[[0, 0, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((pix[[0, 1, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((pix[[0, 2, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;

            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img_buf.save(path)?;
    println!("Debug image saved to: {}", path);
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let image_dir = get_asset("img");
    let model_dir = get_asset("model");
    let query_text = "A photo of Rocks";
    let image_files = vec![
        "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
        "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg"
    ];

    // 1. Load Config
    let config_str = fs::read_to_string(format!("{}/model_config.json", model_dir))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    // 2. Load Models
    println!("Loading models...");
    let mut vision_model = SigLipVisionModel::new(
        format!("{}/visual.onnx", model_dir),
        config.image_size
    )?;
    let mut text_model = SigLipTextModel::new(
        format!("{}/text.onnx", model_dir),
        format!("{}/tokenizer.json", model_dir),
        config.context_length
    )?;

    // 3. Process Images
    let mut images = Vec::new();
    let mut valid_names = Vec::new();
    for name in &image_files {
        let path = std::path::Path::new(&image_dir).join(name);
        if path.exists() {
            images.push(image::open(path)?);
            valid_names.push(name.to_string());
        }
    }

    println!("Running Vision ONNX model...");
    let img_embs = vision_model.embed_batch(&images)?;

    // 4. Process Text
    println!("Encoding query: '{}'...", query_text);
    let text_emb = text_model.embed(query_text)?;

    // --- DEBUG: MATCHING PYTHON OUTPUT ---
    println!("\n--- DEBUG: ONNX VALUES (SigLIP 2) ---");

    // 1. Text Input IDs
    let ids = text_model.get_ids(query_text)?;
    // Take first 10, or fewer if string is very short
    let display_ids = &ids[..std::cmp::min(10, ids.len())];
    println!("Text Input IDs (first 10): {:?}", display_ids);

    // 2. Image Pixel Values (First Image)
    let pix = vision_model.preprocess(&images[0]);
    save_debug_image(&pix, "debug_input_tensor.png")?;
    // Flatten for stats
    let pix_flat = pix.view().into_shape_with_order(pix.len())?;
    let (pix_mean, pix_std) = get_stats(pix_flat);
    println!("Image Pixel Values - Mean: {:.6}, Std: {:.6}", pix_mean, pix_std);

    // Slice: Batch 0, Channel 0, Row 0, first 5 Columns
    let pix_slice = pix.slice(ndarray::s![0, 0, 0, ..30]).to_vec();
    println!("Image Pixel Values (slice): {:?}", pix_slice);

    // 3. Text Embeddings
    let t_emb_row = text_emb.row(0);
    let (t_mean, t_std) = get_stats(t_emb_row);
    println!("Text Embeds[0] - Mean: {:.6}, Std: {:.6}", t_mean, t_std);
    println!("Text Embeds[0] (first 5): {:?}", t_emb_row.slice(ndarray::s![..5]).to_vec());

    // 4. Image Embeddings
    let i_emb_row = img_embs.row(0);
    let (i_mean, i_std) = get_stats(i_emb_row);
    println!("Image Embeds[0] - Mean: {:.6}, Std: {:.6}", i_mean, i_std);
    println!("Image Embeds[0] (first 5): {:?}", i_emb_row.slice(ndarray::s![..5]).to_vec());
    // -------------------------------------

    // 5. Calculate SigLIP Similarities
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let similarities = img_embs.dot(&text_emb_vec);

    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim * config.logit_scale + config.logit_bias))
        .collect();

    // 6. Rank Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- SIGLIP 2 ONNX RESULTS ---");
    println!("Query: '{}'", query_text);
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };
        println!("{} {}: {:.2}", marker, name, *prob * 100.0);
    }

    Ok(())
}