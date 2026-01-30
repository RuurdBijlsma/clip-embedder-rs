#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use clip_sl2::{ModelConfig, SigLipTextModel, SigLipVisionModel};
use color_eyre::eyre::Result;
use image::{ImageBuffer, Rgb};
use ndarray::ArrayView1;
use std::path::Path;

fn get_stats(data: &ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| (x - mean).mul_add(x - mean, acc)) / data.len() as f32;
    (mean, var.sqrt())
}

// Reconstructs an image from the normalized tensor for visual inspection
fn save_debug_image(pix: &ndarray::Array4<f32>, filename: &str) -> Result<()> {
    let height = pix.shape()[2];
    let width = pix.shape()[3];
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Denormalize: (x * std + mean) * 255.
            let r = (pix[[0, 0, y, x]].mul_add(0.5, 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pix[[0, 1, y, x]].mul_add(0.5, 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pix[[0, 2, y, x]].mul_add(0.5, 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img_buf.save(filename)?;
    println!("Debug image saved to: {filename}");
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let assets = Path::new("models/clip_sl2/assets");
    let model_dir = assets.join("model");
    let img_path = assets.join("img/beach_rocks.jpg");

    let config = ModelConfig::from_file(model_dir.join("model_config.json"))?;

    println!("Loading models for Debug...");
    let mut vision = SigLipVisionModel::new(model_dir.join("visual.onnx"), config.clone())?;
    let mut text = SigLipTextModel::new(
        model_dir.join("text.onnx"),
        model_dir.join("tokenizer.json"),
        config,
    )?;

    let query_text = "A photo of Rocks";
    let img = image::open(&img_path)?;

    println!("\n--- DEBUG: ONNX VALUES (SigLIP 2) ---");

    // 1. Text IDs
    let ids = text.get_ids(query_text)?;
    let first_10_ids: Vec<i64> = ids.iter().take(10).map(|&x| i64::from(x)).collect();
    println!("Text Input IDs (first 10): {first_10_ids:?}");

    // 2. Image Preprocessing & Tensor Stats
    let pixel_tensor = vision.preprocess(&img);
    save_debug_image(&pixel_tensor, "debug_input_tensor.png")?;

    let (pix_mean, pix_std) = get_stats(
        &pixel_tensor
            .view()
            .into_shape_with_order(pixel_tensor.len())?,
    );
    println!("Image Pixel Values - Mean: {pix_mean:.6}, Std: {pix_std:.6}");

    // Check first few raw values (first channel, first row)
    let pix_slice = pixel_tensor.slice(ndarray::s![0, 0, 0, ..10]).to_vec();
    println!("Image Pixel Values (slice): {pix_slice:?}");

    // 3. Inference Embeddings
    let text_ids = text.tokenize(query_text)?;
    let text_embeds = text.inference(text_ids)?;
    let image_embeds = vision.inference(pixel_tensor)?;

    // Text Embed Stats
    let t_row = text_embeds.row(0);
    let (t_mean, t_std) = get_stats(&t_row);
    println!("Text Embeds[0] - Mean: {t_mean:.6}, Std: {t_std:.6}");
    println!(
        "Text Embeds[0] (first 5): {:?}",
        t_row.slice(ndarray::s![..5]).to_vec()
    );

    // Image Embed Stats
    let i_row = image_embeds.row(0);
    let (i_mean, i_std) = get_stats(&i_row);
    println!("Image Embeds[0] - Mean: {i_mean:.6}, Std: {i_std:.6}");
    println!(
        "Image Embeds[0] (first 5): {:?}",
        i_row.slice(ndarray::s![..5]).to_vec()
    );

    Ok(())
}
