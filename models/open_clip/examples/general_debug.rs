#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use color_eyre::eyre::Result;
use image::{ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView1};
use open_clip::ClipEmbedder;
use std::path::Path;

fn get_stats(data: &ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| (x - mean).mul_add(x - mean, acc)) / data.len() as f32;
    (mean, var.sqrt())
}

/// Uses the library's config to reverse normalization for visual inspection.
fn save_debug_image(pix: &Array4<f32>, config: &open_clip::config::ModelConfig, filename: &str) -> Result<()> {
    let height = pix.shape()[2];
    let width = pix.shape()[3];
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    let mean = config.mean;
    let std = config.std;

    for y in 0..height {
        for x in 0..width {
            // Denormalize: (pixel * std + mean) * 255
            // Using [0, channel, y, x]
            let r = (pix[[0, 0, y, x]].mul_add(std[0], mean[0]) * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pix[[0, 1, y, x]].mul_add(std[1], mean[1]) * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pix[[0, 2, y, x]].mul_add(std[2], mean[2]) * 255.0).clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img_buf.save(filename)?;
    println!("📸 Debug image (reconstructed) saved to: {filename}");
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let assets = Path::new("models/open_clip/assets");
    let model_dir = assets.join("model");
    let img_path = assets.join("img/beach_rocks.jpg");

    println!("🚀 Loading Unified CLIP Engine for Debug...");
    let mut embedder = ClipEmbedder::new(&model_dir)?;

    let query_text = "A photo of Rocks";
    let img = image::open(&img_path)?;

    println!("\n--- DEBUG: GENERAL CLIP ---");
    println!("Model Type:     {:?}", embedder.config.model_type);
    println!("Image Mode:     {}", embedder.config.resize_mode);

    // --- 1. Text Preprocessing (Directly using TextProcessor) ---
    // This uses the exact .to_lowercase() fix we just added.
    let (text_ids, text_mask) = embedder.text.process(query_text)?;

    println!("\n[TEXT PREPROCESSING]");
    println!("Input IDs (first 10): {:?}", text_ids.slice(ndarray::s![0, ..10]).to_vec());
    println!("Attention Mask:       {:?}", text_mask.slice(ndarray::s![0, ..10]).to_vec());

    // --- 2. Image Preprocessing (Directly using VisionProcessor) ---
    // This tests your Squash vs ShortestEdge logic.
    let pixel_tensor = embedder.vision.process(&img)?;
    save_debug_image(&pixel_tensor, &embedder.config, "debug_onnx_input.png")?;

    let (pix_mean, pix_std) = get_stats(
        &pixel_tensor
            .view()
            .into_shape_with_order(pixel_tensor.len())?,
    );
    println!("\n[IMAGE PREPROCESSING]");
    println!("Pixel Stats - Mean: {pix_mean:.6}, Std: {pix_std:.6}");
    println!("Pixel Slice (ch0, row0): {:?}", pixel_tensor.slice(ndarray::s![0, 0, 0, ..10]).to_vec());

    // --- 3. Inference (Directly using OnnxRunners) ---
    let text_embeds = embedder.text_ort.run_text(text_ids, text_mask)?;
    let image_embeds = embedder.vision_ort.run_vision(pixel_tensor)?;

    // Text Embed Stats
    let t_row = text_embeds.row(0);
    let (t_mean, t_std) = get_stats(&t_row);
    println!("\n[INFERENCE RESULTS]");
    println!("Text Embeds  - Mean: {t_mean:.6}, Std: {t_std:.6}");
    println!("Text Embeds  (first 5): {:?}", t_row.slice(ndarray::s![..5]).to_vec());

    // Image Embed Stats
    let i_row = image_embeds.row(0);
    let (i_mean, i_std) = get_stats(&i_row);
    println!("Image Embeds - Mean: {i_mean:.6}, Std: {i_std:.6}");
    println!("Image Embeds (first 5): {:?}", i_row.slice(ndarray::s![..5]).to_vec());

    // --- 4. Math Verification ---
    // Verify scale and bias are being applied as expected
    let logits = embedder.compute_logits(&image_embeds, &text_embeds);
    let probs = embedder.compute_probs(&image_embeds, &text_embeds);

    println!("\n[SCORING CHECK]");
    println!("Logit Scale:      {:.4}", embedder.config.logit_scale);
    println!("Logit Bias:       {:.4}", embedder.config.logit_bias);
    println!("Raw Logit:        {:.4}", logits[[0, 0]]);
    println!("Final Match Prob: {:.2}%", probs[[0, 0]] * 100.0);

    Ok(())
}