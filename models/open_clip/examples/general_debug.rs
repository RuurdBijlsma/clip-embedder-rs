#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use color_eyre::eyre::Result;
use image::{ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView1, s};
use open_clip::config::LocalConfig;
use open_clip::{TextTower, VisionTower};
use std::path::Path;

fn get_stats(data: &ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| (x - mean).mul_add(x - mean, acc)) / data.len() as f32;
    (mean, var.sqrt())
}

fn save_debug_image(
    pix: &Array4<f32>,
    config: &open_clip::config::OpenClipConfig,
    filename: &str,
) -> Result<()> {
    let height = pix.shape()[2];
    let width = pix.shape()[3];
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);
    let (mean, std) = (config.preprocess_cfg.mean, config.preprocess_cfg.std);

    for y in 0..height {
        for x in 0..width {
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
    let model_dir = assets.join("timm/ViT-SO400M-14-SigLIP-384");
    let img_path = assets.join("img/beach_rocks.jpg");

    let mut vision_tower = VisionTower::new(
        model_dir.join("visual.onnx"),
        model_dir.join("open_clip_config.json"),
    )?;

    let local_config = LocalConfig::from_file(model_dir.join("model_config.json"))
        .expect("Failed to get local config file");

    // Pass the new model_config.json path here
    let mut text_tower = TextTower::new(
        model_dir.join("text.onnx"),
        model_dir.join("open_clip_config.json"),
        model_dir.join("tokenizer.json"),
        local_config.tokenizer_needs_lowercase,
        local_config.pad_id,
    )?;

    let query_text = "A photo of Rocks";
    let img = image::open(&img_path)?;

    println!("\n--- DEBUG: OPEN_CLIP DECOUPLED ---");
    println!(
        "Image Mode:     {}",
        vision_tower.config.preprocess_cfg.resize_mode
    );
    println!("Lowercase Req:  {}", text_tower.tokenizer_needs_lowercase);

    // 1. Text Preprocessing
    let (ids, mask) = text_tower.tokenize(&[query_text.to_string()])?;
    println!("\n[TEXT PREPROCESSING]");
    println!(
        "Input IDs (first 10): {:?}",
        ids.slice(ndarray::s![0, ..10]).to_vec()
    );
    println!(
        "Attention Mask:       {:?}",
        mask.slice(ndarray::s![0, ..10]).to_vec()
    );

    // --- 2. Image Preprocessing ---
    let pixel_tensor = vision_tower.preprocess(&img)?;
    save_debug_image(&pixel_tensor, &vision_tower.config, "debug_onnx_input.png")?;

    let flat_pixels = pixel_tensor
        .view()
        .into_shape_with_order(pixel_tensor.len())?;
    let (pix_mean, pix_std) = get_stats(&flat_pixels);
    println!("\n[IMAGE PREPROCESSING]");
    println!("Pixel Stats - Mean: {pix_mean:.6}, Std: {pix_std:.6}");
    // Slice first 10 pixels of the first row of the Red channel
    println!(
        "Pixel Slice (ch0, row0): {:?}",
        pixel_tensor.slice(s![0, 0, 0, ..10]).to_vec()
    );

    // --- 3. Inference ---
    let image_embeds = vision_tower.embed_images(&[img])?;
    let text_embeds = text_tower.embed_texts(&[query_text.to_string()])?;

    let t_row = text_embeds.row(0);
    let (t_mean, t_std) = get_stats(&t_row);
    println!("\n[INFERENCE RESULTS]");
    println!("Text Embeds  - Mean: {t_mean:.6}, Std: {t_std:.6}");
    println!(
        "Text Embeds  (first 5): {:?}",
        t_row.slice(s![..5]).to_vec()
    );

    let i_row = image_embeds.row(0);
    let (i_mean, i_std) = get_stats(&i_row);
    println!("Image Embeds - Mean: {i_mean:.6}, Std: {i_std:.6}");
    println!(
        "Image Embeds (first 5): {:?}",
        i_row.slice(s![..5]).to_vec()
    );

    // --- 4. Scoring ---
    let similarity = i_row.dot(&t_row);
    println!("\n[SCORING CHECK]");
    println!("Raw Dot Product (Similarity): {:.4}", similarity);

    Ok(())
}
