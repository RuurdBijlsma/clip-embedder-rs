#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use clip_sl1::{SigLipTextModel, SigLipVisionModel};
use color_eyre::eyre::Result;
use image::{ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView1, s};
use std::path::Path;

fn get_stats(data: &ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| (x - mean).mul_add(x - mean, acc)) / data.len() as f32;
    (mean, var.sqrt())
}

#[allow(clippy::cast_sign_loss)]
fn save_debug_image(pix: &Array4<f32>, filename: &str) -> Result<()> {
    let height = pix.shape()[2];
    let width = pix.shape()[3];
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Reverse (x / 127.5) - 1.0  =>  (x + 1.0) * 127.5
            let r = ((pix[[0, 0, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((pix[[0, 1, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((pix[[0, 2, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img_buf.save(filename)?;
    println!("📸 Debug image (reconstructed) saved to: {filename}");
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let assets = Path::new("models/clip_sl1/assets");
    let model_dir = assets.join("model");
    let img_path = assets.join("img/beach_rocks.jpg");

    let mut vision_model = SigLipVisionModel::new(model_dir.join("visual.onnx"), 384)?;
    let mut text_model = SigLipTextModel::new(
        model_dir.join("text.onnx"),
        model_dir.join("tokenizer.json"),
        64, // context length
    )?;

    let query_text = "A photo of Rocks";
    let img = image::open(&img_path)?;

    println!("\n--- DEBUG: CLIP_SL1 ---");

    // 1. Text Preprocessing
    let ids = text_model.tokenize(query_text)?;
    println!("\n[TEXT PREPROCESSING]");
    println!(
        "Input IDs (first 10): {:?}",
        ids.slice(s![0, ..10]).to_vec()
    );

    // 2. Image Preprocessing
    let pixel_tensor = vision_model.preprocess(&img)?;
    save_debug_image(&pixel_tensor, "debug_sl1_input.png")?;

    let flat_pixels = pixel_tensor
        .view()
        .into_shape_with_order(pixel_tensor.len())?;
    let (pix_mean, pix_std) = get_stats(&flat_pixels);
    println!("\n[IMAGE PREPROCESSING]");
    println!("Pixel Stats - Mean: {pix_mean:.6}, Std: {pix_std:.6}");
    println!(
        "Pixel Slice (ch0, row0): {:?}",
        pixel_tensor.slice(s![0, 0, 0, ..30]).to_vec()
    );

    // 3. Inference
    let i_vec = vision_model.embed(&img)?;
    let t_vec = text_model.embed(query_text)?;

    let i_row = ArrayView1::from(&i_vec);
    let t_row = ArrayView1::from(&t_vec);

    let (t_mean, t_std) = get_stats(&t_row);
    println!("\n[INFERENCE RESULTS]");
    println!("Text Embeds  - Mean: {t_mean:.6}, Std: {t_std:.6}");
    println!("Text Embeds  (first 5): {:?}", &t_vec[..5]);

    let (i_mean, i_std) = get_stats(&i_row);
    println!("Image Embeds - Mean: {i_mean:.6}, Std: {i_std:.6}");
    println!("Image Embeds (first 5): {:?}", &i_vec[..5]);

    // 4. Scoring
    let similarity: f32 = i_vec.iter().zip(t_vec.iter()).map(|(a, b)| a * b).sum();
    println!("\n[SCORING CHECK]");
    println!("Raw Dot Product (Similarity): {similarity:.4}");

    Ok(())
}
