use clip_sl2::{SigLipTextModel, SigLipVisionModel, sigmoid};
use color_eyre::eyre::Result;
use image::DynamicImage;
use ndarray::{ArrayView1, Axis};
use serde::Deserialize;
use std::fs;
use clip_sl2::perf::perf_main;

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

fn get_stats(data: ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| acc + (x - mean).powi(2)) / data.len() as f32;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let image_dir = get_asset("img");
    let model_dir = get_asset("model");
    let query_text = "A photo of Rocks";
    let image_files = vec![
        "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
        "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg",
    ];

    let config_str = fs::read_to_string(format!("{}/model_config.json", model_dir))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    println!("Loading models...");
    let mut vision_model = SigLipVisionModel::new(format!("{}/visual.onnx", model_dir), config.image_size)?;
    let mut text_model = SigLipTextModel::new(
        format!("{}/text.onnx", model_dir),
        format!("{}/tokenizer.json", model_dir),
        config.context_length,
    )?;

    // 1. Load all images into memory
    let mut images = Vec::new();
    let mut valid_names = Vec::new();
    for name in &image_files {
        let path = std::path::Path::new(&image_dir).join(name);
        if path.exists() {
            images.push(image::open(path)?);
            valid_names.push(name.to_string());
        }
    }

    // 2. Vision Inference
    println!("Running Vision batch inference ({} images)...", images.len());
    let image_embeds = vision_model.embed_batch(&images)?;

    // 3. Text Inference
    println!("Encoding query and running Text inference...");
    let text_ids = text_model.tokenize(query_text)?;
    let text_embeds = text_model.inference(text_ids)?;

    // --- DEBUG OUTPUT ---
    println!("\n--- DEBUG INFO ---");

    // Check first image embedding stats
    let i_row = image_embeds.row(0);
    let (i_mean, i_std) = get_stats(i_row);
    println!("Image Embeds[0] - Mean: {:.6}, Std: {:.6}", i_mean, i_std);
    println!("Image Embeds[0] (first 5): {:?}", i_row.slice(ndarray::s![..5]).to_vec());

    // Check text embedding stats
    let t_row = text_embeds.row(0);
    let (t_mean, t_std) = get_stats(t_row);
    println!("Text Embeds[0]  - Mean: {:.6}, Std: {:.6}", t_mean, t_std);
    println!("Text Embeds[0]  (first 5): {:?}", t_row.slice(ndarray::s![..5]).to_vec());

    // 4. Calculate Similarities & Probabilities
    // (Batch, Dim) dot (Dim) -> (Batch)
    let text_emb_vec = text_embeds.index_axis(Axis(0), 0);
    let similarities = image_embeds.dot(&text_emb_vec);

    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim * config.logit_scale + config.logit_bias))
        .collect();

    // 5. Rank and Print Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- SIGLIP 2 RESULTS ---");
    println!("Query: '{}'", query_text);
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };
        println!("{} {:<20}: {:.2}%", marker, name, *prob * 100.0);
    }

    let _ = perf_main();

    Ok(())
}