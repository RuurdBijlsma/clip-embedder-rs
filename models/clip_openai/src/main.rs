#![allow(clippy::cast_precision_loss)]

use clip_openai::{ClipTextModel, ClipVisionModel, softmax};
use color_eyre::eyre::Result;
use ndarray::{Axis, ArrayView1};
use serde::Deserialize;
use std::fs;

const ASSETS_FOLDER: &str = "models/clip_openai/assets";

fn get_asset(name: &str) -> String {
    format!("{ASSETS_FOLDER}/{name}")
}

#[derive(Deserialize)]
struct ModelConfig {
    logit_scale: f32,
}

/// Helper to calculate standard deviation since ndarray-stats is a separate crate
fn get_stats(data: &ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let std = data.fold(0.0, |acc, &x| (x - mean).mul_add(x - mean, acc));
    let std = (std / data.len() as f32).sqrt();
    (mean, std)
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let query_text = "a photo of rocks";
    let image_files = vec![
        "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
        "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg"
    ];

    // 1. Load Config
    let config_str = fs::read_to_string(get_asset("model/model_config.json"))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    // 2. Load Models
    println!("Loading models...");
    let mut vision_model = ClipVisionModel::new(get_asset("model/visual.onnx"), 224)?;
    let mut text_model = ClipTextModel::new(
        get_asset("model/text.onnx"),
        get_asset("model/tokenizer.json"),
        77
    )?;

    // 3. Process Images
    println!("Processing {} images...", image_files.len());
    let mut images = Vec::new();
    let mut valid_names = Vec::new();

    for name in &image_files {
        let path = std::path::Path::new(&get_asset("img")).join(name);
        if path.exists() {
            images.push(image::open(path)?);
            valid_names.push(name.to_string());
        }
    }

    let img_embs = vision_model.embed_batch(&images)?;

    // 4. Process Text
    println!("Encoding query: '{query_text}'...");
    let text_emb = text_model.embed(query_text)?;

    // --- DEBUG: MATCHING PYTHON OUTPUT ---
    println!("\n--- DEBUG: ONNX (NO-HF) VALUES ---");

    // 1. Text Input IDs
    let ids = text_model.get_ids(query_text)?;
    println!("Text Input IDs (first 10): {:?}", &ids[..10]);

    // 2. Image Input Tensors (of the first image)
    let pix = vision_model.preprocess(&images[0]);
    let pix_view = pix.view().into_shape_with_order(pix.len())?;
    let (pix_mean, pix_std) = get_stats(&pix_view);
    println!("Image Pixel Values - Mean: {pix_mean:.6}, Std: {pix_std:.6}");

    // Slice: first 5 pixels of the first channel, first row
    let pix_slice = pix.slice(ndarray::s![0, 0, 0, ..5]).to_vec();
    println!("Image Pixel Values (slice): {pix_slice:?}");

    // 3. Text Embeddings
    let t_emb_view = text_emb.row(0);
    let (t_mean, t_std) = get_stats(&t_emb_view);
    println!("Text Embeds - Mean: {t_mean:.6}, Std: {t_std:.6}");
    println!("Text Embeds (first 5): {:?}", t_emb_view.slice(ndarray::s![..5]).to_vec());

    // 4. Image Embeddings (First Image)
    let i_emb_view = img_embs.row(0);
    let (i_mean, i_std) = get_stats(&i_emb_view);
    println!("Image Embeds[0] - Mean: {i_mean:.6}, Std: {i_std:.6}");
    println!("Image Embeds[0] (first 5): {:?}", i_emb_view.slice(ndarray::s![..5]).to_vec());
    // -------------------------------------

    // 5. Calculate Similarities
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let mut similarities = img_embs.dot(&text_emb_vec);
    similarities *= config.logit_scale;
    let probs = softmax(&similarities);

    // 6. Rank Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- SEARCH RESULTS ---");
    println!("Query: '{query_text}'");
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };
        println!("{} {}: {:.2}%", marker, name, *prob * 100.0);
    }

    Ok(())
}