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

fn get_stats(data: ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let std = data.fold(0.0, |acc, &x| acc + (x - mean).powi(2));
    let std = (std / data.len() as f32).sqrt();
    (mean, std)
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
    let lower_text = query_text.to_lowercase();
    let text_emb = text_model.embed(&lower_text)?;

    // --- DEBUG: MATCHING PYTHON OUTPUT ---
    println!("\n--- DEBUG: ONNX VALUES (SigLIP 2 Rust) ---");
    let ids = text_model.get_ids(&lower_text)?;
    println!("Text Input IDs (first 10): {:?}", &ids[..std::cmp::min(10, ids.len())]);

    let pix = vision_model.preprocess(&images[0]);
    let pix_view = pix.view().into_shape_with_order(pix.len())?;
    let (pix_mean, pix_std) = get_stats(pix_view);
    println!("Image Pixel Values - Mean: {:.6}, Std: {:.6}", pix_mean, pix_std);

    let t_emb_view = text_emb.row(0);
    let (t_mean, t_std) = get_stats(t_emb_view);
    println!("Text Embeds - Mean: {:.6}, Std: {:.6}", t_mean, t_std);

    let i_emb_view = img_embs.row(0);
    let (i_mean, i_std) = get_stats(i_emb_view);
    println!("Image Embeds[0] - Mean: {:.6}, Std: {:.6}", i_mean, i_std);

    // 5. Calculate SigLIP Similarities
    // Logic: sigmoid( (img_emb @ text_emb.T) * scale + bias )
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