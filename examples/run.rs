use clip_rs::{ClipTextModel, ClipVisionModel, softmax};
use color_eyre::eyre::Result;
use ndarray::Axis;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct ModelConfig {
    logit_scale: f32,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let image_dir = "assets/img";
    let model_dir = "assets/model";
    let query_text = "a photo of rocks";
    let image_files = vec![
        "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
        "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg"
    ];

    // 1. Load Config
    let config_str = fs::read_to_string(format!("{}/model_config.json", model_dir))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    // 2. Load Models
    println!("Loading models...");
    let mut vision_model = ClipVisionModel::new(format!("{}/visual.onnx", model_dir), 224)?;
    let mut text_model = ClipTextModel::new(
        format!("{}/text.onnx", model_dir),
        format!("{}/tokenizer.json", model_dir),
        77
    )?;

    // 3. Process Images
    println!("Processing {} images...", image_files.len());
    let mut images = Vec::new();
    let mut valid_names = Vec::new();

    for name in &image_files {
        let path = std::path::Path::new(image_dir).join(name);
        if path.exists() {
            images.push(image::open(path)?);
            valid_names.push(name.to_string());
        }
    }

    let img_embs = vision_model.embed_batch(&images)?;

    // 4. Process Text
    println!("Encoding query: '{}'...", query_text);
    let text_emb = text_model.embed(query_text)?;

    // 5. Calculate Similarities
    // Logic: (Batch_Img, Dim) @ (Dim, 1) -> (Batch_Img)
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let mut similarities = img_embs.dot(&text_emb_vec);

    // Apply logit scale
    similarities *= config.logit_scale;

    // Apply Softmax
    let probs = softmax(&similarities);

    // 6. Rank Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- RUST SEARCH RESULTS ---");
    println!("Query: '{}'", query_text);
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };
        println!("{} {}: {:.2}%", marker, name, *prob * 100.0);
    }

    Ok(())
}