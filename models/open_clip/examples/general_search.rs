use color_eyre::eyre::Result;
use open_clip::config::LocalConfig;
use open_clip::{TextTower, VisionTower};
use std::path::PathBuf;
use std::time::Instant;

const ASSETS_FOLDER: &str = "models/open_clip/assets";

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_dir = PathBuf::from(format!("{ASSETS_FOLDER}/model"));
    let img_dir = PathBuf::from(format!("{ASSETS_FOLDER}/img"));

    println!("🚀 Loading Towers...");
    let start = Instant::now();

    let local_config = LocalConfig::from_file(model_dir.join("model_config.json"))
        .expect("Failed to load model_config.json");

    let mut vision_tower = VisionTower::new(
        model_dir.join("visual.onnx"),
        model_dir.join("open_clip_config.json"),
    )?;

    let mut text_tower = TextTower::new(
        model_dir.join("text.onnx"),
        model_dir.join("open_clip_config.json"),
        model_dir.join("tokenizer.json"),
        local_config.tokenizer_needs_lowercase,
    )?;

    println!("✅ Loaded in {:.2?}", start.elapsed());

    let query_text = "A photo of Rocks";
    let image_files = vec![
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];

    let mut images = Vec::new();
    let mut valid_names = Vec::new();
    for name in image_files {
        if let Ok(img) = image::open(img_dir.join(name)) {
            images.push(img);
            valid_names.push(name.to_string());
        }
    }

    println!("🧠 Embedding {} images...", images.len());
    let start_inf = Instant::now();

    // 2. Run Inference
    let img_embs = vision_tower.embed_images(&images)?;
    let text_embs = text_tower.embed_texts(&[query_text.to_string()])?;

    println!("⚡ Inference completed in {:.2?}", start_inf.elapsed());

    // 3. Calculate Similarities (Dot Product)
    // img_embs is [M, D], text_embs is [1, D].
    // Similarity is a [M] vector.
    let text_vec = text_embs.row(0);
    let similarities = img_embs.dot(&text_vec);

    // 4. Calculate Probabilities using LocalConfig math
    // Formula: sigmoid(similarity * scale + bias)
    let scale = local_config.logit_scale.unwrap_or(1.0);
    let bias = local_config.logit_bias.unwrap_or(0.0);

    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim.mul_add(scale, bias)))
        .collect();

    // 5. Process and Sort Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // 6. Display Results
    println!("\n🔍 SEARCH RESULTS (SigLIP Probabilities)");
    println!("Query: \"{}\"", query_text);
    println!("Logit Scale: {:.4} | Logit Bias: {:.4}", scale, bias);
    println!("{:-<50}", "");

    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "★ [BEST]" } else { "  " };
        println!("{} {:<20} | {:.2}", marker, name, *prob * 100.0);
    }

    Ok(())
}
