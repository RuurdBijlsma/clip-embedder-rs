use clip_sl2::{ModelConfig, SigLipTextModel, SigLipVisionModel, sigmoid};
use color_eyre::eyre::Result;
use ndarray::Axis;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    // --- 1. Setup Paths & Load Config ---
    let assets = Path::new("models/clip_sl2/assets");
    let model_dir = assets.join("model");
    let img_dir = assets.join("img");

    let config = ModelConfig::from_file(model_dir.join("model_config.json"))?;

    println!("Loading models...");
    let mut vision = SigLipVisionModel::new(model_dir.join("visual.onnx"), config.clone())?;
    let mut text = SigLipTextModel::new(
        model_dir.join("text.onnx"),
        model_dir.join("tokenizer.json"),
        config.clone(),
    )?;

    // --- 2. Load Data ---
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

    println!("Loading images...");
    for name in &image_files {
        let path = img_dir.join(name);
        if path.exists() {
            images.push(image::open(&path)?);
            valid_names.push(name);
        }
    }

    // --- 3. Run Inference ---
    println!("Running inference...");
    let image_embeds = vision.embed_batch(&images)?;

    let text_ids = text.tokenize(query_text)?;
    let text_embeds = text.inference(text_ids)?;

    // --- 4. Calculate Similarity ---
    let text_emb_vec = text_embeds.index_axis(Axis(0), 0);
    let similarities = image_embeds.dot(&text_emb_vec);

    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim.mul_add(config.logit_scale, config.logit_bias)))
        .collect();

    // --- 5. Sort and Print ---
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- SEARCH RESULTS ---");
    println!("Query: '{query_text}'");
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "★ [BEST]" } else { "  " };
        println!("{} {}: {:.2}%", marker, name, *prob * 100.0);
    }

    Ok(())
}
