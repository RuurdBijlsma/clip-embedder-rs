use clip_sl1::{SigLipTextModel, SigLipVisionModel};
use color_eyre::eyre::Result;
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::time::Instant;

const ASSETS_FOLDER: &str = "models/clip_sl1/assets";

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_dir = PathBuf::from(format!("{ASSETS_FOLDER}/model"));
    let img_dir = PathBuf::from(format!("{ASSETS_FOLDER}/img"));

    println!("🚀 Loading SL1 Models...");
    let start = Instant::now();

    let mut vision_model = SigLipVisionModel::new(model_dir.join("visual.onnx"), 384)?;
    let mut text_model = SigLipTextModel::new(
        model_dir.join("text.onnx"),
        model_dir.join("tokenizer.json"),
        64,
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
        let path = img_dir.join(name);
        if let Ok(img) = image::open(&path) {
            images.push(img);
            valid_names.push(name.to_string());
        }
    }

    println!("🧠 Embedding {} images...", images.len());
    let start_inf = Instant::now();

    // Collect image embeddings
    let mut img_embeddings = Vec::new();
    for img in &images {
        img_embeddings.extend(vision_model.embed(img)?);
    }
    let img_embs_arr = Array2::from_shape_vec((images.len(), img_embeddings.len() / images.len()), img_embeddings)?;

    // Text embedding
    let text_emb = text_model.embed(query_text)?;
    let text_vec = ndarray::Array1::from_vec(text_emb);

    println!("⚡ Inference completed in {:.2?}", start_inf.elapsed());

    // SigLIP models typically have a learnable logit_scale and logit_bias
    // If you don't have a config, these are often 10.0 to 20.0 for scale
    let logit_scale = 112.33287048339844;
    let logit_bias = -16.54642105102539;

    // 1. Calculate similarities (Dot product)
    let similarities = img_embs_arr.dot(&text_vec);

    // 2. Apply Scale and Bias and Sigmoid
    // Note: SigLIP specifically uses Sigmoid rather than Softmax
    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim.mul_add(logit_scale, logit_bias)))
        .collect();

    // 3. Sort Results
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n🔎 SL1 SEARCH RESULTS (SIGMOID)");
    println!("Query: \"{}\"", query_text);
    println!("{:-<60}", "");

    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "★ [BEST]" } else { "  " };
        println!("{} {:<20} | {:>6.2}", marker, name, *prob * 100.0);
    }

    Ok(())
}