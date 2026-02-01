use color_eyre::eyre::Result;
use open_clip_inference::{TextEmbedder, VisionEmbedder};
use std::path::PathBuf;
use std::time::Instant;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/common/examples_common.rs"
));

fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = PathBuf::from("assets/img".to_owned());
    println!(" - Loading Embedders...");
    let start = Instant::now();

    let model_id = "timm/ViT-SO400M-16-SigLIP2-384";
    let mut vision_embedder = VisionEmbedder::from_model_id(model_id)?;
    let mut text_embedder = TextEmbedder::from_model_id(model_id)?;

    println!(" - Loaded in {:.2?}", start.elapsed());

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

    println!(" - Embedding {} images...", images.len());
    let start_inf = Instant::now();
    let img_embs = vision_embedder.embed_images(&images)?;
    let text_embs = text_embedder.embed_texts(&[query_text.to_string()])?;
    println!(" - Inference completed in {:.2?}", start_inf.elapsed());

    // Calculate Raw Similarities
    let text_vec = text_embs.row(0);
    let similarities = img_embs.dot(&text_vec);

    // Post-processing (sigmoid or softmax)
    let scale = text_embedder.model_config.logit_scale.unwrap_or(1.0);
    let bias = text_embedder.model_config.logit_bias.unwrap_or(0.0);
    let logits: Vec<f32> = similarities
        .iter()
        .map(|&sim| sim.mul_add(scale, bias))
        .collect();

    let activation = text_embedder
        .model_config
        .activation_function
        .as_deref()
        .unwrap_or("softmax");
    let probs = if activation == "sigmoid" {
        logits.iter().map(|&l| sigmoid(l)).collect()
    } else {
        softmax(&logits)
    };
    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // Display Results
    println!(" - Search results ({})", activation.to_uppercase());
    println!("Query: \"{query_text}\"");
    println!("Logit Scale: {scale:.4} | Logit Bias: {bias:.4}");
    println!("{:-<60}", "");
    let score_suffix = if activation.to_lowercase() == "softmax" {
        "%"
    } else {
        ""
    };

    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "★ " } else { "  " };
        // Sigmoid probabilities are independent 0.0-1.0
        // Softmax probabilities sum to 1.0 across the whole list
        println!(
            "{} {:<20} | {:>6.2}{score_suffix}",
            marker,
            name,
            *prob * 100.0
        );
    }

    Ok(())
}
