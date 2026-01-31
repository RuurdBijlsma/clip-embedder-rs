use color_eyre::eyre::Result;
use open_clip::ClipEmbedder;
use std::path::PathBuf;
use std::time::Instant;

const ASSETS_FOLDER: &str = "models/open_clip/assets";

fn main() -> Result<()> {
    color_eyre::install()?;

    // 1. Setup Paths
    let model_dir = PathBuf::from(format!("{ASSETS_FOLDER}/model"));
    let img_dir = PathBuf::from(format!("{ASSETS_FOLDER}/img"));

    // 2. Initialize the Unified Embedder
    // This loads config, tokenizer, and both ONNX towers
    println!("🚀 Loading Unified CLIP Engine...");
    let start = Instant::now();
    let mut embedder = ClipEmbedder::new(&model_dir)?;
    println!("✅ Model loaded in {:.2?}", start.elapsed());

    // 3. Define Query and Images
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

    // 4. Load and Decode Images
    let mut images = Vec::new();
    let mut valid_names = Vec::new();

    for name in &image_files {
        let path = img_dir.join(name);
        if let Ok(img) = image::open(&path) {
            images.push(img);
            valid_names.push(name.to_string());
        }
    }

    if images.is_empty() {
        return Err(color_eyre::eyre::eyre!("No images found in {:?}", img_dir));
    }

    // 5. Batch Inference
    println!("🧠 Running batch inference for {} images...", images.len());
    let start_inf = Instant::now();

    // Embed all images at once (uses parallel preprocessing)
    let img_embs = embedder.embed_images(&images)?;

    // Embed the text query (wrapped in a Vec for batch processing)
    let text_embs = embedder.embed_texts(&[query_text.to_string()])?;

    println!("⚡ Inference completed in {:.2?}", start_inf.elapsed());

    // 6. Calculate Probabilities
    // This uses Matrix Multiplication: (Images x Dim) • (Texts x Dim)^T
    // It then automatically applies Sigmoid (SigLIP) or Softmax (CLIP)
    let probs = embedder.compute_probs(&img_embs, &text_embs);

    // 7. Process and Sort Results
    // probs is an Array2 where row = image, column = text
    let mut results: Vec<_> = valid_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let score = probs[[i, 0]]; // Index into the first (and only) text column
            (name, score)
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 8. Display Results
    println!("\n🔎 SEARCH RESULTS");
    println!("Query: \"{}\"", query_text);
    println!("{:=<40}", "");

    for (i, (name, score)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };

        // Format as percentage
        println!("{} {:<20} | {:.2}", marker, name, *score * 100.0);
    }

    Ok(())
}
