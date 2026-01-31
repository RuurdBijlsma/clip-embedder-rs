use color_eyre::eyre::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use open_clip::ClipEmbedder;

fn main() -> Result<()> {
    color_eyre::install()?;

    // 1. Setup Paths
    // Point this to your exported model folder (containing text.onnx, visual.onnx, etc.)
    let model_dir = PathBuf::from("assets/model");
    let img_dir = PathBuf::from("assets/img");

    // 2. Load the Unified Embedder
    println!("Loading model from {:?}...", model_dir);
    let start_load = Instant::now();
    let embedder = ClipEmbedder::new(&model_dir)
        .wrap_err("Failed to load generic CLIP model")?;
    println!("Model loaded in {:.2?}", start_load.elapsed());

    // 3. Prepare Data
    let query_text = "A photo of a relaxed cat";
    let image_filenames = vec![
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];

    println!("Loading {} images...", image_filenames.len());
    let mut images = Vec::new();
    let mut valid_names = Vec::new();

    for name in &image_filenames {
        let path = img_dir.join(name);
        if path.exists() {
            // Load and decode image
            let img = image::open(&path)
                .wrap_err_with(|| format!("Failed to open image: {:?}", path))?;
            images.push(img);
            valid_names.push(name);
        } else {
            println!("Warning: Image not found at {:?}", path);
        }
    }

    if images.is_empty() {
        println!("No images found. Exiting.");
        return Ok(());
    }

    // 4. Inference (Batch Image Embedding)
    // The embedder automatically handles resizing (squash vs crop) and normalization
    println!("Embedding images...");
    let start_img = Instant::now();
    let image_embeddings = embedder.embed_image_batch(&images)?;
    println!("Image batch processed in {:.2?}", start_img.elapsed());

    // 5. Inference (Text Embedding)
    // The embedder automatically handles tokenization, padding, and masking
    println!("Embedding query: '{}'...", query_text);
    let start_text = Instant::now();
    let text_embedding = embedder.embed_text(query_text)?;
    println!("Text processed in {:.2?}", start_text.elapsed());

    // 6. Calculate Similarities & Rank
    println!("\n--- Search Results ---");

    let mut results = Vec::new();

    // Iterate over the rows of the image embedding matrix
    for (i, name) in valid_names.iter().enumerate() {
        // ndarray: Get row 'i' as a slice
        let img_vec = image_embeddings.row(i);
        let img_slice = img_vec.as_slice().unwrap();

        // Use the embedder's unified scoring function
        // This handles Sigmoid (SigLIP) vs Raw Logits (CLIP) automatically
        let score = embedder.similarity_score(img_slice, &text_embedding);

        results.push((name, score));
    }

    // Sort by score descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Print
    for (i, (name, score)) in results.iter().enumerate() {
        let marker = if i == 0 { "🏆" } else { "  " };

        // If it's SigLIP, score is 0.0-1.0 (Probability)
        // If it's CLIP, score is arbitrary logit (usually 20.0 - 100.0)
        // We print broadly to accommodate both.
        println!("{} {}: {:.4}", marker, name, score);
    }

    Ok(())
}