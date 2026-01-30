use crate::image_encoder::ImageEncoder;
use crate::text_encoder::TextEncoder;
use crate::utils::load_images;
use anyhow::{anyhow, Context, Result};
use image::DynamicImage;
use std::path::Path;

pub fn search_images(query: &str, images: &[DynamicImage]) -> Result<usize> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let image_encoder = ImageEncoder::new(data_dir.join("clip_vision.onnx"), None, None, None)?;
    let text_encoder = TextEncoder::new(
        data_dir.join("clip_text.onnx"),
        data_dir.join("tokenizer.json"),
        None,
    )?;

    let image_embeddings = image_encoder.encode(images)?;
    let text_embeddings = text_encoder.encode(&[query])?;

    // Compute cosine similarities
    let similarities = text_embeddings.dot(&image_embeddings.t());

    // Find the index of the highest similarity
    let best_match_idx = similarities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow!("Failed to determine best match index"))?;

    Ok(best_match_idx)
}

pub fn run_image_search() -> Result<()> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let image_names = [
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];
    let query = "A grassy depression in a mountain.";

    let best_idx = search_images(
        query,
        &load_images(&data_dir.join("images"), image_names).context("Failed to load images")?,
    )?;

    println!("========== [ SEARCHING IMAGES ] ==========");
    println!();
    println!("Image list: \n{}", image_names.join("\n"));
    println!();

    println!("Query: {}", query);
    println!("Closest image: {}", image_names[best_idx]);

    Ok(())
}
