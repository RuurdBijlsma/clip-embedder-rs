use anyhow::{Context, Result};
use clip_embedder::search_images::search_images;
use clip_embedder::utils::load_images;
use std::path::Path;

#[test]
pub fn test_image_search() -> Result<()> {
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
    let expected_idx = image_names
        .iter()
        .position(|&name| name == "verdant_cliff.jpg")
        .ok_or_else(|| anyhow::anyhow!("Expected image not found"))?;
    assert_eq!(
        best_idx, expected_idx,
        "Expected {} at index {}, but got index {}",
        image_names[expected_idx], expected_idx, best_idx
    );

    Ok(())
}
