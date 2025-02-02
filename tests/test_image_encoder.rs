use anyhow::{Context, Result};
use clip_embedder::image_encoder::ImageEncoder;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use std::path::Path;

#[test]
fn test_encode_images() -> Result<()> {
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
    let images = image_names
        .par_iter()
        .map(|name| {
            let path = data_dir.join("imgs").join(name);
            image::ImageReader::open(&path)
                .with_context(|| format!("Failed to open {}", path.display()))?
                .decode()
                .with_context(|| format!("Failed to decode {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let image_encoder = ImageEncoder::new(data_dir.join("clip_vision.onnx"), None, None, None)?;

    let embeddings = image_encoder.encode(&images)?;

    // Test 1: Verify output shape
    assert_eq!(
        embeddings.shape(),
        &[images.len(), 768],
        "Embedding shape mismatch"
    );

    // Test 2: Verify all embeddings are normalized
    for (i, row) in embeddings.rows().into_iter().enumerate() {
        let norm = row.dot(&row).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Embedding {} is not normalized (norm: {})",
            i,
            norm
        );
    }

    Ok(())
}
