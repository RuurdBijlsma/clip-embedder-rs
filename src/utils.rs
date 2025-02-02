use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;
pub fn load_images(folder: &Path, image_names: [&str; 8]) -> Result<Vec<image::DynamicImage>> {
    let images = image_names
        .par_iter()
        .map(|name| {
            let path = folder.join(name);
            image::ImageReader::open(&path)
                .with_context(|| format!("Failed to open {}", path.display()))?
                .decode()
                .with_context(|| format!("Failed to decode {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(images)
}
