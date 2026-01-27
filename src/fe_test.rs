use color_eyre::eyre::{Result, eyre};
use fastembed::{ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};

fn main() -> Result<()> {
    color_eyre::install()?;

    // With custom options
    let mut model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
    )
    .map_err(|e| eyre!(e))?;

    let images = vec![
        "assets/img/beach_rocks.jpg",
        "assets/img/beetle_car.jpg",
        "assets/img/cat_face.jpg",
        "assets/img/dark_sunset.jpg",
        "assets/img/palace.jpg",
        "assets/img/rocky_coast.jpg",
        "assets/img/stacked_plates.jpg",
        "assets/img/verdant_cliff.jpg",
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(images, None).map_err(|e| eyre!(e))?;

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 2
    println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 512
    Ok(())
}
