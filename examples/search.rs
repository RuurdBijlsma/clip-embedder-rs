use color_eyre::eyre::Result;
use open_clip_inference::Clip;
use ort::ep::{CUDA, CoreML, DirectML, TensorRT};
use std::path::PathBuf;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = PathBuf::from("assets/img".to_owned());
    println!(" - Loading Embedders...");
    let start = Instant::now();

    let mut embedder = Clip::from_hf("RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX")
        .with_execution_providers(&[
            TensorRT::default().build(),
            CUDA::default().build(),
            DirectML::default().build(),
            CoreML::default().build(),
        ])
        .build().await?;

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
    let results = embedder.rank_images(&images, query_text)?;
    println!(" - Inference completed in {:.2?}", start_inf.elapsed());

    // Display Results
    println!(" - Search results\n");
    for (i, &(idx, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "★ " } else { "  " };
        println!("{} {:<20} | {:>6.2}", marker, valid_names[idx], prob * 100.);
    }

    Ok(())
}
