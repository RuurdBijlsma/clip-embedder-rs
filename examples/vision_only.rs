use color_eyre::Result;
use open_clip_inference::VisionEmbedder;
use ort::ep::{CUDA, CoreML, DirectML, TensorRT};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "hf-hub")]
#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
    let mut embedder = VisionEmbedder::from_hf(model_id)
        .with_execution_providers(&[
            TensorRT::default().build(),
            CUDA::default().build(),
            DirectML::default().build(),
            CoreML::default().build(),
        ])
        .build()
        .await?;

    let img_dir = PathBuf::from("assets/img".to_owned());
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
    for name in image_files {
        if let Ok(img) = image::open(img_dir.join(name)) {
            images.push(img);
        }
    }

    let now = Instant::now();
    println!("Embedding {} images...", images.len());
    let results = embedder.embed_images(&images)?;
    println!("Finished in {:?}", now.elapsed());

    println!("Result shape: {:?}", results.shape());

    Ok(())
}
