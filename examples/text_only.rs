use color_eyre::Result;
use open_clip_inference::TextEmbedder;
use ort::ep::{CUDA, CoreML, DirectML, TensorRT};
use std::time::Instant;

#[cfg(feature = "hf-hub")]
#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
    let mut embedder = TextEmbedder::from_hf(model_id)
        .with_execution_providers(&[
            TensorRT::default().build(),
            CUDA::default().build(),
            DirectML::default().build(),
            CoreML::default().build(),
        ])
        .build()
        .await?;

    let texts = vec![
        "Some beachy rocks",
        "This is a beetle",
        "Face of a cat",
        "An underexposed sunset",
        "Some kind of palace",
        "A rocky coast",
        "Stacked plates on a table",
        "Grassy cliff, odd perspective",
    ];

    let now = Instant::now();
    println!("Embedding {} texts...", texts.len());
    let results = embedder.embed_texts(&texts)?;
    println!("Finished in {:?}", now.elapsed());

    println!("Result shape: {:?}", results.shape());

    Ok(())
}