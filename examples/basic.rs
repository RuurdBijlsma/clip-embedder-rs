use open_clip_inference::Clip;
use std::path::Path;

#[cfg(feature = "hf-hub")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";
    let mut clip = Clip::from_hf(model_id).build().await?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let results = clip.classify(&img, texts)?;

    for (text, prob) in results {
        println!("{}: {:.2}", text, prob * 100.0);
    }

    Ok(())
}