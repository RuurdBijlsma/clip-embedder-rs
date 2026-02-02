use color_eyre::Result;
use open_clip_inference::Clip;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;
    let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
    let mut clip = Clip::from_model_id(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let results = clip.classify(&img, texts)?;

    for (text, prob) in results {
        println!("{}: {:.4}%", text, prob * 100.0);
    }

    Ok(())
}
