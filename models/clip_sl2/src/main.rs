use clip_sl1::{SigLipTextModel, SigLipVisionModel};
use color_eyre::eyre::Result;
use std::time::Instant;

const ASSETS_FOLDER: &str = "models/clip_sl1/assets";

fn get_asset(name: &str) -> String {
    format!("{ASSETS_FOLDER}/{name}")
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let mut vision_model = SigLipVisionModel::new(get_asset("model/visual.onnx"), 384)?;
    let mut text_model = SigLipTextModel::new(
        get_asset("model/text.onnx"),
        get_asset("model/tokenizer.json"),
        64,
    )?;
    let img = image::open(get_asset("img/beach_rocks.jpg"))?;

    // Warmup
    let _ = vision_model.embed(&img)?;
    let _ = text_model.embed("warmup text")?;

    // Bench Vision
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = vision_model.embed(&img)?;
    }
    println!("Vision Embedding Avg: {:?}", start.elapsed() / iters);

    // Bench Text
    let start = Instant::now();
    for _ in 0..iters {
        let _ = text_model.embed("rocks in the rock business")?;
    }
    println!("Text Embedding Avg: {:?}", start.elapsed() / iters);

    // Output check
    let img_emb = vision_model.embed(&img)?;
    let text_emb = text_model.embed("rocks in the rock business")?;

    println!("\nVision [0:50]: {:?}", &img_emb[..50]);
    println!("Text [0:50]:   {:?}", &text_emb[..50]);

    Ok(())
}
