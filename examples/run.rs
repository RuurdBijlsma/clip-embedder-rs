use color_eyre::eyre::Result;
use open_clip_rs::SigLipModel;
use std::time::Instant;

fn main() -> Result<()> {
    color_eyre::install()?;

    let mut model = SigLipModel::new(
        "assets/model/visual.onnx",
        "assets/model/text.onnx",
        "assets/model/tokenizer.json",
        384,
        64,
    )?;

    let img = image::open("assets/img/beach_rocks.jpg")?;

    // Warmup
    let _ = model.embed_image(&img)?;
    let _ = model.embed_text("warmup text")?;

    // Bench
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.embed_image(&img)?;
    }
    println!("Image Embedding Avg: {:?}", start.elapsed() / iters);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.embed_text("rocks in the rock business")?;
    }
    println!("Text Embedding Avg: {:?}", start.elapsed() / iters);

    // Print output
    let img_emb = model.embed_image(&img)?;
    let text_emb = model.embed_text("rocks in the rock business")?;
    println!("Image Embedding [0:50]\n{:?}", img_emb.iter().take(50).collect::<Vec<_>>());
    println!();
    println!("Text Embedding [0:50]\n{:?}", text_emb.iter().take(50).collect::<Vec<_>>());
    Ok(())
}
