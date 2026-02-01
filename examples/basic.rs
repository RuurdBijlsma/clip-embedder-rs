use color_eyre::Result;
use open_clip::{TextEmbedder, VisionEmbedder};
use std::path::Path;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/common/examples_common.rs"
));

fn main() -> Result<()> {
    color_eyre::install()?;
    let model_id = "timm/MobileCLIP2-S2-OpenCLIP";
    let mut vision_embedder = VisionEmbedder::from_model_id(model_id)?;
    let mut text_embedder = TextEmbedder::from_model_id(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let img_emb = vision_embedder.embed_image(&img)?;
    let text_embs = text_embedder.embed_texts(texts)?;

    let similarities = text_embs.dot(&img_emb);
    let scale = text_embedder.model_config.logit_scale.unwrap_or(1.0);
    let bias = text_embedder.model_config.logit_bias.unwrap_or(0.0);
    let logits: Vec<f32> = similarities
        .iter()
        .map(|&s| s.mul_add(scale, bias))
        .collect();

    for (text, prob) in texts.iter().zip(softmax(&logits)) {
        println!("{}: {:.2}%", text, prob * 100.0);
    }

    Ok(())
}
