use color_eyre::eyre::Result;
use open_clip::{TextEmbedder, VisionEmbedder};
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_id = "timm/ViT-SO400M-16-SigLIP2-384";
    let mut vision_embedder = VisionEmbedder::new(model_id)?;
    let mut text_embedder = TextEmbedder::new(model_id)?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let img_embs = vision_embedder.embed_image(&img)?;
    let text_embs = text_embedder.embed_texts(texts)?;

    dbg!(img_embs);
    dbg!(text_embs);

    Ok(())
}
