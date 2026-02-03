#[cfg(test)]
mod tests {
    use color_eyre::Result;
    use open_clip_inference::{Clip, TextEmbedder, VisionEmbedder};
    use std::path::{Path, PathBuf};

    const LAION_MODEL_ID: &str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K";
    const OPENAI_MODEL_ID: &str = "timm/vit_base_patch32_clip_224.openai";
    const SIGLIP1_MODEL_ID: &str = "timm/ViT-SO400M-14-SigLIP-384";
    const SIGLIP2_MODEL_ID: &str = "timm/ViT-SO400M-16-SigLIP2-384";
    const MOBILECLIP2_MODEL_ID: &str = "timm/MobileCLIP2-S2-OpenCLIP";

    fn run_model_integration_test(model_id: &str, expected_dim: usize) -> Result<()> {
        let mut vision_embedder = VisionEmbedder::from_local_id(model_id).build()?;
        let mut text_embedder = TextEmbedder::from_local_id(model_id).build()?;

        // Test Vision Embedding
        let img_path = PathBuf::from("assets/img/beach_rocks.jpg");
        let img = image::open(img_path)?;
        let vision_embs = vision_embedder.embed_image(&img)?;
        assert_eq!(
            vision_embs.len(),
            expected_dim,
            "Vision embedding dimension mismatch for {model_id}"
        );

        // Test Text Embedding
        let text = "A photo of rocks";
        let text_embs = text_embedder.embed_text(text)?;
        assert_eq!(
            text_embs.len(),
            expected_dim,
            "Text embedding dimension mismatch for {model_id}"
        );

        // Test Search Logic
        let img_dir = PathBuf::from("assets/img");
        let image_files = vec!["beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg"];

        let mut images = Vec::new();
        let mut valid_names = Vec::new();
        for name in &image_files {
            if let Ok(img) = image::open(img_dir.join(name)) {
                images.push(img);
                valid_names.push(name.to_string());
            }
        }

        let img_embs = vision_embedder.embed_images(&images)?;
        let query_embs = text_embedder.embed_texts(&[text.to_string()])?;

        let text_vec = query_embs.row(0);
        let similarities = img_embs.dot(&text_vec);

        // Find index of max similarity
        let mut max_sim = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &sim) in similarities.iter().enumerate() {
            if sim > max_sim {
                max_sim = sim;
                max_idx = i;
            }
        }

        // beach_rocks.jpg should be the most similar to "A photo of rocks"
        assert_eq!(
            valid_names[max_idx], "beach_rocks.jpg",
            "Search logic failure for {model_id}"
        );

        Ok(())
    }

    #[test]
    fn test_openai() -> Result<()> {
        run_model_integration_test(OPENAI_MODEL_ID, 512)
    }

    #[test]
    fn test_laion() -> Result<()> {
        run_model_integration_test(LAION_MODEL_ID, 512)
    }

    #[test]
    fn test_mobileclip() -> Result<()> {
        run_model_integration_test(MOBILECLIP2_MODEL_ID, 512)
    }

    #[test]
    fn test_siglip1() -> Result<()> {
        run_model_integration_test(SIGLIP1_MODEL_ID, 1152)
    }

    #[test]
    fn test_siglip2() -> Result<()> {
        run_model_integration_test(SIGLIP2_MODEL_ID, 1152)
    }

    #[tokio::test]
    async fn test_hf() -> Result<()> {
        let mut embedder = Clip::from_hf("RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX")
            .build()
            .await?;

        let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
        let match_text = "A photo of a cat";
        let texts = &[
            match_text,
            "A photo of a dog",
            "A photo of a beignet",
        ];

        let results = embedder.classify(&img, texts)?;

        let (best_tag, score) = results.first().unwrap();
        assert_eq!(best_tag, match_text);
        assert!(*score > 0.9);

        for (text, prob) in results {
            println!("{}: {:.4}%", text, prob * 100.0);
        }

        Ok(())
    }
}
