#[cfg(test)]
mod tests {
    use color_eyre::Result;
    #[cfg(feature = "hf-hub")]
    use open_clip_inference::Clip;
    #[cfg(feature = "hf-hub")]
    use std::path::Path;

    #[cfg(feature = "hf-hub")]
    #[tokio::test]
    async fn test_hf() -> Result<()> {
        dbg!(1);
        let embedder = Clip::from_hf("RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX")
            .build()
            .await?;

        dbg!(1);
        let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("Failed to load image");
        let cat_text = "A photo of a cat";
        let texts = &[cat_text, "A photo of a dog", "A photo of a beignet"];
        dbg!(1);

        let results = embedder.classify(&img, texts)?;
        dbg!(1);

        // Check first result
        let (best_tag, prob) = &results[0];
        assert_eq!(best_tag, cat_text);
        assert!(*prob > 0.99);

        // Check second result
        let (_second_tag, second_prob) = &results[1];
        assert!(*second_prob < 0.1);

        for (text, prob) in results {
            println!("{}: {:.2}%", text, prob * 100.0);
        }

        Ok(())
    }
}
