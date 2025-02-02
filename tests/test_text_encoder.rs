use anyhow::Result;
use clip_embedder::text_encoder::TextEncoder;
use std::path::PathBuf;

#[test]
fn test_encode_texts() -> Result<()> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");

    let text_encoder = TextEncoder::new(
        data_dir.join("clip_text.onnx"),
        data_dir.join("tokenizer.json"),
        Some(77),
    )?;

    let texts = ["A photo of a cat", "An image of a dog", ""];

    let embeddings = text_encoder.encode(&texts)?;

    // Test 1: Verify output shape
    assert_eq!(
        embeddings.shape(),
        &[texts.len(), 768],
        "Embedding shape mismatch"
    );

    // Test 2: Verify all embeddings are normalized
    for (i, row) in embeddings.rows().into_iter().enumerate() {
        let norm = row.dot(&row).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Embedding {} is not normalized (norm: {})",
            i,
            norm
        );
    }

    // Test 3: Verify empty text handling (should produce some embedding)
    let empty_embedding = embeddings.row(2);
    assert!(
        empty_embedding.iter().any(|&x| x != 0.0),
        "Empty text should produce a non-zero embedding"
    );

    Ok(())
}
