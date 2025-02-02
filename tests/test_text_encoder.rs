// tests/test_text_encoder.rs

use anyhow::{Context, Result};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;
use std::path::Path;
use tokenizers::Tokenizer;
use clip_embedder::text_encoder::{encode_texts, MAX_LENGTH};  // Use your crate name

#[test]
fn test_encode_texts() -> Result<()> {
    // Setup paths and initialize components
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    // Initialize tokenizer with same config as main
    let mut tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {:?}", e))?;

    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_LENGTH,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            stride: 0,
            direction: tokenizers::TruncationDirection::Right,
        }))
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation error: {:?}", e))?;

    tokenizer
        .with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(MAX_LENGTH),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

    // Initialize ONNX session
    let text_session = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(data_dir.join("clip_text.onnx"))
        .with_context(|| format!("Failed to load model from {:?}", data_dir.join("clip_text.onnx")))?;

    // Test cases
    let texts = [
        "A photo of a cat",
        "An image of a dog",
        "",
    ];

    let embeddings = encode_texts(&tokenizer, &text_session, &texts)?;

    // Test 1: Verify output shape
    assert_eq!(embeddings.shape(), &[texts.len(), 768], "Embedding shape mismatch");

    // Test 2: Verify all embeddings are normalized
    for (i, row) in embeddings.rows().into_iter().enumerate() {
        let norm = row.dot(&row).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Embedding {} is not normalized (norm: {})",
            i, norm
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