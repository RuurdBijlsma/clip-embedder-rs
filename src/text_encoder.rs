use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2};
use ort::session::SessionInputValue;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::path::Path;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

const MAX_LENGTH: usize = 77; // CLIP's max sequence length

fn encode_texts(tokenizer: &Tokenizer, text_session: &Session, texts: &[&str]) -> Result<Array2<f32>> {
    let encodings = texts
        .iter()
        .map(|text| {
            tokenizer.encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {:?}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut input_ids = Array2::<i64>::zeros((texts.len(), MAX_LENGTH));
    let mut attention_mask = Array2::<i64>::zeros((texts.len(), MAX_LENGTH));

    for (i, encoding) in encodings.into_iter().enumerate() {
        let tokens = encoding.get_ids();
        let seq_len = tokens.len().min(MAX_LENGTH); // Ensure we don't exceed bounds

        input_ids
            .slice_mut(s![i, ..seq_len])
            .assign(&Array1::from_iter(tokens[..seq_len].iter().map(|&id| id as i64)));

        attention_mask
            .slice_mut(s![i, ..seq_len])
            .fill(1);
    }

    let input_ids_tensor = Tensor::from_array(input_ids.into_dyn())?;
    let attention_mask_tensor = Tensor::from_array(attention_mask.into_dyn())?;

    let outputs = text_session.run(vec![
        (
            "input_ids",
            SessionInputValue::from(input_ids_tensor),
        ),
        (
            "attention_mask",
            SessionInputValue::from(attention_mask_tensor),
        ),
    ])?;

    let mut embeddings = outputs["text_embeddings"]
        .try_extract_tensor::<f32>()?
        .into_dimensionality()?
        .to_owned();

    // Normalize embeddings to unit length
    let norms = embeddings.map_axis(ndarray::Axis(1), |row| row.dot(&row).sqrt());
    embeddings /= &norms.insert_axis(ndarray::Axis(1));
    
    Ok(embeddings)
}

pub fn text_main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {:?}", e))?;

    // Configure truncation and padding
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_LENGTH,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            stride: 0,
            direction: tokenizers::TruncationDirection::Right,
        }))
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation error: {:?}", e))?;

    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_LENGTH), // Fixed here
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

    let text_session = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(data_dir.join("clip_text.onnx"))
        .with_context(|| format!("Failed to load model from {:?}", data_dir.join("clip_text.onnx")))?;

    let texts = [
        "The weather outside is lovely.",
        "It's so sunny outside!",
        "She drove to the stadium."
    ];

    let text_embeddings = encode_texts(&tokenizer, &text_session, &texts)?;
    println!("Embeddings shape: {:?}", text_embeddings.shape());

    let query_embedding = text_embeddings.row(0);
    println!("\nQuery: {}", texts[0]);

    for (i, text) in texts.iter().enumerate().skip(1) {
        let target_embedding = text_embeddings.row(i);
        let similarity = query_embedding.dot(&target_embedding);
        println!("Similarity to '{}': {:.2}", text, similarity);
    }

    Ok(())
}