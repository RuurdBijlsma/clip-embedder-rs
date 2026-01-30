use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2};
use ort::session::SessionInputValue;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::path::{Path, PathBuf};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

pub struct TextEncoder {
    tokenizer: Tokenizer,
    text_session: Session,
    max_length: usize,
}

impl TextEncoder {
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        max_length: Option<usize>,
    ) -> Result<Self> {
        let max_length = max_length.unwrap_or(77);

        // Initialize tokenizer with truncation/padding
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("Tokenizer truncation error: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(max_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        // Configure session with execution providers
        let text_session = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        Ok(Self {
            tokenizer,
            text_session,
            max_length,
        })
    }

    pub fn encode(&self, texts: &[&str]) -> Result<Array2<f32>> {
        let encodings = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization error: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut input_ids = Array2::<i64>::zeros((texts.len(), self.max_length));
        let mut attention_mask = Array2::<i64>::zeros((texts.len(), self.max_length));

        for (i, encoding) in encodings.into_iter().enumerate() {
            // Get tokenizer-generated values (already padded/truncated)
            let tokens = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            input_ids
                .slice_mut(s![i, ..])
                .assign(&Array1::from_iter(tokens.iter().map(|&id| id as i64)));

            attention_mask
                .slice_mut(s![i, ..])
                .assign(&Array1::from_iter(mask.iter().map(|&val| val as i64)));
        }

        let input_ids_tensor = Tensor::from_array(input_ids.into_dyn())?;
        let attention_mask_tensor = Tensor::from_array(attention_mask.into_dyn())?;

        let outputs = self.text_session.run(vec![
            ("input_ids", SessionInputValue::from(input_ids_tensor)),
            (
                "attention_mask",
                SessionInputValue::from(attention_mask_tensor),
            ),
        ])?;

        let mut embeddings = outputs["text_embeddings"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality()?
            .to_owned();

        // Normalize embeddings
        let norms = embeddings.map_axis(ndarray::Axis(1), |row| row.dot(&row).sqrt());
        embeddings /= &norms.insert_axis(ndarray::Axis(1));

        Ok(embeddings)
    }
}

pub fn run_text_encoding() -> Result<()> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");

    let text_encoder = TextEncoder::new(
        data_dir.join("clip_text.onnx"),
        data_dir.join("tokenizer.json"),
        Some(77),
    )?;

    let texts = [
        "The weather outside is lovely.",
        "It's so sunny outside!",
        "She drove to the stadium.",
    ];

    let embeddings = text_encoder.encode(&texts)?;

    // Calculate similarities.
    let query_embedding = embeddings.row(0);
    println!("\nQuery: {}", texts[0]);

    for (i, text) in texts.iter().enumerate().skip(1) {
        let target_embedding = embeddings.row(i);
        let similarity = query_embedding.dot(&target_embedding);
        println!("\tSimilarity to '{}': {:.2}", text, similarity);
    }

    Ok(())
}
