use anyhow::Result;
use ndarray::{s, Array1, Array2, ArrayView1};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::Session,
    value::Tensor,
};
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_LENGTH: usize = 77; // CLIP's max sequence length

fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot_product = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    dot_product / (norm_a * norm_b)
}

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
        let seq_len = tokens.len();

        let tokens_array = Array1::from_iter(
            tokens.iter()
                .map(|&id| id as i64)
        );

        input_ids
            .slice_mut(s![i, ..seq_len])
            .assign(&tokens_array);

        attention_mask
            .slice_mut(s![i, ..seq_len])
            .fill(1);
    }

    let input_ids_tensor = Tensor::from_array(input_ids.into_dyn())?;
    let attention_mask_tensor = Tensor::from_array(attention_mask.into_dyn())?;

    let outputs = text_session.run(vec![
        (
            // Explicit type for string key
            std::borrow::Cow::<str>::Borrowed("input_ids").into_owned(),
            // Explicit conversion for tensor value
            ort::session::SessionInputValue::from(input_ids_tensor)
        ),
        (
            std::borrow::Cow::<str>::Borrowed("attention_mask").into_owned(),
            ort::session::SessionInputValue::from(attention_mask_tensor)
        ),
    ])?;

    let mut embeddings = outputs["text_embeddings"]
        .try_extract_tensor::<f32>()?
        .into_dimensionality()?
        .to_owned();

    // Normalize embeddings to unit length
    for mut row in embeddings.rows_mut() {
        let norm = row.dot(&row).sqrt();
        row /= norm;
    }

    Ok(embeddings)
}

pub fn text_main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
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

    let text_session = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(data_dir.join("clip_text.onnx"))?;

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
        let similarity = cosine_similarity(query_embedding, target_embedding);
        println!("Similarity to '{}': {:.2}", text, similarity);
    }

    Ok(())
}