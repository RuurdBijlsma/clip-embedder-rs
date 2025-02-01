use anyhow::Result;
use ndarray::{s, Array1};
use ndarray::{Array2, ArrayView2};
use ort::{
    session::Session,
    value::Tensor,
};
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_LENGTH: usize = 77; // CLIP's max sequence length

fn cosine_similarity(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> f32 {
    let dot_product = a.dot(&b.t()).sum();
    let norm_a = a.dot(&a.t()).sum().sqrt();
    let norm_b = b.dot(&b.t()).sum().sqrt();
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
        let tokens = &encoding.get_ids()[..encoding.get_ids().len().min(MAX_LENGTH)];
        let seq_len = tokens.len();

        // Create 1D array of i64 tokens
        let tokens_array = Array1::from_iter(
            tokens.iter()
                .map(|&id| id as i64)
        );

        // Assign to 2D slice
        input_ids
            .slice_mut(s![i, ..seq_len])
            .assign(&tokens_array);

        attention_mask
            .slice_mut(s![i, ..seq_len])
            .fill(1);
    }

    // Create tensors with explicit i64 type
    let input_ids_tensor = Tensor::from_array(input_ids.clone().into_dyn())?;
    let attention_mask_tensor = Tensor::from_array(attention_mask.clone().into_dyn())?;

    let inputs: Vec<(std::borrow::Cow<'static, str>, ort::session::SessionInputValue<'_>)> = vec![
        ("input_ids".into(), input_ids_tensor.into()),
        ("attention_mask".into(), attention_mask_tensor.into()),
    ];

    let outputs = text_session.run(inputs)?;

    let embeddings = outputs["text_embeddings"]
        .try_extract_tensor::<f32>()?
        .into_dimensionality()?
        .to_owned();

    Ok(embeddings)
}

pub async fn text_main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().init();

    // Load resources
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("Tokenizer error: {:?}", e))?;
    let text_session = Session::builder()?.commit_from_file(data_dir.join("clip_text.onnx"))?;

    // Example texts
    let texts = [
        "The weather outside is lovely.",
        "It's so sunny outside!",
        "She drove to the stadium."
    ];

    // Get embeddings
    let text_embeddings = encode_texts(&tokenizer, &text_session, &texts)?;
    println!("Embeddings shape: {:?}", text_embeddings.shape());

    // Calculate similarities
    let query_embedding = text_embeddings.slice(s![0..1, ..]);
    println!("\nQuery: {}", texts[0]);

    for (i, text) in texts.iter().enumerate().skip(1) {
        let target_embedding = text_embeddings.slice(s![i..i+1, ..]);
        let similarity = cosine_similarity(&query_embedding, &target_embedding);
        println!("Similarity to '{}': {:.2}", text, similarity);
    }

    Ok(())
}