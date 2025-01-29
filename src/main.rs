use std::path::Path;

use ndarray::{Array2, Axis, Ix3};
use ort::{
	Error,
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel}
};
use tokenizers::Tokenizer;

fn main() -> ort::Result<()> {
	tracing_subscriber::fmt::init();

	ort::init()
		.with_name("clip")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	// Load the CLIP text encoder ONNX model
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file("data/clip_text_encoder.onnx")?;

	// Load the CLIP tokenizer
	let tokenizer = Tokenizer::from_file(Path::new("data/tokenizer.json")).unwrap();

	let inputs = vec!["The weather outside is lovely.", "It's so sunny outside!", "She drove to the stadium."];

	// Encode the input strings
	let encodings = tokenizer.encode_batch(inputs.clone(), false).map_err(|e| Error::new(e.to_string()))?;

	// Get the padded length of each encoding
	let padded_token_length = encodings[0].len();

	// Get token IDs and attention mask
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();

	// Convert to 2D arrays
	let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
	let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

	// Run the model
	let outputs = session.run(ort::inputs![a_ids, a_mask]?)?;

	// Extract and pool the embeddings
	// Extract the `last_hidden_state` tensor (shape: [batch_size, sequence_length, hidden_size])
	let last_hidden_state = outputs[0]
		.try_extract_tensor::<f32>()?
		.into_dimensionality::<Ix3>()
		.map_err(|e| ort::Error::new(e.to_string()))?;	let embeddings = last_hidden_state.map_axis(Axis(1), |row| row.mean().unwrap());

	// Normalize and compute cosine similarity
	println!("Similarity for '{}'", inputs[0]);
	// Normalize the query embedding
	let query = embeddings.index_axis(Axis(0), 0).to_owned(); // Convert to owned array
	let query_norm = query.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
	let query_normalized = query / query_norm;

	// Normalize and compute cosine similarity for other embeddings
	for (embeddings, sentence) in embeddings.axis_iter(Axis(0)).zip(inputs.iter()).skip(1) {
		let embeddings = embeddings.to_owned(); // Convert to owned array
		let embedding_norm = embeddings.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
		let embedding_normalized = embeddings / embedding_norm;

		let dot_product: f32 = query_normalized.iter().zip(embedding_normalized.iter()).map(|(a, b)| a * b).sum();
		println!("\t'{}': {:.1}%", sentence, dot_product * 100.);
	}

	Ok(())
}