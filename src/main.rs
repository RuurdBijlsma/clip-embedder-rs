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
		.with_optimization_level(GraphOptimizationLevel::Level3)?
		.with_intra_threads(1)?
		.commit_from_file("data/clip_text_encoder.onnx")?;

	// Load the CLIP tokenizer
	let tokenizer = Tokenizer::from_file(Path::new("data/tokenizer.json")).unwrap();

	let inputs = vec!["The weather outside is lovely.", "It's so sunny outside!", "She drove to the stadium."];

	// Encode the input strings
	let encodings = tokenizer.encode_batch(inputs.clone(), false).map_err(|e| Error::new(e.to_string()))?;

	// Get MAX sequence length across all inputs
	let padded_token_length = encodings.iter().map(|e| e.len()).max().unwrap();

	// Add 2 for [SOS] and [EOS]
	let total_length = padded_token_length + 2;

	// Define special tokens
	let start_token = 49406;
	let end_token = 49407;

	// Modify ids and mask extraction to include special tokens
	// Remove manual [SOS] addition - CLIP typically only uses [EOS]
	let ids: Vec<i64> = encodings.iter().flat_map(|e| {
		let mut tokens = e.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();
		tokens.push(end_token);  // Only add [EOS]
		tokens
	}).collect();

	let mask: Vec<i64> = encodings.iter().flat_map(|e| {
		let mut attention_mask = vec![1]; // Start token should have attention 1
		attention_mask.extend(e.get_attention_mask().iter().map(|i| *i as i64)); // Original attention mask
		attention_mask.push(1); // End token should have attention 1
		attention_mask
	}).collect();

	// Convert to 2D arrays
	let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length + 2], ids).unwrap();
	let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length + 2], mask).unwrap();

	println!("ids: {:?}", a_ids);
	println!("mask: {:?}", a_mask);

	// Run the model (now outputs projected embeddings)
	let outputs = session.run(ort::inputs![a_ids, a_mask]?)?;

	// Directly extract text embeddings (already projected)
	let embeddings = outputs[0]
		.try_extract_tensor::<f32>()?
		.into_dimensionality::<ndarray::Ix2>()?;

	println!("Projected embeddings: {:?}", embeddings);


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