use crate::config::{ModelConfig, ModelType, SpecialTokensMap};
use color_eyre::eyre::{Context, OptionExt, Result};
use ndarray::Array2;
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Debug)]
pub struct TextProcessor {
    tokenizer: Tokenizer,
    lowercase: bool,
}

impl TextProcessor {
    /// Initializes the tokenizer with padding and truncation rules derived from the config.
    ///
    /// Arguments:
    /// - `tokenizer_path`: Path to `tokenizer.json`
    /// - `tokens_map_path`: Path to `special_tokens_map.json`
    /// - `config`: The parsed `open_clip_config.json`
    pub fn new(
        tokenizer_path: impl AsRef<Path>,
        tokens_map_path: impl AsRef<Path>,
        config: &ModelConfig,
    ) -> Result<Self> {
        // 1. Load the base Tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).expect(&format!(
            "Failed to load tokenizer from {:?}",
            tokenizer_path.as_ref()
        ));

        // 2. Load Special Tokens Map to find the Pad Token string
        let map = SpecialTokensMap::from_file(&tokens_map_path)
            .wrap_err("Failed to parse special_tokens_map.json")?;

        // 3. Determine Pad Token ID
        // We look for the string in the map (e.g., "<pad>" or "<|endoftext|>")
        // then ask the tokenizer for the corresponding ID.
        let pad_token_str = map
            .pad_token
            .map(|t| t.content)
            .ok_or_eyre("No 'pad_token' found in special_tokens_map.json")?;

        let pad_id = tokenizer.token_to_id(&pad_token_str).ok_or_eyre(format!(
            "Tokenizer does not contain the pad token ID for string: '{}'",
            pad_token_str
        ))?;

        // 4. Configure Padding & Truncation
        // We pad to the exact context length (e.g. 64 for SigLIP, 77 for CLIP)
        let context_len = config.context_length;

        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(context_len),
                pad_id,
                pad_token: pad_token_str,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: context_len,
                ..Default::default()
            }))
            .map_err(|e| color_eyre::eyre::eyre!("Failed to set tokenizer params: {e}"))?;

        let lowercase = match config.model_type {
            ModelType::Siglip => true,
            ModelType::Clip => false,
        };

        Ok(Self {
            tokenizer,
            lowercase,
        })
    }

    /// Tokenizes a single string.
    /// Returns (input_ids, attention_mask).
    /// Shape: (1, SequenceLength)
    pub fn process(&self, text: &str) -> Result<(Array2<i64>, Array2<i64>)> {
        // Note: The tokenizer.json usually handles normalization (lowercasing etc) internally
        // via its Normalizer pipeline, so we don't need to manually lowercase here
        // unless the specific model config demands it outside the tokenizer pipeline.
        // For standard OpenCLIP/HF exports, the generic encode is sufficient.
        let cased_text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_owned()
        };
        let encoding = self
            .tokenizer
            .encode(cased_text, true)
            .map_err(|e| color_eyre::eyre::eyre!("Tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| i64::from(x)).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| i64::from(x))
            .collect();

        let len = ids.len();

        let ids_array =
            Array2::from_shape_vec((1, len), ids).wrap_err("Failed to create input_ids tensor")?;

        let mask_array = Array2::from_shape_vec((1, len), mask)
            .wrap_err("Failed to create attention_mask tensor")?;

        Ok((ids_array, mask_array))
    }

    /// Tokenizes a batch of strings.
    /// Returns (input_ids, attention_mask).
    /// Shape: (Batch, SequenceLength)
    pub fn process_batch(&self, texts: &[String]) -> Result<(Array2<i64>, Array2<i64>)> {
        if texts.is_empty() {
            return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
        }

        let cased_texts: Vec<String> = if self.lowercase {
            texts.iter().map(|s|s.to_lowercase()).collect()
        } else {
            texts.to_vec()
        };

        let encodings = self
            .tokenizer
            .encode_batch(cased_texts, true)
            .map_err(|e| color_eyre::eyre::eyre!("Batch tokenization failed: {e}"))?;

        // Calculate dimensions
        let batch_size = encodings.len();
        let seq_len = encodings.first().map(|e| e.len()).unwrap_or(0);

        let mut flat_ids = Vec::with_capacity(batch_size * seq_len);
        let mut flat_mask = Vec::with_capacity(batch_size * seq_len);

        for encoding in encodings {
            flat_ids.extend(encoding.get_ids().iter().map(|&x| i64::from(x)));
            flat_mask.extend(encoding.get_attention_mask().iter().map(|&x| i64::from(x)));
        }

        let ids_array = Array2::from_shape_vec((batch_size, seq_len), flat_ids)
            .wrap_err("Failed to create batch input_ids tensor")?;

        let mask_array = Array2::from_shape_vec((batch_size, seq_len), flat_mask)
            .wrap_err("Failed to create batch attention_mask tensor")?;

        Ok((ids_array, mask_array))
    }

    /// Helper to get the underlying tokenizer if needed (e.g. for decoding)
    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
