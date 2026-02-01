use crate::config::{OnnxModelConfig, OpenClipConfig};
use crate::error::ClipError;
use crate::onnx::OnnxSession;
use ndarray::Array2;
use ort::value::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub struct TextEmbedder {
    pub session: OnnxSession,
    pub config: OpenClipConfig,
    pub model_config: OnnxModelConfig,
    tokenizer: Tokenizer,
    id_name: String,
    mask_name: Option<String>,
}

impl TextEmbedder {
    // todo: use bon and let user set cache folder+model_id, or model folder directly
    pub fn new(model_id: &str) -> Result<Self, ClipError> {
        let model_dir = OnnxSession::get_model_dir(model_id)?;
        let model_path = model_dir.join("text.onnx");
        let config_path = model_dir.join("open_clip_config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_config_path = model_dir.join("model_config.json");

        let model_config = OnnxModelConfig::from_file(model_config_path)?;
        let session = OnnxSession::new(model_path)?;
        let config = OpenClipConfig::from_file(config_path)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let pad_id = model_config
            .pad_id
            .or_else(|| tokenizer.get_vocab(true).get("<pad>").copied())
            .ok_or_else(|| ClipError::Config("No pad token found in tokenizer".into()))?;
        let ctx_len = config.model_cfg.text_cfg.context_length;

        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(ctx_len),
                pad_id,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: ctx_len,
                ..Default::default()
            }))
            .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let id_name = session
            .find_input(&["input_ids"])
            .ok_or_else(|| ClipError::Config("Could not find text input node".into()))?;
        let mask_name = session.find_input(&["attention_mask"]);

        Ok(Self {
            session,
            config,
            model_config,
            tokenizer,
            id_name,
            mask_name,
        })
    }

    pub fn tokenize<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> Result<(Array2<i64>, Array2<i64>), ClipError> {
        let encodings = if self.model_config.tokenizer_needs_lowercase {
            let lowered = texts.iter().map(|s| s.as_ref().to_lowercase()).collect();
            self.tokenizer.encode_batch(lowered, true)
        } else {
            let texts = texts.iter().map(AsRef::as_ref).collect();
            self.tokenizer.encode_batch(texts, true)
        }
        .map_err(|e| ClipError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = self.config.model_cfg.text_cfg.context_length;

        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&x| i64::from(x)))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&x| i64::from(x)))
            .collect();

        let ids_array = Array2::from_shape_vec((batch_size, seq_len), ids)
            .map_err(|e| ClipError::Inference(e.to_string()))?;
        let mask_array = Array2::from_shape_vec((batch_size, seq_len), mask)
            .map_err(|e| ClipError::Inference(e.to_string()))?;

        Ok((ids_array, mask_array))
    }

    pub fn embed_text(&mut self, text: &str) -> Result<ndarray::Array1<f32>, ClipError> {
        let embs = self.embed_texts(&[text])?;
        let len = embs.len();
        embs.into_shape_with_order(len)
            .map_err(|e| ClipError::Inference(e.to_string()))
    }

    pub fn embed_texts<T: AsRef<str>>(&mut self, texts: &[T]) -> Result<Array2<f32>, ClipError> {
        let (ids_tensor, mask_tensor) = self.tokenize(texts)?;

        let ort_ids = Value::from_array(ids_tensor)?;
        let outputs = if let Some(m_name) = &self.mask_name {
            let ort_mask = Value::from_array(mask_tensor)?;
            self.session
                .session
                .run(ort::inputs![&self.id_name => ort_ids, m_name => ort_mask])?
        } else {
            self.session
                .session
                .run(ort::inputs![&self.id_name => ort_ids])?
        };

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let view = ndarray::ArrayView::from_shape(ndarray::IxDyn(&shape_usize), data)
            .map_err(|e| ClipError::Inference(e.to_string()))?;
        Ok(view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ClipError::Inference(e.to_string()))?
            .to_owned())
    }
}
