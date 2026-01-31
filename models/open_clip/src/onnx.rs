use color_eyre::eyre::{Context, OptionExt, Result};
use ndarray::{Array2, Array4, ArrayView, IxDyn};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::path::Path;

pub struct OnnxRunner {
    session: Session,
    // Cached input names derived from introspection
    input_ids_name: Option<String>,
    attention_mask_name: Option<String>,
    pixel_values_name: Option<String>,
}

impl OnnxRunner {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();

        // 1. Initialize Session
        let threads = num_cpus::get();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)
            .wrap_err_with(|| format!("Failed to load ONNX model at {:?}", path))?;

        // 2. Introspect Inputs
        // We look at the inputs defined in the loaded graph to determine naming.
        let inputs = session.inputs();

        let mut input_ids_name = None;
        let mut attention_mask_name = None;
        let mut pixel_values_name = None;

        for input in inputs {
            let name = input.name();
            match name {
                // Common names for text
                "input_ids" => input_ids_name = Some(name.to_string()),
                "attention_mask" => attention_mask_name = Some(name.to_string()),
                // Common names for vision
                "pixel_values" | "input" => pixel_values_name = Some(name.to_string()),
                _ => {}
            }
        }

        Ok(Self {
            session,
            input_ids_name,
            attention_mask_name,
            pixel_values_name,
        })
    }

    /// Runs the vision tower.
    /// Expects shape (Batch, 3, H, W).
    pub fn run_vision(&self, pixel_values: Array4<f32>) -> Result<Array2<f32>> {
        let input_name = self.pixel_values_name
            .as_deref()
            .ok_or_eyre("Model does not appear to accept image inputs (no 'pixel_values' or 'input' node found)")?;

        // 1. Create Input Tensor
        let input_tensor = Value::from_array(pixel_values)?;

        // 2. Run
        let outputs = self.session.run(ort::inputs![
            input_name => input_tensor
        ]?)?;

        // 3. Extract Output (Batch, EmbedDim)
        // We assume the first output is the embeddings.
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Convert dynamic shape to explicit 2D
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;

        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }

    /// Runs the text tower.
    /// Expects `input_ids` and optional `attention_mask`.
    ///
    /// If the ONNX model doesn't ask for an attention mask, the `mask` argument is ignored.
    pub fn run_text(&self, ids: Array2<i64>, mask: Array2<i64>) -> Result<Array2<f32>> {
        let id_name = self.input_ids_name
            .as_deref()
            .ok_or_eyre("Model does not appear to accept text inputs (no 'input_ids' node found)")?;

        let id_tensor = Value::from_array(ids)?;

        // 1. Build Inputs dynamically
        // We can't use the simple ort::inputs! macro easily with conditionals,
        // but we can construct the map directly or chain inputs.
        let outputs = if let Some(mask_name) = &self.attention_mask_name {
            // Case A: Model wants mask (Standard CLIP)
            let mask_tensor = Value::from_array(mask)?;
            self.session.run(ort::inputs![
                id_name => id_tensor,
                mask_name => mask_tensor
            ]?)?
        } else {
            // Case B: Model ignores mask (SigLIP)
            self.session.run(ort::inputs![
                id_name => id_tensor
            ]?)?
        };

        // 2. Extract Output
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let view = ArrayView::from_shape(IxDyn(&shape_usize), data)?;

        Ok(view.into_dimensionality::<ndarray::Ix2>()?.to_owned())
    }
}