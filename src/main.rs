use color_eyre::eyre::{Result, eyre};
use image::GenericImageView;
use image::imageops::FilterType;
use ndarray::{Array, Array4};
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

// SigLIP Model Constants
const IMAGE_SIZE: u32 = 384;
const CONTEXT_LENGTH: usize = 64;

// SigLIP uses 0.5/0.5 mean/std (unlike standard OpenAI CLIP)
const NORM_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const NORM_STD: [f32; 3] = [0.5, 0.5, 0.5];

fn main() -> Result<()> {
    // 1. Setup Logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    color_eyre::install()?;

    info!("Initializing ORT environment (CPU Static)...");

    // We look in the current directory or a 'lib' folder.
    let dll_name = if cfg!(target_os = "windows") {
        "C:/Apps/onnxruntime/lib/onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    let dll_path = Path::new(dll_name);
    if !dll_path.exists() {
        return Err(eyre!(
            "Could not find '{}' in the current directory.\n\
            To fix this:\n\
            1. Download the ONNX Runtime GPU package for Windows (v1.19.0+ recommended).\n\
               Link: https://github.com/microsoft/onnxruntime/releases\n\
            2. Extract 'onnxruntime.dll' (and 'onnxruntime_providers_cuda.dll' for GPU) next to this executable/project root.",
            dll_name
        ));
    }
    // Initialize using the found DLL
    ort::init_from(dll_path.to_str().unwrap())?
        .with_execution_providers([
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .commit();

    // 3. Load Models
    let visual_path = Path::new("assets/model/visual.onnx");
    let text_path = Path::new("assets/model/text.onnx");
    let tokenizer_path = Path::new("assets/model/tokenizer.json");

    if !visual_path.exists() || !text_path.exists() || !tokenizer_path.exists() {
        return Err(eyre!(
            "Model files missing! Please ensure assets/model/ contains visual.onnx, text.onnx, and tokenizer.json"
        ));
    }

    info!("Loading Visual Model...");
    let mut visual_model = Session::builder()?.commit_from_file(visual_path)?;

    info!("Loading Text Model...");
    let mut text_model = Session::builder()?.commit_from_file(text_path)?;

    // 4. Preprocess Image
    info!("Preprocessing Image...");
    let image_path = Path::new("assets/img/beach_rocks.jpg");
    // Check if image exists before opening to give a clear error
    if !image_path.exists() {
        return Err(eyre!("Image file not found at {:?}", image_path));
    }

    let img = image::open(image_path).map_err(|e| eyre!("Failed to open image: {}", e))?;

    // Resize (Keep Ratio) + Center Crop
    let img_resized = img.resize_to_fill(IMAGE_SIZE, IMAGE_SIZE, FilterType::CatmullRom);

    // Normalize and Convert to CHW Tensor
    let mut image_array = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
    for (x, y, pixel) in img_resized.pixels() {
        let r = (pixel[0] as f32 / 255.0 - NORM_MEAN[0]) / NORM_STD[0];
        let g = (pixel[1] as f32 / 255.0 - NORM_MEAN[1]) / NORM_STD[1];
        let b = (pixel[2] as f32 / 255.0 - NORM_MEAN[2]) / NORM_STD[2];

        image_array[[0, 0, y as usize, x as usize]] = r;
        image_array[[0, 1, y as usize, x as usize]] = g;
        image_array[[0, 2, y as usize, x as usize]] = b;
    }

    // 5. Preprocess Text
    info!("Preprocessing Text...");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| eyre!(e))?;

    let padding = PaddingParams {
        strategy: PaddingStrategy::Fixed(CONTEXT_LENGTH),
        pad_id: 1,
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: CONTEXT_LENGTH,
        ..Default::default()
    }));

    let text_input = "rocks!";
    let encoding = tokenizer.encode(text_input, true).map_err(|e| eyre!(e))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let text_array = Array::from_shape_vec((1, CONTEXT_LENGTH), ids)?;

    // 6. Run Inference
    info!("Running Inference...");

    let image_tensor = Tensor::from_array(image_array)?;
    let text_tensor = Tensor::from_array(text_array)?;

    let visual_outputs = visual_model.run(inputs![
        "pixel_values" => image_tensor
    ])?;

    let text_outputs = text_model.run(inputs![
        "input_ids" => text_tensor
    ])?;

    // 7. Extract and Print Embeddings
    let image_emb = visual_outputs["image_embeddings"].try_extract_array::<f32>()?;
    let text_emb = text_outputs["text_embeddings"].try_extract_array::<f32>()?;

    println!("\nText Embedding [0:50]:");
    let text_slice = text_emb.slice(ndarray::s![0, 0..50]);
    println!("{:?}", text_slice.iter().collect::<Vec<_>>());

    println!("Image Embedding [0:50]:");
    let image_slice = image_emb.slice(ndarray::s![0, 0..50]);
    println!("{:?}", image_slice.iter().collect::<Vec<_>>());

    Ok(())
}