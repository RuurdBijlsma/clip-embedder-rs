use color_eyre::eyre::{Result, eyre};
use image::imageops::FilterType;
use ndarray::{Array, Array4};
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

const IMAGE_SIZE: u32 = 384;
const CONTEXT_LENGTH: usize = 64;

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    color_eyre::install()?;

    let dll_path = if cfg!(target_os = "windows") {
        "C:/Apps/onnxruntime/lib/onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };

    ort::init_from(dll_path)?
        .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().build()])
        .commit();

    let mut visual_model = Session::builder()?.commit_from_file("assets/model/visual.onnx")?;
    let mut text_model = Session::builder()?.commit_from_file("assets/model/text.onnx")?;

    // 1. Preprocess Image
    info!("Preprocessing Image...");
    let img = image::open("assets/img/beach_rocks.jpg")?.to_rgb8();
    let (width, height) = img.dimensions();

    // MATCH PILLOW: Calculate dimensions with f64 and round
    let scale = IMAGE_SIZE as f64 / (std::cmp::min(width, height) as f64);
    let nw = (width as f64 * scale).round() as u32;
    let nh = (height as f64 * scale).round() as u32;

    // Note: CatmullRom is closest to Bicubic.
    // If parity is still off, try FilterType::Lanczos3 (which matches PIL's high-quality downsampling)
    let img_scaled = image::imageops::resize(&img, nw, nh, FilterType::CatmullRom);

    // MATCH TORCHVISION: Center Crop offsets
    // Python: int(round((image_height - crop_height) / 2.0))
    let x_offset = ((nw as f32 - IMAGE_SIZE as f32) / 2.0).round() as u32;
    let y_offset = ((nh as f32 - IMAGE_SIZE as f32) / 2.0).round() as u32;

    let img_cropped = image::imageops::crop_imm(
        &img_scaled,
        x_offset,
        y_offset,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to_image();

    let mut image_array = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
    for (x, y, pixel) in img_cropped.enumerate_pixels() {
        // SigLIP normalization: (pixel / 255.0 - 0.5) / 0.5 => (pixel / 127.5) - 1.0
        // We use f32 conversion early to maintain precision
        image_array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 127.5) - 1.0;
        image_array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 127.5) - 1.0;
        image_array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 127.5) - 1.0;
    }

    // 2. Preprocess Text (Fixed Padding Logic)
    info!("Preprocessing Text...");
    let tokenizer = Tokenizer::from_file("assets/model/tokenizer.json").map_err(|e| eyre!(e))?;
    let text_input = "rocks in the rock business";
    let encoding = tokenizer.encode(text_input, false).map_err(|e| eyre!(e))?;
    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

    // Pad with 1 to match Python
    ids.push(1); // EOS
    while ids.len() < CONTEXT_LENGTH {
        ids.push(1); // PAD
    }
    ids.truncate(CONTEXT_LENGTH);

    let text_array = Array::from_shape_vec((1, CONTEXT_LENGTH), ids)?;

    // 3. Run Inference
    let visual_outputs = visual_model.run(inputs!["pixel_values" => Tensor::from_array(image_array)?])?;
    let text_outputs = text_model.run(inputs!["input_ids" => Tensor::from_array(text_array)?])?;

    let image_emb = visual_outputs["image_embeddings"].try_extract_array::<f32>()?;
    let text_emb = text_outputs["text_embeddings"].try_extract_array::<f32>()?;

    println!("\nText Embedding [0:50]:");
    println!("{:?}", text_emb.slice(ndarray::s![0, 0..50]).to_vec());

    println!("Image Embedding [0:50]:");
    println!("{:?}", image_emb.slice(ndarray::s![0, 0..50]).to_vec());

    Ok(())
}