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

    // 1. Preprocess Image (OpenCLIP Parity)
    info!("Preprocessing Image...");
    let img = image::open("assets/img/beach_rocks.jpg")?.to_rgb8();
    let (width, height) = img.dimensions();

    // Resize shortest edge to 384
    let (nw, nh) = if width < height {
        (IMAGE_SIZE, (height as f32 * (IMAGE_SIZE as f32 / width as f32)) as u32)
    } else {
        ((width as f32 * (IMAGE_SIZE as f32 / height as f32)) as u32, IMAGE_SIZE)
    };

    // Note: CatmullRom is the closest Rust equivalent to Pillow's Bicubic.
    let img_scaled = image::imageops::resize(&img, nw, nh, FilterType::CatmullRom);

    // Center Crop
    let x_offset = (nw - IMAGE_SIZE) / 2;
    let y_offset = (nh - IMAGE_SIZE) / 2;
    let img_cropped = image::imageops::crop_imm(&img_scaled, x_offset, y_offset, IMAGE_SIZE, IMAGE_SIZE).to_image();

    let mut image_array = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
    for (x, y, pixel) in img_cropped.enumerate_pixels() {
        // SigLIP: (pixel / 255.0 - 0.5) / 0.5
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

    // Pad with 1 to match Python [7832, 1, 1, 1...]
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