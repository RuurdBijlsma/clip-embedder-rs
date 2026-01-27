use color_eyre::eyre::{Result, eyre};
use image::imageops::FilterType;
use ndarray::{Array, Array4, Array2};
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const IMAGE_SIZE: u32 = 384;
const CONTEXT_LENGTH: usize = 64;
const BENCHMARK_ITERS: u32 = 10;

fn run_image_pipeline(visual_model: &mut Session) -> Result<Duration> {
    let start = Instant::now();

    // 1. Preprocess
    let img = image::open("assets/img/beach_rocks.jpg")?.to_rgb8();
    let (width, height) = img.dimensions();
    let scale = IMAGE_SIZE as f64 / (std::cmp::min(width, height) as f64);
    let nw = (width as f64 * scale).round() as u32;
    let nh = (height as f64 * scale).round() as u32;
    let img_scaled = image::imageops::resize(&img, nw, nh, FilterType::CatmullRom);
    let x_offset = ((nw as f32 - IMAGE_SIZE as f32) / 2.0).round() as u32;
    let y_offset = ((nh as f32 - IMAGE_SIZE as f32) / 2.0).round() as u32;
    let img_cropped = image::imageops::crop_imm(&img_scaled, x_offset, y_offset, IMAGE_SIZE, IMAGE_SIZE).to_image();

    let mut image_array = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
    for (x, y, pixel) in img_cropped.enumerate_pixels() {
        image_array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 127.5) - 1.0;
        image_array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 127.5) - 1.0;
        image_array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 127.5) - 1.0;
    }

    // 2. Inference
    let _ = visual_model.run(inputs!["pixel_values" => Tensor::from_array(image_array)?])?;

    Ok(start.elapsed())
}

fn run_text_pipeline(text_model: &mut Session, tokenizer: &Tokenizer) -> Result<Duration> {
    let start = Instant::now();

    // 1. Preprocess
    let text_input = "rocks in the rock business";
    let encoding = tokenizer.encode(text_input, false).map_err(|e| eyre!(e))?;
    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    ids.push(1);
    while ids.len() < CONTEXT_LENGTH { ids.push(1); }
    ids.truncate(CONTEXT_LENGTH);
    let text_array = Array2::from_shape_vec((1, CONTEXT_LENGTH), ids)?;

    // 2. Inference
    let _ = text_model.run(inputs!["input_ids" => Tensor::from_array(text_array)?])?;

    Ok(start.elapsed())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let dll_path = if cfg!(target_os = "windows") { "C:/Apps/onnxruntime/lib/onnxruntime.dll" } else { "libonnxruntime.so" };
    ort::init_from(dll_path)?
        .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().build()])
        .commit();

    let mut visual_model = Session::builder()?.commit_from_file("assets/model/visual.onnx")?;
    let mut text_model = Session::builder()?.commit_from_file("assets/model/text.onnx")?;
    let tokenizer = Tokenizer::from_file("assets/model/tokenizer.json").map_err(|e| eyre!(e))?;

    println!("Starting Warmup...");
    run_image_pipeline(&mut visual_model)?;
    run_text_pipeline(&mut text_model, &tokenizer)?;

    println!("Starting Benchmark ({} iterations)...", BENCHMARK_ITERS);

    let mut img_total = Duration::ZERO;
    for _ in 0..BENCHMARK_ITERS {
        img_total += run_image_pipeline(&mut visual_model)?;
    }

    let mut text_total = Duration::ZERO;
    for _ in 0..BENCHMARK_ITERS {
        text_total += run_text_pipeline(&mut text_model, &tokenizer)?;
    }

    println!("\n--- RUST RESULTS (AVG PER RUN) ---");
    println!("Image Pipeline: {:?}ms", (img_total / BENCHMARK_ITERS).as_millis());
    println!("Text Pipeline:  {:?}ms", (text_total / BENCHMARK_ITERS).as_millis());

    Ok(())
}