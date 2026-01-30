use clip_sl2::perf::perf_main;
use clip_sl2::{SigLipTextModel, SigLipVisionModel, sigmoid};
use color_eyre::eyre::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array2, ArrayView1, Axis};
use serde::Deserialize;
use std::fs;
use std::time::Instant;

const ASSETS_FOLDER: &str = "models/clip_sl2/assets";

fn get_asset(name: &str) -> String {
    format!("{ASSETS_FOLDER}/{name}")
}

#[derive(Deserialize)]
struct ModelConfig {
    logit_scale: f32,
    logit_bias: f32,
    image_size: u32,
    context_length: usize,
}

fn get_stats(data: ArrayView1<f32>) -> (f32, f32) {
    let mean = data.mean().unwrap_or(0.0);
    let var = data.fold(0.0, |acc, &x| acc + (x - mean).powi(2)) / data.len() as f32;
    (mean, var.sqrt())
}

fn save_debug_image(pix: &ndarray::Array4<f32>, path: &str) -> Result<()> {
    let height = pix.shape()[2];
    let width = pix.shape()[3];
    let mut img_buf = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = ((pix[[0, 0, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((pix[[0, 1, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((pix[[0, 2, y, x]] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img_buf.save(path)?;
    println!("Debug image saved to: {}", path);
    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let image_dir = get_asset("img");
    let model_dir = get_asset("model");
    let query_text = "A photo of Rocks";
    let image_files = vec![
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];

    let config_str = fs::read_to_string(format!("{}/model_config.json", model_dir))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    println!("Loading models...");
    let mut vision_model =
        SigLipVisionModel::new(format!("{}/visual.onnx", model_dir), config.image_size)?;
    let mut text_model = SigLipTextModel::new(
        format!("{}/text.onnx", model_dir),
        format!("{}/tokenizer.json", model_dir),
        config.context_length,
    )?;

    // Load a sample image for warmup and timing
    let sample_img_path = std::path::Path::new(&image_dir).join(&image_files[0]);
    let sample_img = image::open(sample_img_path)?;

    // --- 1. WARMUP ---
    println!("Warming up models...");
    {
        let v_pre = vision_model.preprocess(&sample_img);
        let _ = vision_model.inference(v_pre)?;
        let t_pre = text_model.tokenize(query_text)?;
        let _ = text_model.inference(t_pre)?;
    }

    // --- 2. TIMED PERFORMANCE RUN (Single item) ---
    println!("\n--- PERFORMANCE TIMING (Single Item) ---");

    let start_v_pre = Instant::now();
    let img_tensor = vision_model.preprocess(&sample_img);
    let dur_v_pre = start_v_pre.elapsed();

    let start_v_inf = Instant::now();
    let single_img_emb = vision_model.inference(img_tensor.clone())?;
    let dur_v_inf = start_v_inf.elapsed();

    let start_t_pre = Instant::now();
    let text_ids = text_model.tokenize(query_text)?;
    let dur_t_pre = start_t_pre.elapsed();

    let start_t_inf = Instant::now();
    let text_emb = text_model.inference(text_ids)?;
    let dur_t_inf = start_t_inf.elapsed();

    println!(
        "Vision Preproc: {:>8.2}ms",
        dur_v_pre.as_secs_f64() * 1000.0
    );
    println!(
        "Vision Infer:   {:>8.2}ms",
        dur_v_inf.as_secs_f64() * 1000.0
    );
    println!(
        "Text Preproc:   {:>8.2}ms",
        dur_t_pre.as_secs_f64() * 1000.0
    );
    println!(
        "Text Infer:     {:>8.2}ms",
        dur_t_inf.as_secs_f64() * 1000.0
    );

    // --- 3. PROCESS FULL BATCH ---
    println!("\nProcessing full batch of {} images...", image_files.len());
    let mut valid_names = Vec::new();
    let mut all_img_embs = Vec::new();

    for name in &image_files {
        let path = std::path::Path::new(&image_dir).join(name);
        if path.exists() {
            let img = image::open(path)?;
            let tensor = vision_model.preprocess(&img);
            let emb = vision_model.inference(tensor)?;
            all_img_embs.push(emb);
            valid_names.push(name.to_string());
        }
    }

    // Stack single embeddings into one Array2 (Batch, Dim)
    let dim = all_img_embs[0].shape()[1];
    let mut img_embs = Array2::<f32>::zeros((all_img_embs.len(), dim));
    for (i, emb) in all_img_embs.iter().enumerate() {
        img_embs.row_mut(i).assign(&emb.row(0));
    }

    // --- 4. DEBUG OUTPUT (SigLIP 2 parity) ---
    println!("\n--- DEBUG: ONNX VALUES ---");
    let ids = text_model.get_ids(query_text)?;
    println!(
        "Text Input IDs (first 10): {:?}",
        &ids[..std::cmp::min(10, ids.len())]
    );

    save_debug_image(&img_tensor, "debug_input_tensor.png")?;
    let (pix_mean, pix_std) = get_stats(img_tensor.view().into_shape_with_order(img_tensor.len())?);
    println!(
        "Image Pixel Values - Mean: {:.6}, Std: {:.6}",
        pix_mean, pix_std
    );
    println!(
        "Image Pixel Values (slice): {:?}",
        img_tensor.slice(ndarray::s![0, 0, 0, ..30]).to_vec()
    );

    let t_row = text_emb.row(0);
    let (t_mean, t_std) = get_stats(t_row);
    println!("Text Embeds[0] - Mean: {:.6}, Std: {:.6}", t_mean, t_std);

    let i_row = img_embs.row(0);
    let (i_mean, i_std) = get_stats(i_row);
    println!("Image Embeds[0] - Mean: {:.6}, Std: {:.6}", i_mean, i_std);

    // --- 5. RANK RESULTS ---
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let similarities = img_embs.dot(&text_emb_vec);

    let probs: Vec<f32> = similarities
        .iter()
        .map(|&sim| sigmoid(sim * config.logit_scale + config.logit_bias))
        .collect();

    let mut results: Vec<_> = valid_names.iter().zip(probs.iter()).collect();
    results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n--- SIGLIP 2 ONNX RESULTS ---");
    println!("Query: '{}'", query_text);
    for (i, (name, prob)) in results.iter().enumerate() {
        let marker = if i == 0 { "⭐ [BEST]" } else { "  " };
        println!("{} {}: {:.2}%", marker, name, *prob * 100.0);
    }

    let _ = perf_main();

    Ok(())
}
