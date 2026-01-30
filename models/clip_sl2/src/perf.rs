use color_eyre::eyre::Result;
use ndarray::Axis;
use serde::Deserialize;
use std::fs;
use std::time::Instant;
use crate::{sigmoid, SigLipTextModel, SigLipVisionModel};

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

pub fn perf_main() -> Result<()> {
    color_eyre::install()?;

    let image_dir = get_asset("img");
    let model_dir = get_asset("model");
    let query_text = "A photo of Rocks";
    let image_path = std::path::Path::new(&image_dir).join("beach_rocks.jpg");

    // 1. Load Config
    let config_str = fs::read_to_string(format!("{}/model_config.json", model_dir))?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    // 2. Load Models
    println!("Loading models...");
    let mut vision_model = SigLipVisionModel::new(
        format!("{}/visual.onnx", model_dir),
        config.image_size
    )?;
    let mut text_model = SigLipTextModel::new(
        format!("{}/text.onnx", model_dir),
        format!("{}/tokenizer.json", model_dir),
        config.context_length
    )?;

    let sample_img = image::open(&image_path)?;

    // 3. WARMUP
    // ONNX Runtime often takes longer on the first run to allocate memory/init graphs
    println!("Warming up models...");
    {
        let img_tensor = vision_model.preprocess(&sample_img);
        let _ = vision_model.inference(img_tensor)?;
        let text_tensor = text_model.tokenize(query_text)?;
        let _ = text_model.inference(text_tensor)?;
    }

    println!("\n--- PERFORMANCE TIMING ---");

    // 4. Timed Vision Pipeline (1 Image)
    let start_v_pre = Instant::now();
    let img_tensor = vision_model.preprocess(&sample_img);
    let duration_v_pre = start_v_pre.elapsed();

    let start_v_inf = Instant::now();
    let img_emb = vision_model.inference(img_tensor)?;
    let duration_v_inf = start_v_inf.elapsed();

    println!("Vision Preprocessing:  {:>8.2}ms", duration_v_pre.as_secs_f64() * 1000.0);
    println!("Vision Inference:      {:>8.2}ms", duration_v_inf.as_secs_f64() * 1000.0);
    println!("Vision Total:          {:>8.2}ms", (duration_v_pre + duration_v_inf).as_secs_f64() * 1000.0);

    // 5. Timed Text Pipeline (1 String)
    let start_t_pre = Instant::now();
    let text_tensor = text_model.tokenize(query_text)?;
    let duration_t_pre = start_t_pre.elapsed();

    let start_t_inf = Instant::now();
    let text_emb = text_model.inference(text_tensor)?;
    let duration_t_inf = start_t_inf.elapsed();

    println!("Text Preprocessing:    {:>8.2}ms", duration_t_pre.as_secs_f64() * 1000.0);
    println!("Text Inference:        {:>8.2}ms", duration_t_inf.as_secs_f64() * 1000.0);
    println!("Text Total:            {:>8.2}ms", (duration_t_pre + duration_t_inf).as_secs_f64() * 1000.0);

    // 6. Final Logic (Similarity)
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let img_emb_vec = img_emb.index_axis(Axis(0), 0);
    let similarity = img_emb_vec.dot(&text_emb_vec);
    let prob = sigmoid(similarity * config.logit_scale + config.logit_bias);

    println!("\nResult for 'beach_rocks.jpg': {:.2}% match", prob * 100.0);

    Ok(())
}