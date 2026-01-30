use clip_sl2::{sigmoid, ModelConfig, SigLipTextModel, SigLipVisionModel};
use color_eyre::eyre::Result;
use ndarray::Axis;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    color_eyre::install()?;

    let assets = Path::new("models/clip_sl2/assets");
    let model_dir = assets.join("model");
    let img_path = assets.join("img/beach_rocks.jpg");

    let config = ModelConfig::from_file(model_dir.join("model_config.json"))?;

    println!("Loading models...");
    let mut vision = SigLipVisionModel::new(model_dir.join("visual.onnx"), config.clone())?;
    let mut text = SigLipTextModel::new(
        model_dir.join("text.onnx"),
        model_dir.join("tokenizer.json"),
        config.clone()
    )?;

    let sample_img = image::open(&img_path)?;
    let query_text = "A photo of Rocks";

    // --- WARMUP ---
    println!("Warming up models...");
    {
        let img_tensor = vision.preprocess(&sample_img);
        let _ = vision.inference(img_tensor)?;
        let text_tensor = text.tokenize(query_text)?;
        let _ = text.inference(text_tensor)?;
    }

    println!("\n--- PERFORMANCE TIMING ---");

    // --- Vision Pipeline Timing ---
    let start_v_pre = Instant::now();
    let img_tensor = vision.preprocess(&sample_img);
    let duration_v_pre = start_v_pre.elapsed();

    let start_v_inf = Instant::now();
    let img_emb = vision.inference(img_tensor)?;
    let duration_v_inf = start_v_inf.elapsed();

    println!("Vision Preprocessing:  {:>8.2}ms", duration_v_pre.as_secs_f64() * 1000.0);
    println!("Vision Inference:      {:>8.2}ms", duration_v_inf.as_secs_f64() * 1000.0);
    println!("Vision Total:          {:>8.2}ms", (duration_v_pre + duration_v_inf).as_secs_f64() * 1000.0);

    // --- Text Pipeline Timing ---
    let start_t_pre = Instant::now();
    let text_tensor = text.tokenize(query_text)?;
    let duration_t_pre = start_t_pre.elapsed();

    let start_t_inf = Instant::now();
    let text_emb = text.inference(text_tensor)?;
    let duration_t_inf = start_t_inf.elapsed();

    println!("Text Preprocessing:    {:>8.2}ms", duration_t_pre.as_secs_f64() * 1000.0);
    println!("Text Inference:        {:>8.2}ms", duration_t_inf.as_secs_f64() * 1000.0);
    println!("Text Total:            {:>8.2}ms", (duration_t_pre + duration_t_inf).as_secs_f64() * 1000.0);

    // --- Result Calculation ---
    let text_emb_vec = text_emb.index_axis(Axis(0), 0);
    let img_emb_vec = img_emb.index_axis(Axis(0), 0);
    let similarity = img_emb_vec.dot(&text_emb_vec);
    let prob = sigmoid(similarity * config.logit_scale + config.logit_bias);

    println!("\nResult match: {:.2}%", prob * 100.0);

    Ok(())
}