use criterion::{Criterion, criterion_group};
use open_clip::{TextEmbedder, VisionEmbedder};
use std::path::PathBuf;

const MODELS: &[&str] = &[
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "timm/MobileCLIP2-S2-OpenCLIP",
    "timm/vit_base_patch32_clip_224.openai",
    "timm/ViT-SO400M-16-SigLIP2-384",
];

fn benchmark_models(c: &mut Criterion) {
    let img_path = PathBuf::from("assets/img/beach_rocks.jpg");
    let img = image::open(img_path).expect("Failed to load benchmark image");
    let text = "A photo of rocks";

    for model_id in MODELS {
        let mut vision_embedder = VisionEmbedder::from_model_id(model_id)
            .unwrap_or_else(|_| panic!("Failed to load vision embedder for {model_id}"));
        let mut text_embedder = TextEmbedder::from_model_id(model_id)
            .unwrap_or_else(|_| panic!("Failed to load text embedder for {model_id}"));

        let mut group = c.benchmark_group(*model_id);
        group.sample_size(20);

        // 1. Vision Preprocessing
        group.bench_function("vision/preprocess", |b| {
            b.iter(|| vision_embedder.preprocess(&img));
        });

        // 2. Vision Full Embedding (Preprocess + Inference)
        group.bench_function("vision/embed", |b| {
            b.iter(|| vision_embedder.embed_image(&img));
        });

        // 4. Text Full Embedding (Tokenize + Inference)
        group.bench_function("text/embed", |b| {
            b.iter(|| text_embedder.embed_text(text));
        });

        group.finish();
    }
}

criterion_group!(benches, benchmark_models);

fn main() {
    let mut criterion = Criterion::default();
    let args: Vec<String> = std::env::args().collect();
    let has_ide_flags = args.iter().any(|arg| {
        arg.starts_with("--format")
            || arg.starts_with("-Z")
            || arg == "--show-output"
            || arg == "--no-fail-fast"
    });
    if !has_ide_flags {
        criterion = criterion.configure_from_args();
    }
    benchmark_models(&mut criterion);
}
