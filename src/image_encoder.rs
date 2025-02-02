use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{s, Axis};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;

/// CLIP preprocessing constants.
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const IMAGE_SIZE: u32 = 224;

/// Preprocess an image: resize, center crop, and normalize.
/// Returns an image tensor in CHW format.
fn preprocess_image(img: DynamicImage) -> Result<ndarray::Array3<f32>> {
    let (w, h) = (img.width(), img.height());
    let target_size = IMAGE_SIZE;

    let (new_w, new_h) = if w < h {
        (
            target_size,
            (h as f32 * (target_size as f32 / w as f32)) as u32,
        )
    } else {
        (
            (w as f32 * (target_size as f32 / h as f32)) as u32,
            target_size,
        )
    };

    // Resize with CatmullRom filter.
    let resized = img.resize_exact(new_w, new_h, FilterType::CatmullRom);

    // Center crop.
    let left = (new_w - target_size) / 2;
    let top = (new_h - target_size) / 2;
    let cropped = resized.crop_imm(left, top, target_size, target_size);

    let img_array = ndarray::Array::from_shape_fn(
        (target_size as usize, target_size as usize, 3),
        |(y, x, c)| {
            let pixel = cropped.get_pixel(x as u32, y as u32);
            let channel = pixel.0[c];
            (channel as f32 / 255.0 - MEAN[c]) / STD[c]
        },
    );

    // Transpose to CHW (3, 224, 224).
    let chw_array = img_array
        .permuted_axes([2, 0, 1])
        .into_dimensionality::<ndarray::Ix3>()
        .context("Failed to convert array to CHW format")?;
    Ok(chw_array)
}

/// Compute cosine similarity between a query vector and a set of embedding vectors.
fn cosine_similarity(
    query: &ndarray::ArrayView2<f32>,
    embeddings: &ndarray::ArrayView2<f32>,
) -> ndarray::Array1<f32> {
    query.dot(&embeddings.t()).row(0).to_owned()
}

/// Encodes a batch of images using the provided ONNX session.
/// Returns a 2D array of embeddings.
pub fn encode_images(images: Vec<DynamicImage>, session: &Session) -> Result<ndarray::Array2<f32>> {
    let preprocessed = images
        .into_iter()
        .map(preprocess_image)
        .collect::<Result<Vec<_>>>()
        .context("Image preprocessing failed")?;

    // Stack images along the batch dimension.
    let batch = ndarray::stack(
        Axis(0),
        &preprocessed.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .context("Failed to stack image tensors")?;

    // Run inference.
    let input_tensor = Tensor::from_array(batch)?;
    let outputs = session.run(ort::inputs![input_tensor]?)?;
    let embeddings = outputs["image_embeddings"]
        .try_extract_tensor::<f32>()
        .context("Failed to extract embeddings")?;

    // Convert dynamic dimensions to a fixed 2D array.
    embeddings
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .context("Failed to convert embeddings to 2D")
}

/// Main entry point for image encoding.
pub fn run_image_encoding() -> Result<()> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let session = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(data_dir.join("clip_vision.onnx"))
        .with_context(|| {
            format!(
                "Failed to load model from {:?}",
                data_dir.join("clip_vision.onnx")
            )
        })?;

    // List of image file names.
    let image_names = [
        "beach_rocks.jpg",
        "beetle_car.jpg",
        "cat_face.jpg",
        "dark_sunset.jpg",
        "palace.jpg",
        "rocky_coast.jpg",
        "stacked_plates.jpg",
        "verdant_cliff.jpg",
    ];
    let images = image_names
        .iter()
        .map(|name| {
            let path = data_dir.join("imgs").join(name);
            image::ImageReader::open(&path)
                .with_context(|| format!("Failed to open image {}", path.display()))?
                .decode()
                .with_context(|| format!("Failed to decode image {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    
    let start = Instant::now();
    let embeddings = encode_images(images, &session).context("Failed to encode images")?;
    println!("Encoding images took: {:?}", start.elapsed());

    // Calculate similarities.
    let query = embeddings.slice(s![0, ..]).insert_axis(Axis(0));
    let others = embeddings.slice(s![1.., ..]);
    let similarities = cosine_similarity(&query, &others);

    // Print results.
    println!("Query: {}", image_names[0]);
    for (i, similarity) in similarities.iter().enumerate() {
        println!("\tSimilarity to {}: {:.2}", image_names[i + 1], similarity);
    }

    Ok(())
}
