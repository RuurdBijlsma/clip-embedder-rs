pub mod image_encoder;
pub mod search_images;
pub mod text_encoder;
pub mod utils;

use anyhow::Result;
use clip_openai_old::search_images::run_image_search;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("warn"))
        .init();

    text_encoder::run_text_encoding()?;
    println!();
    image_encoder::run_image_encoding()?;
    println!();
    run_image_search()?;

    Ok(())
}
