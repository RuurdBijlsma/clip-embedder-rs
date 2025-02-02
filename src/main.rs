pub mod image_encoder;
pub mod text_encoder;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("warn")) // Only show warnings and errors
        .init();

    text_encoder::run_text_encoding()?;
    println!();
    image_encoder::run_image_encoding()?;

    Ok(())
}
