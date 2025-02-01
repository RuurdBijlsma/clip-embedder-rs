mod text_encoder;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
	text_encoder::text_main().await?;

	Ok(())
}