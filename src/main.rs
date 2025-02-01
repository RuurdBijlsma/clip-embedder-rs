mod text_encoder;

use anyhow::Result;

fn main() -> Result<()> {
	text_encoder::text_main()?;

	Ok(())
}