# CLIP Embedder

* To run this, you need to put the onnx file for [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) at `data/clip_text.onnx`.
* You can generate this onnx file with the following Python script:

Run [create_onnx.py](data/create_onnx.py) with `uv run create_onnx.py`, [uv](https://docs.astral.sh/uv/getting-started/installation/) is required for this.

## Quality checks

```shell
cargo test --all-targets --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --
```