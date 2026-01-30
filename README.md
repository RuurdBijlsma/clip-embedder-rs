# CLIP Embedder in Rust

Run siglip2 CLIP model via ONNX in Rust.
https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384

## Prereqs:

* To get the onnx files: `uv` is needed.
* onnxruntime must be installed.
* Cargo & Rust

## Usage

### Get onnx files:

```shell
cd models/clip_sl2
uv run export_onnx.py
# Wait for it to finish generating the 5 files
```

### Run the program

```shell
cargo run --package clip_sl2 --example search
```

```shell
cargo run --package clip_sl2 --example debug
```

```shell
cargo run --package clip_sl2 --example perf
```

## Troubleshooting

Onnxruntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime:

http://github.com/microsoft/onnxruntime/releases

And put the dll location in your `PATH`, or point the env var to it like this:

```shell
$env:ORT_DYLIB_PATH="C:/Apps/onnxruntime/lib/onnxruntime.dll"
```