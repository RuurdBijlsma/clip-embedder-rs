# open-clip-rs

Run https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384 CLIP (vision & text embedding) via onnx in rust.

## Prereqs:

* To get the onnx files: `uv` is needed.
* onnxruntime must be installed.
* Cargo & Rust

## Usage

### Get onnx files:

```shell
cd assets/model
uv run export_onnx.py
# Wait for it to finish generating the 5 files
```

### Run the program

```shell
cargo run --example run --release
```

## Troubleshooting

Onnxruntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime:

http://github.com/microsoft/onnxruntime/releases

And put the dll location in your `PATH`, or point the env var to it like this:

```shell
$env:ORT_DYLIB_PATH="C:/Apps/onnxruntime/lib/onnxruntime.dll"
```