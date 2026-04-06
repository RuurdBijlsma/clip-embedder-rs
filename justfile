set windows-shell := ["powershell.exe", "-Command"]
set export := true
ORT_DYLIB_PATH := "C:/Apps/onnxruntime/lib/onnxruntime.dll"

# --- Lints:

check: fmt clippy test

fmt:
    cargo fmt --all

clippy:
    cargo clippy --no-deps --all-features --tests --benches -- \
        -D clippy::all \
        -D clippy::pedantic \
        -D clippy::nursery

# --- Misc:

clean:
    cargo clean

# --- Execution:

test:
    cargo test --profile release --features load-dynamic -- --nocapture

bench:
    cargo bench --features load-dynamic

run:
    cargo run --example search --profile release --features load-dynamic

