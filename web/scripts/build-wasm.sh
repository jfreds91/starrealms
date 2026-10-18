#!/bin/sh
# Compile sr-wasm and generate its JS bindings into web/src/wasm/ (gitignored).
set -eu
cd "$(dirname "$0")/../.."
cargo build -p sr-wasm --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir web/src/wasm --out-name engine \
  target/wasm32-unknown-unknown/release/sr_wasm.wasm
