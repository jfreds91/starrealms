# One image: sr-server plus the built web client and downloaded card art.
#
#   docker build -t starrealms .
#   docker run -p 3030:3030 -v starrealms-data:/data -v ./models:/models:ro starrealms
#
# Card images are fetched from starrealms.com while building; they belong to
# Wise Wizard Games, so do not push this image to a public registry.

FROM rust:1-bookworm AS rust
ARG WASM_BINDGEN=0.2.129
RUN rustup target add wasm32-unknown-unknown \
 && cargo install --locked wasm-bindgen-cli --version ${WASM_BINDGEN}
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY data data
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p sr-server \
 && cargo build --release -p sr-wasm --target wasm32-unknown-unknown \
 && wasm-bindgen --target web --out-dir /out/wasm --out-name engine \
      target/wasm32-unknown-unknown/release/sr_wasm.wasm \
 && cp target/release/sr-server /out/sr-server

FROM node:24-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
COPY --from=rust /out/wasm src/wasm
RUN npx tsc --noEmit && npx vite build

FROM python:3.12-slim AS assets
WORKDIR /src
COPY tools/fetch_assets.py tools/
COPY data data
RUN python tools/fetch_assets.py

FROM debian:bookworm-slim
RUN useradd --system --home /data starrealms && mkdir -p /data /models && chown starrealms /data
COPY --from=rust /out/sr-server /usr/local/bin/sr-server
COPY --from=web /src/web/dist /app/web
COPY --from=assets /src/assets /app/assets
COPY --chown=starrealms:starrealms models /models
USER starrealms
ENV SR_BIND=0.0.0.0:3030 \
    SR_DB=/data/starrealms.db \
    SR_WEB=/app/web \
    SR_ASSETS=/app/assets \
    SR_MODELS=/models
EXPOSE 3030
VOLUME /data
CMD ["sr-server"]
