FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG JWT_SECRET
ARG DATABASE_URL
ARG FRONTEND_URL
ARG BACKEND_URL
ARG APEX_COOKIE_DOMAIN
ARG BACKEND_COOKIE_DOMAIN

ENV JWT_SECRET=${JWT_SECRET}
ENV DATABASE_URL=${DATABASE_URL}
ENV FRONTEND_URL=${FRONTEND_URL}
ENV BACKEND_URL=${BACKEND_URL}
ENV APEX_COOKIE_DOMAIN=${APEX_COOKIE_DOMAIN}
ENV BACKEND_COOKIE_DOMAIN=${BACKEND_COOKIE_DOMAIN}

COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
RUN cargo build --release --bin othello-api

# We do not need the Rust toolchain to run the binary!
FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=builder /app/target/release/othello-api /usr/local/bin
ENTRYPOINT ["/usr/local/bin/othello-api"]
