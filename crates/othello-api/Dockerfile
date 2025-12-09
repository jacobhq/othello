FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY othello ./othello
COPY othello-api ./othello-api
WORKDIR /app/othello-api
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG JWT_SECRET
ARG DATABASE_URL
ARG FRONTEND_URL
ARG BACKEND_URL
ARG APEX_COOKIE_DOMAIN
ARG BACKEND_COOKIE_DOMAIN
ARG POSTHOG_API_KEY
ARG AXIOM_DATASET
ARG AXIOM_API_TOKEN

ENV JWT_SECRET=${JWT_SECRET}
ENV DATABASE_URL=${DATABASE_URL}
ENV FRONTEND_URL=${FRONTEND_URL}
ENV BACKEND_URL=${BACKEND_URL}
ENV APEX_COOKIE_DOMAIN=${APEX_COOKIE_DOMAIN}
ENV BACKEND_COOKIE_DOMAIN=${BACKEND_COOKIE_DOMAIN}
ENV POSTHOG_API_KEY=${POSTHOG_API_KEY}
ENV AXIOM_DATASET=${AXIOM_DATASET}
ENV AXIOM_API_TOKEN=${AXIOM_API_TOKEN}

WORKDIR /app/othello-api
COPY --from=planner /app/othello-api/recipe.json recipe.json
COPY othello /app/othello
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
COPY othello-api .
RUN cargo build --release --bin othello-api

# We do not need the Rust toolchain to run the binary!
FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/othello-api/target/release/othello-api /usr/local/bin
ENTRYPOINT ["/usr/local/bin/othello-api"]