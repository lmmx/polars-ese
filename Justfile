import ".just/ship.just"
import ".just/bench.just"

set shell := ["bash", "-cu"]

default:
    @just --list

build:
    cargo build

# Release build
build-release:
    cargo build --release

check:
    cargo check

# Build the Python wheel via maturin
wheel:
    maturin build --release --uv

# Install into the current venv for smoke-testing the Python entry point
develop:
    maturin develop --uv

# Install into the current venv for smoke-testing the Python entry point
develop-release:
    maturin develop --release --uv

clippy:
    cargo clippy --all-targets -- -D warnings

# Format Rust + Python
fmt:
    cargo fmt --all
    ruff format python/

# Lint Rust + Python
lint:
    just clippy
    ruff check python/

# Tests
test:
    cargo nextest run --no-fail-fast

# Pre-commit: fast stuff only — fmt check + clippy + python lint
pre-commit:
    just fmt
    cargo fmt --all -- --check
    just clippy
    ruff check python/
    ruff format --check python/

# Pre-push: also run the test suite
pre-push:
    just pre-commit
    just test
