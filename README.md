# polars-ese

A Polars plugin for [ESE](https://github.com/flowercomputers/ese) (Extremely Static Embeddings) — the fastest static text embedding model, baked into the binary at compile time.

## Why

ESE is a purpose-built runtime for the `static-retrieval-mrl-en-v1` model, with the tokenizer and weights materialised as static globals via a `build.rs` perfect-hash-function. No model loading, no ONNX, no dylib juggling, no HF Hub cache. Just a function call that produces an embedding.

This plugin wires that into Polars as an expression returning `Array[f32, DIMENSIONS]`.

## Install

```bash
pip install polars-ese
```

Or, from source:

```bash
maturin develop --release
```

For maximum throughput on your machine, build natively:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

## Usage

```python
import polars as pl
from polars_ese import embed_text, DIMENSIONS

df = pl.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Polars is a fast DataFrame library",
        "Static embeddings skip transformer inference entirely",
        "The Eiffel Tower is in Paris",
    ],
})

embedded = df.with_columns(embed_text("text").alias("embedding"))
print(embedded)

# Namespace API
embedded = df.ese.embed(columns="text")
results = embedded.ese.retrieve(query="speed of dataframes", k=2)
print(results.select("id", "text", "similarity"))
```

## Configuration

ESE's embedding dimension and quantization are chosen at **build time** via crate features. The defaults shipped here are `dim-512` + `quant-8`. To change them, edit `Cargo.toml`:

```toml
ese = { path = "../ese", features = ["rayon", "dim-256", "quant-8"] }
```

and rebuild. Available options: `dim-{32,64,128,256,512,768,1024}`, `quant-{8,16}` (or neither for f32). The chosen value is exposed at runtime as `polars_ese.DIMENSIONS`.

## Not for search queries

Like most static embedding models distilled from symmetric training, ESE is best suited to document-document similarity (clustering, dedup, classification). It is NOT trained for query-document asymmetric retrieval. Short queries against long documents will produce weaker rankings than a transformer-based retriever. For that, use [polars-fastembed](https://github.com/lmmx/polars-fastembed).

## License

Apache-2.0