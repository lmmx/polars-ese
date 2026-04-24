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

## Benchmark

### Benchmark 1: PEPs

<img width="2137" height="1912" alt="Screenshot from 2026-04-24 13-35-22" src="https://github.com/user-attachments/assets/e68d4ad8-a5c8-4ffd-a6cc-eed684e32c40" />

- `ese` crate built at 512D in Q8, measuring 0.0165 ms/kT
- `polars-luxical` had [previously](https://cog.spin.systems/2025-in-review-inverse-problems)
  measured at 0.5ms/kT but here was measured at 1.5ms/kT

Against previous best, `ese` is 30x faster than Luxical One,
and against current best in the same benchmark `ese` is 90x faster.

Run `just bench` (see `.just/bench.just`) to execute the benchmark during development.

### Benchmark 2: GooAQ

This is the one used by ese, and from the results it appears that at a larger scale the Polars formb
becomes +50% slower, pure Rust at 0.027 ms/kT vs. polars-ese at 0.038 ms/kT (but that is still exceptionally fast).

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

## Demos

### Demo 1: simple example use

Run `demo.py` produces the following example:

```python
ESE embedding dimension: 512 (Array[f32, 512])

Step 1 — Embed 4 short texts
------------------------------------------------------------
shape: (4, 3)
┌─────┬─────────────────────────────────┬─────────────────────────────────┐
│ id  ┆ text                            ┆ embedding                       │
│ --- ┆ ---                             ┆ ---                             │
│ i64 ┆ str                             ┆ array[f32, 512]                 │
╞═════╪═════════════════════════════════╪═════════════════════════════════╡
│ 1   ┆ Polars is a blazingly fast Dat… ┆ [-1.933818, 3.065067, … -0.934… │
│ 2   ┆ Static embedding models are mu… ┆ [-9.807067, 6.439317, … -4.558… │
│ 3   ┆ The Eiffel Tower is in Paris    ┆ [-1.259356, -6.883103, … -2.04… │
│ 4   ┆ Rust makes systems programming… ┆ [-6.17557, 5.905078, … 4.65535… │
└─────┴─────────────────────────────────┴─────────────────────────────────┘
Each row's text becomes a 512-dim float vector. Cosine similarity between two such vectors tells us how related the texts are.

Step 2 — Run queries against the embedded corpus
------------------------------------------------------------

Query: "speed of dataframes"
shape: (3, 3)
┌─────┬─────────────────────────────────┬────────────┐
│ id  ┆ text                            ┆ similarity │
│ --- ┆ ---                             ┆ ---        │
│ i64 ┆ str                             ┆ f32        │
╞═════╪═════════════════════════════════╪════════════╡
│ 1   ┆ Polars is a blazingly fast Dat… ┆ 0.55803    │
│ 2   ┆ Static embedding models are mu… ┆ 0.223786   │
│ 4   ┆ Rust makes systems programming… ┆ 0.07403    │
└─────┴─────────────────────────────────┴────────────┘
  → Top hit: id=1 (similarity 0.558)
    "Polars is a blazingly fast DataFrame library"

Query: "famous landmarks in France"
shape: (3, 3)
┌─────┬─────────────────────────────────┬────────────┐
│ id  ┆ text                            ┆ similarity │
│ --- ┆ ---                             ┆ ---        │
│ i64 ┆ str                             ┆ f32        │
╞═════╪═════════════════════════════════╪════════════╡
│ 3   ┆ The Eiffel Tower is in Paris    ┆ 0.321781   │
│ 1   ┆ Polars is a blazingly fast Dat… ┆ 0.056369   │
│ 2   ┆ Static embedding models are mu… ┆ -0.029398  │
└─────┴─────────────────────────────────┴────────────┘
  → Top hit: id=3 (similarity 0.322)
    "The Eiffel Tower is in Paris"

Query: "low-level programming languages"
shape: (3, 3)
┌─────┬─────────────────────────────────┬────────────┐
│ id  ┆ text                            ┆ similarity │
│ --- ┆ ---                             ┆ ---        │
│ i64 ┆ str                             ┆ f32        │
╞═════╪═════════════════════════════════╪════════════╡
│ 4   ┆ Rust makes systems programming… ┆ 0.234005   │
│ 2   ┆ Static embedding models are mu… ┆ 0.067669   │
│ 3   ┆ The Eiffel Tower is in Paris    ┆ 0.023093   │
└─────┴─────────────────────────────────┴────────────┘
  → Top hit: id=4 (similarity 0.234)
    "Rust makes systems programming less painful"
```

### Demo 2: searching Python PEPs

Running `demo_peps.py` produces this example:

```python
polars-ese PEP retrieval demo (dim=512)

Loaded 726 PEPs. Embedding…
Embedded 726 docs · 13,932,707 chars total

Query: "typed dictionaries and mappings"
-------------------------------------------------------------------------------------------------
  PEP  764  sim=0.472  Inline typed dictionaries
  PEP  412  sim=0.468  Key-Sharing Dictionary
  PEP  589  sim=0.406  TypedDict: Type Hints for Dictionaries with a Fixed Set of Keys
  PEP  804  sim=0.353  An external dependency registry and name mapping mechanism
  PEP  372  sim=0.344  Adding an ordered dictionary to collections

Query: "pattern matching and structural destructuring"
-------------------------------------------------------------------------------------------------
  PEP  634  sim=0.475  Structural Pattern Matching: Specification
  PEP  642  sim=0.405  Explicit Pattern Syntax for Structural Pattern Matching
  PEP  622  sim=0.401  Structural Pattern Matching
  PEP  635  sim=0.382  Structural Pattern Matching: Motivation and Rationale
  PEP  653  sim=0.378  Precise Semantics for Pattern Matching

Query: "asynchronous programming with coroutines"
-------------------------------------------------------------------------------------------------
  PEP  525  sim=0.456  Asynchronous Generators
  PEP  492  sim=0.446  Coroutines with async and await syntax
  PEP  220  sim=0.445  Coroutines, Generators, Continuations
  PEP  828  sim=0.387  Supporting 'yield from' in asynchronous generators
  PEP  530  sim=0.368  Asynchronous Comprehensions

Query: "packaging and dependency specification"
-------------------------------------------------------------------------------------------------
  PEP  735  sim=0.559  Dependency Groups in pyproject.toml
  PEP  631  sim=0.515  Dependency specification in pyproject.toml based on PEP 508
  PEP  722  sim=0.458  Dependency specification for single-file scripts
  PEP  725  sim=0.419  Specifying external dependencies in pyproject.toml
  PEP  633  sim=0.418  Dependency specification in pyproject.toml using an exploded TOML table

Query: "removing the global interpreter lock"
-------------------------------------------------------------------------------------------------
  PEP  684  sim=0.432  A Per-Interpreter GIL
  PEP  734  sim=0.409  Multiple Interpreters in the Stdlib
  PEP  554  sim=0.401  Multiple Interpreters in the Stdlib
  PEP  797  sim=0.381  Shared Object Proxies
  PEP  371  sim=0.243  Addition of the multiprocessing package to the standard library

Query: "f-strings and string interpolation syntax"
-------------------------------------------------------------------------------------------------
  PEP  701  sim=0.660  Syntactic formalization of f-strings
  PEP  498  sim=0.652  Literal String Interpolation
  PEP  536  sim=0.651  Final Grammar for Literal String Interpolation
  PEP  502  sim=0.597  String Interpolation - Extended Discussion
  PEP  750  sim=0.580  Template Strings
```

There is also an attempt to score how well they did at retrieval vs some reasonable expected results:

```bash
python demo_peps_scored.py  | rg 'Query|Recall|Rank score'
```

```python
Query: "typed dictionaries and mappings"
  Recall@5   : 0.20
  Rank score : 0.067
Query: "pattern matching and structural destructuring"
  Recall@5   : 0.75
  Rank score : 0.458
Query: "asynchronous programming with coroutines"
  Recall@5   : 0.60
  Rank score : 0.267
Query: "packaging and dependency specification"
  Recall@5   : 0.00
  Rank score : 0.000
Query: "removing the global interpreter lock"
  Recall@5   : 0.67
  Rank score : 0.500
Query: "f-strings and string interpolation syntax"
  Recall@5   : 0.40
  Rank score : 0.200
Avg Recall@5   : 0.436
```

## Useful embeddings for retrieval

Retrieval seems to work well. ESE's underlying model (`static-retrieval-mrl-en-v1`)
was trained by Tom Aarsen (Sentence Transformers) with contrastive loss on query-document pairs,
specifically for asymmetric retrieval. Despite the zero-parameter runtime,
it performs strongly on short-query-to-long-document search — see
[`demo_peps.py`](demo_peps.py) for retrieval against the full Python PEP
corpus, where top-5 results are consistently on-topic (e.g. all five
Structural Pattern Matching PEPs surface for "pattern matching and structural
destructuring").

**What ESE is good for:**

- Semantic search over document collections (RAG retrieval, internal knowledge bases)
- Clustering, deduplication, near-duplicate detection
- Classification by nearest-centroid or k-NN
- Any pipeline where you need embeddings and can't afford transformer inference latency

**Where you may want a transformer instead:**

- Reranking the top candidates from a first-pass retrieval
- Tasks requiring long-context nuance (dense logical inference, negation handling)
- Domains far outside the `static-retrieval-mrl-en-v1` training distribution

For those cases see [polars-fastembed](https://github.com/lmmx/polars-fastembed),
which wraps transformer-based embedders via ONNX.
