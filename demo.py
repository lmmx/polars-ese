"""A tiny end-to-end demo of polars-ese.

Embeds four short texts, runs three queries against them, and prints the ranked
results with a short explanation of what you're looking at.
"""

from __future__ import annotations

import polars as pl
from polars_ese import DIMENSIONS, embed_text

print(f"ESE embedding dimension: {DIMENSIONS} (Array[f32, {DIMENSIONS}])")
print()

# ----------------------------------------------------------------------------
# 1. Embed a small corpus
# ----------------------------------------------------------------------------
df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "text": [
            "Polars is a blazingly fast DataFrame library",
            "Static embedding models are much faster than transformers",
            "The Eiffel Tower is in Paris",
            "Rust makes systems programming less painful",
        ],
    }
)

print("Step 1 — Embed 4 short texts")
print("-" * 60)
embedded = df.with_columns(embed_text("text").alias("embedding"))
print(embedded)
print(
    "Each row's text becomes a 512-dim float vector. "
    "Cosine similarity between two such vectors tells us how related the texts are."
)
print()

# ----------------------------------------------------------------------------
# 2. Retrieve against a few queries
# ----------------------------------------------------------------------------
queries = [
    "speed of dataframes",
    "famous landmarks in France",
    "low-level programming languages",
]

print("Step 2 — Run queries against the embedded corpus")
print("-" * 60)
for q in queries:
    results = embedded.ese.retrieve(query=q, k=3).select("id", "text", "similarity")
    print(f'\nQuery: "{q}"')
    print(results)
    top = results.row(0, named=True)
    print(f"  → Top hit: id={top['id']} (similarity {top['similarity']:.3f})")
    print(f'    "{top["text"]}"')
