from __future__ import annotations

import polars as pl
from polars_ese import DIMENSIONS, embed_text

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

print(f"ESE dim: {DIMENSIONS}")

embedded = df.with_columns(embed_text("text").alias("embedding"))
print(embedded)

results = embedded.ese.retrieve(query="speed of dataframes", k=3)
print(results.select("id", "text", "similarity"))
