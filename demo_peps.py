"""Retrieval demo: query the Python PEPs corpus with polars-ese.

Run `bash benchmark/download_peps.sh` first to fetch the PEPs.

The README cautions that ESE was trained for document-document similarity,
not asymmetric query-document retrieval — but the underlying model
(static-retrieval-mrl-en-v1) *was* contrastively trained on query-doc pairs,
so in practice retrieval works surprisingly well. This script tries a handful
of queries against all 700+ PEPs so you can judge the quality yourself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from polars_ese import DIMENSIONS, embed_text

PEP_DIR = Path(__file__).parent / "benchmark" / "benchmark_data" / "peps"

QUERIES = [
    "typed dictionaries and mappings",
    "pattern matching and structural destructuring",
    "asynchronous programming with coroutines",
    "packaging and dependency specification",
    "removing the global interpreter lock",
    "f-strings and string interpolation syntax",
]

TOP_K = 5


def load_peps() -> pl.DataFrame:
    if not PEP_DIR.exists():
        sys.exit(
            f"PEP corpus not found at {PEP_DIR}.\n"
            f"Run: bash benchmark/download_peps.sh"
        )
    rows = []
    for path in sorted(PEP_DIR.glob("pep-*.rst")):
        text = path.read_text(errors="replace").strip()
        if not text:
            continue
        # First non-empty line of each PEP is typically "PEP: N", second is "Title: ..."
        title = ""
        for line in text.splitlines()[:10]:
            if line.startswith("Title:"):
                title = line.removeprefix("Title:").strip()
                break
        rows.append(
            {
                "pep": int(path.stem.split("-")[1]),
                "title": title or "(no title)",
                "text": text,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    WIDTH = 65
    print(f"polars-ese PEP retrieval demo (dim={DIMENSIONS})")
    print()

    df = load_peps()
    print(f"Loaded {df.height} PEPs. Embedding…")
    embedded = df.with_columns(embed_text("text").alias("embedding"))
    total_chars = df.select(pl.col("text").str.len_chars().sum()).item()
    print(f"Embedded {df.height} docs · {total_chars:,} chars total")
    print()

    for query in QUERIES:
        results = embedded.ese.retrieve(query=query, k=TOP_K).select(
            "pep", "title", "similarity"
        )
        print(f'Query: "{query}"')
        print("-" * (WIDTH + 32))
        for row in results.iter_rows(named=True):
            title = row["title"]
            if len(title) > (WIDTH + 10):
                title = title[: (WIDTH + 7)] + "…"
            print(
                f"  PEP {row['pep']:>4}  " f"sim={row['similarity']:.3f}  " f"{title}"
            )
        print()


if __name__ == "__main__":
    main()
