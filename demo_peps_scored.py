"""
Retrieval demo: query the Python PEPs corpus with polars-ese,
now with evaluation scoring vs expected PEP targets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

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


# -----------------------------
# Expected ground truth PEPs
# -----------------------------
EXPECTED = {
    "typed dictionaries and mappings": [484, 526, 544, 585, 589],
    "pattern matching and structural destructuring": [634, 635, 636, 622, 0],
    "asynchronous programming with coroutines": [492, 525, 530, 380, 3156],
    "packaging and dependency specification": [621, 517, 518, 508, 440],
    "removing the global interpreter lock": [703, 684, 554, 0, 0],
    "f-strings and string interpolation syntax": [498, 701, 3101, 292, 215],
}


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


def score_query(query: str, retrieved: List[int], expected: List[int]) -> dict:
    expected_set = {x for x in expected if x != 0}
    retrieved_set = set(retrieved)

    hits = retrieved_set & expected_set

    # Recall@K
    recall = len(hits) / len(expected_set) if expected_set else 0.0

    # Rank-sensitive score (simple inverse distance weighting)
    rank_score = 0.0
    for i, pep in enumerate(retrieved):
        if pep in expected_set:
            true_rank = expected.index(pep)
            rank_score += 1 / (1 + abs(true_rank - i))

    rank_score /= len(expected_set)

    return {
        "recall": recall,
        "rank_score": rank_score,
        "hits": sorted(list(hits)),
    }


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

    all_scores = []

    for query in QUERIES:
        results = embedded.ese.retrieve(query=query, k=TOP_K).select(
            "pep", "title", "similarity"
        )

        retrieved_peps = results["pep"].to_list()
        expected = EXPECTED[query]

        score = score_query(query, retrieved_peps, expected)
        all_scores.append(score)

        print(f'Query: "{query}"')
        print("-" * (WIDTH + 32))

        for row in results.iter_rows(named=True):
            title = row["title"]
            if len(title) > (WIDTH + 10):
                title = title[: (WIDTH + 7)] + "…"

            print(
                f"  PEP {row['pep']:>4}  "
                f"sim={row['similarity']:.3f}  "
                f"{title}"
            )

        print()
        print(f"  Recall@5   : {score['recall']:.2f}")
        print(f"  Rank score : {score['rank_score']:.3f}")
        print(f"  Hits       : {score['hits']}")
        print()

    # -----------------------------
    # Aggregate system score
    # -----------------------------
    avg_recall = sum(s["recall"] for s in all_scores) / len(all_scores)
    avg_rank = sum(s["rank_score"] for s in all_scores) / len(all_scores)

    print("=" * 80)
    print("OVERALL SYSTEM SCORE")
    print(f"Avg Recall@5   : {avg_recall:.3f}")
    print(f"Avg Rank Score : {avg_rank:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()