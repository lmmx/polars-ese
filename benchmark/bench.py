"""Two-way benchmark: polars-ese vs polars-luxical.

Usage:
    uv run --group bench python benchmark/run.py

Both plugins embed the same 708 PEP corpus. We report:
  - Cold-start time (first embed call — for luxical this includes model
    download/load; for ese this is near-zero because weights are in the binary)
  - Steady-state throughput (best-of-N after warm-up)
  - Top-K retrieval overlap against a fixed query
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import polars as pl
from tabulate import tabulate

from dataset import EMB_COL, LABEL_COL, TEXT_COL, load_peps

QUERY = "Typed dictionaries and mappings"
TOP_K = 10

# Per-backend timing config. If cold embed takes > FAST_THRESHOLD_S, we trust
# a single run (compute-dominated, not noise-dominated) and skip warm-up.
FAST_THRESHOLD_S = 1.0
WARMUP_FAST = 2
WARMUP_SLOW = 0
REPEATS_FAST = 5
REPEATS_SLOW = 1


def say(msg: str = "") -> None:
    """Print with immediate flush so we see progress even if the next step aborts."""
    print(msg, flush=True)


@dataclass
class BenchResult:
    name: str
    dim: int
    cold_start_s: float
    steady_state_s: float
    repeats: int
    top_k_peps: list[int] = field(default_factory=list)
    top_1_pep: int = -1
    top_1_sim: float = float("nan")
    note: str = ""

    @property
    def docs_per_sec(self) -> float:
        return 708 / self.steady_state_s if self.steady_state_s > 0 else 0.0


# -----------------------------------------------------------------------------
# Backend adapters
# -----------------------------------------------------------------------------

def backend_ese():
    from polars_ese import DIMENSIONS, embed_text

    def embed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(embed_text(TEXT_COL).alias(EMB_COL))

    def retrieve(df: pl.DataFrame) -> pl.DataFrame:
        return df.ese.retrieve(query=QUERY, embedding_column=EMB_COL, k=TOP_K)

    return embed, retrieve, DIMENSIONS


def backend_luxical(model_id: str = "DatologyAI/luxical-one"):
    from polars_luxical import embed_text, register_model

    register_model(model_id)

    def embed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL))

    def retrieve(df: pl.DataFrame) -> pl.DataFrame:
        return df.luxical.retrieve(
            query=QUERY, model_name=model_id,
            embedding_column=EMB_COL, k=TOP_K,
        )

    probe = pl.DataFrame({TEXT_COL: ["probe"]})
    probed = probe.with_columns(embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL))
    dim = probed[EMB_COL].dtype.size
    return embed, retrieve, dim


# -----------------------------------------------------------------------------
# Timing harness
# -----------------------------------------------------------------------------

def time_once(fn: Callable[[], pl.DataFrame]) -> tuple[float, pl.DataFrame]:
    t0 = time.perf_counter()
    result = fn()
    _ = result.height  # force materialization
    return time.perf_counter() - t0, result


def fmt_time(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


def run_backend(name: str, loader: Callable) -> BenchResult:
    say(f"\n{'=' * 60}")
    say(f"Running backend: {name}")
    say(f"{'=' * 60}")

    df = load_peps()
    n = df.height

    t_load_start = time.perf_counter()
    embed_fn, retrieve_fn, dim = loader()
    t_load = time.perf_counter() - t_load_start
    say(f"  Backend init: {fmt_time(t_load)} (dim={dim})")

    t_cold, embedded = time_once(lambda: embed_fn(df))
    say(f"  Cold embed ({n} docs): {fmt_time(t_cold)}")

    is_fast = t_cold < FAST_THRESHOLD_S
    if is_fast:
        warmup, repeats = WARMUP_FAST, REPEATS_FAST
    else:
        warmup, repeats = WARMUP_SLOW, REPEATS_SLOW
        say(f"  (slow backend: skipping warm-up, running {repeats} repeat)")

    for _ in range(warmup):
        embed_fn(df)

    if repeats > 0:
        times = [time_once(lambda: embed_fn(df))[0] for _ in range(repeats)]
        t_hot = min(times)
    else:
        t_hot = t_cold

    say(f"  Steady embed (best of {max(repeats, 1)}): {fmt_time(t_hot)} "
        f"({n / t_hot:,.0f} docs/s)")

    top = retrieve_fn(embedded).select(LABEL_COL, "similarity")
    top_peps = top[LABEL_COL].to_list()
    say(f"  Top-{TOP_K} PEPs for query '{QUERY}':")
    say(f"    {top_peps}")

    return BenchResult(
        name=name, dim=dim,
        cold_start_s=t_cold, steady_state_s=t_hot, repeats=max(repeats, 1),
        top_k_peps=top_peps, top_1_pep=top_peps[0],
        top_1_sim=float(top["similarity"][0]),
        note="single run" if repeats <= 1 else "",
    )


def main() -> None:
    backends = [
        ("polars-ese", backend_ese),
        ("polars-luxical", backend_luxical),
    ]

    results: list[BenchResult] = []
    for name, loader in backends:
        try:
            results.append(run_backend(name, loader))
        except Exception as e:
            say(f"\n!! {name} failed: {e!r}")
            traceback.print_exc()

    if not results:
        say("\nNo backends succeeded.")
        sys.exit(1)

    say(f"\n{'=' * 60}")
    say("Summary")
    say(f"{'=' * 60}")
    rows = [
        [r.name, r.dim, fmt_time(r.cold_start_s), fmt_time(r.steady_state_s),
         f"{r.docs_per_sec:,.0f}", r.top_1_pep, f"{r.top_1_sim:.3f}",
         r.note or "-"]
        for r in results
    ]
    say(tabulate(
        rows,
        headers=["Backend", "Dim", "Cold", "Hot", "docs/s",
                 "Top-1 PEP", "Top-1 sim", "Note"],
        tablefmt="github",
    ))

    if len(results) >= 2:
        slowest = max(r.steady_state_s for r in results)
        say(f"\nSpeedup relative to slowest ({fmt_time(slowest)}):")
        for r in results:
            say(f"  {r.name:<25} {slowest / r.steady_state_s:>8.1f}x")

        say("\nTop-K retrieval agreement (Jaccard):")
        sets = {r.name: set(r.top_k_peps) for r in results}
        names = list(sets.keys())
        overlap_rows = []
        for a in names:
            row = [a]
            for b in names:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                row.append(f"{inter / union:.2f}" if union else "-")
            overlap_rows.append(row)
        say(tabulate(overlap_rows, headers=["", *names], tablefmt="github"))


if __name__ == "__main__":
    main()