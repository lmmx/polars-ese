from __future__ import annotations

import time

import polars as pl
import polars_ese
from dataset_gooaq import TEXT_COL, load_gooaq

# -----------------------------------------------------------------------------
# Dataset (cached parquet via Polars)
# -----------------------------------------------------------------------------


def get_sentences() -> list[str]:
    df = load_gooaq()
    return df[TEXT_COL].to_list()


# -----------------------------------------------------------------------------
# Bench: encode_single equivalent
# -----------------------------------------------------------------------------


def bench_encode_single(sentences: list[str]) -> None:
    df = pl.DataFrame({TEXT_COL: sentences})

    # warmup
    _ = df.head(1).ese.embed(columns=TEXT_COL)

    s = sentences[0]

    iters = 5000

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pl.DataFrame({TEXT_COL: [s]}).ese.embed(columns=TEXT_COL)
    dt = time.perf_counter() - t0

    print("\n[encode_single]")
    print(f"short_qps: {iters/dt:.2f}")


# -----------------------------------------------------------------------------
# Bench: encode_batch equivalent
# -----------------------------------------------------------------------------


def bench_encode_batch(sentences: list[str]) -> None:
    sizes = [1, 16, 64, 256, 1000]

    print("\n[encode_batch]")

    for size in sizes:
        if size > len(sentences):
            continue

        batch = sentences[:size]
        df = pl.DataFrame({TEXT_COL: batch})

        # warmup
        _ = df.ese.embed(columns=TEXT_COL)

        iters = 200

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = df.ese.embed(columns=TEXT_COL)
        dt = time.perf_counter() - t0

        qps = (iters * size) / dt
        print(f"size={size:>6}  throughput={qps:,.0f} rows/s")


# -----------------------------------------------------------------------------
# Bench: encode_by_length equivalent
# -----------------------------------------------------------------------------


def bench_encode_by_length(sentences: list[str]) -> None:
    targets = [10, 50, 100, 250, 500, 1000, 2000]

    print("\n[encode_by_length]")

    for char_len in targets:
        s = next((x for x in sentences if len(x) >= char_len), None)
        if not s:
            continue

        trimmed = s[:char_len]
        df = pl.DataFrame({TEXT_COL: [trimmed]})

        # warmup
        _ = df.ese.embed(columns=TEXT_COL)

        iters = 1000

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = pl.DataFrame({TEXT_COL: [trimmed]}).ese.embed(columns=TEXT_COL)
        dt = time.perf_counter() - t0

        print(f"chars={char_len:<5}  ops/s={iters/dt:,.0f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run():
    sentences = get_sentences()
    assert sentences, "dataset empty"

    print(f"Loaded {len(sentences)} GooAQ questions")

    bench_encode_single(sentences)
    bench_encode_batch(sentences)
    bench_encode_by_length(sentences)


if __name__ == "__main__":
    run()
