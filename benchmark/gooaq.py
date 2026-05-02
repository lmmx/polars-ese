"""Benchmark polars-ese against polars-luxical and a pure-Rust ESE baseline
on the GooAQ question corpus.

Mirrors the three criterion-bench groups from the Rust bench:
    - bulk throughput (N rows, cold + hot)
    - encode_batch sweep (batch size -> rows/s)
    - encode_by_length sweep (input chars -> ops/s + µs/op)

Usage:
    uv run --group bench python benchmark/gooaq.py
    uv run --group bench python benchmark/gooaq.py --skip-rust
    uv run --group bench python benchmark/gooaq.py --n-bulk 100000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from dataset_gooaq import TEXT_COL, load_gooaq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

EMB_COL = "embedding"
TOK_COL = "token_count"

DEFAULT_N_BULK = 100_000  # matches the Luxical paper's FineWeb throughput sample
BATCH_SIZES = [1, 16, 64, 256, 1000, 10_000]
LENGTH_TARGETS = [10, 50, 100, 250, 500, 1000, 2000]
FAST_THRESHOLD_S = 1.0

SHORT_QUERY = "typed dictionaries"
LONG_QUERY = "typed dictionaries and mappings " * 20  # ~620 chars


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


@dataclass
class BulkResult:
    name: str
    kind: str  # "polars-plugin" or "native-rust"
    dim: int
    cold_s: float
    hot_s: float
    n_docs: int
    n_tokens: int
    n_chars: int
    short_qps: float = 0.0
    long_qps: float = 0.0
    single_run: bool = False

    @property
    def qps(self) -> float:
        return self.n_docs / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.n_tokens / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.n_chars / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def ms_per_kt(self) -> float:
        return (self.hot_s * 1000) / (self.n_tokens / 1000) if self.n_tokens else 0.0

    @property
    def us_per_doc(self) -> float:
        return (self.hot_s * 1e6) / self.n_docs if self.n_docs else 0.0


@dataclass
class BatchSweepResult:
    name: str
    kind: str
    points: dict[int, float] = field(default_factory=dict)  # size -> rows/s


@dataclass
class LengthSweepResult:
    name: str
    kind: str
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    # char_len -> (ops_per_sec, us_per_op)


# -----------------------------------------------------------------------------
# Token counting
# -----------------------------------------------------------------------------


def add_token_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Count BERT wordpiece tokens per doc — ESE's own vocab."""
    if TOK_COL in df.columns:
        return df
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("bert-base-uncased")
    with console.status(
        f"[dim]Tokenising {df.height:,} docs with bert-base-uncased…[/dim]",
        spinner="dots",
    ):
        encodings = tok.encode_batch(df[TEXT_COL].to_list(), add_special_tokens=True)
        counts = [len(e.ids) for e in encodings]
    return df.with_columns(pl.Series(TOK_COL, counts))


# -----------------------------------------------------------------------------
# Backends — same shape as run.py
# -----------------------------------------------------------------------------


def backend_ese():
    from polars_ese import DIMENSIONS, embed_text

    def embed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(embed_text(TEXT_COL).alias(EMB_COL))

    return embed, DIMENSIONS


def backend_luxical(model_id: str = "DatologyAI/luxical-one"):
    from polars_luxical import embed_text, register_model

    register_model(model_id)

    def embed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL))

    probe = pl.DataFrame({TEXT_COL: ["probe"]}).with_columns(
        embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL)
    )
    return embed, probe[EMB_COL].dtype.size


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------


def _time_once(fn: Callable[[], pl.DataFrame]) -> tuple[float, pl.DataFrame]:
    t0 = time.perf_counter()
    result = fn()
    _ = result.height
    return time.perf_counter() - t0, result


def measure_query_qps(
    embed_fn: Callable[[pl.DataFrame], pl.DataFrame],
    warmup: int = 100,
    iters: int = 1_000,
) -> tuple[float, float]:
    short_df = pl.DataFrame({TEXT_COL: [SHORT_QUERY]})
    long_df = pl.DataFrame({TEXT_COL: [LONG_QUERY]})

    for _ in range(warmup):
        embed_fn(short_df)
    t0 = time.perf_counter()
    for _ in range(iters):
        embed_fn(short_df)
    short_qps = iters / (time.perf_counter() - t0)

    for _ in range(warmup):
        embed_fn(long_df)
    t0 = time.perf_counter()
    for _ in range(iters):
        embed_fn(long_df)
    long_qps = iters / (time.perf_counter() - t0)

    return short_qps, long_qps


# -----------------------------------------------------------------------------
# Plugin harness
# -----------------------------------------------------------------------------


def run_plugin_backend(
    name: str,
    loader: Callable,
    df_bulk: pl.DataFrame,
    sentences: list[str],
) -> tuple[BulkResult, BatchSweepResult, LengthSweepResult]:
    n_docs = df_bulk.height
    n_chars = df_bulk.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df_bulk[TOK_COL].sum())

    with console.status(f"[dim]Initialising {name}…[/dim]", spinner="dots"):
        embed_fn, dim = loader()

    # --- Bulk: cold + hot ---
    with console.status(
        f"[dim]Cold embed ({name}, {n_docs:,} rows)…[/dim]", spinner="dots"
    ):
        t_cold, _ = _time_once(lambda: embed_fn(df_bulk))

    is_fast = t_cold < FAST_THRESHOLD_S
    warmup, repeats = (2, 5) if is_fast else (0, 1)

    for _ in range(warmup):
        embed_fn(df_bulk)

    with console.status(
        f"[dim]Steady-state ({repeats} run{'s' if repeats > 1 else ''})…[/dim]",
        spinner="dots",
    ):
        if repeats > 0:
            t_hot = min(
                _time_once(lambda: embed_fn(df_bulk))[0] for _ in range(repeats)
            )
        else:
            t_hot = t_cold

    with console.status(f"[dim]Single-query QPS ({name})…[/dim]", spinner="dots"):
        short_qps, long_qps = measure_query_qps(embed_fn)

    bulk = BulkResult(
        name=name,
        kind="polars-plugin",
        dim=dim,
        cold_s=t_cold,
        hot_s=t_hot,
        n_docs=n_docs,
        n_tokens=n_tokens,
        n_chars=n_chars,
        short_qps=short_qps,
        long_qps=long_qps,
        single_run=(repeats <= 1),
    )

    # --- Batch sweep ---
    batch_res = BatchSweepResult(name=name, kind="polars-plugin")
    with console.status(f"[dim]Batch-size sweep ({name})…[/dim]", spinner="dots"):
        for size in BATCH_SIZES:
            if size > len(sentences):
                continue
            batch = sentences[:size]
            bdf = pl.DataFrame({TEXT_COL: batch})

            # warmup + iter budget: smaller batches need more iters to be stable
            if size <= 16:
                iters = 200
            elif size <= 1000:
                iters = 50
            else:
                iters = 10

            for _ in range(3):
                embed_fn(bdf)

            t0 = time.perf_counter()
            for _ in range(iters):
                embed_fn(bdf)
            dt = time.perf_counter() - t0
            batch_res.points[size] = (iters * size) / dt

    # --- Length sweep ---
    length_res = LengthSweepResult(name=name, kind="polars-plugin")
    with console.status(f"[dim]Length sweep ({name})…[/dim]", spinner="dots"):
        for char_len in LENGTH_TARGETS:
            s = next((x for x in sentences if len(x) >= char_len), None)
            if s is None:
                # synthesise by repeating a short one
                base = sentences[0]
                s = (base * ((char_len // len(base)) + 1))[:char_len]
            trimmed = s[:char_len]
            ldf = pl.DataFrame({TEXT_COL: [trimmed]})

            for _ in range(20):
                embed_fn(ldf)

            iters = 500
            t0 = time.perf_counter()
            for _ in range(iters):
                embed_fn(ldf)
            dt = time.perf_counter() - t0
            ops_s = iters / dt
            us_op = (dt * 1e6) / iters
            length_res.points[char_len] = (ops_s, us_op)

    return bulk, batch_res, length_res


# -----------------------------------------------------------------------------
# Rust baseline
# -----------------------------------------------------------------------------


def run_rust_backend(
    df_bulk: pl.DataFrame,
    sentences: list[str],
) -> tuple[BulkResult, BatchSweepResult, LengthSweepResult] | None:
    n_docs = df_bulk.height
    n_chars = df_bulk.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df_bulk[TOK_COL].sum())

    bench_dir = Path(__file__).parent / "rust"
    txt_path = Path(__file__).parent / "benchmark_data" / "gooaq_texts.tsv"
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_bulk = [
        t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        for t in df_bulk[TEXT_COL].to_list()
    ]
    txt_path.write_text("\n".join(escaped_bulk))

    # Sweep inputs — write as separate files so the Rust binary doesn't
    # reinvent sentence sampling. Sizes are capped at len(sentences).
    sweep_dir = txt_path.parent / "gooaq_sweep"
    sweep_dir.mkdir(exist_ok=True)
    batch_sizes_used: list[int] = []
    for size in BATCH_SIZES:
        if size > len(sentences):
            continue
        batch_sizes_used.append(size)
        escaped = [
            t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            for t in sentences[:size]
        ]
        (sweep_dir / f"batch_{size}.tsv").write_text("\n".join(escaped))

    length_targets_used: list[int] = []
    for char_len in LENGTH_TARGETS:
        s = next((x for x in sentences if len(x) >= char_len), None)
        if s is None:
            base = sentences[0]
            s = (base * ((char_len // len(base)) + 1))[:char_len]
        trimmed = s[:char_len]
        escaped = (
            trimmed.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        )
        (sweep_dir / f"len_{char_len}.tsv").write_text(escaped)
        length_targets_used.append(char_len)

    with console.status(
        "[dim]Building Rust bench (cargo --release)…[/dim]", spinner="dots"
    ):
        build = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--bin",
                "bench_ese_gooaq",
                "--manifest-path",
                str(bench_dir / "Cargo.toml"),
            ],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            console.print(
                "[yellow]Rust gooaq bench build failed — skipping native backend[/yellow]"
            )
            console.print(Text(build.stderr, style="dim"))
            return None

    binary = bench_dir / "target" / "release" / "bench_ese_gooaq"

    with console.status("[dim]Running native Rust gooaq bench…[/dim]", spinner="dots"):
        proc = subprocess.run(
            [
                str(binary),
                str(txt_path),
                str(sweep_dir),
                ",".join(str(s) for s in batch_sizes_used),
                ",".join(str(c) for c in length_targets_used),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            console.print("[yellow]Rust gooaq bench failed — skipping[/yellow]")
            console.print(Text(proc.stderr, style="dim"))
            return None

    # Parse stdout: lines of `key value` with prefixes.
    vals: dict[str, str] = {}
    batch_points: dict[int, float] = {}
    length_points: dict[int, tuple[float, float]] = {}
    for line in proc.stdout.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "batch":
            # batch <size> <rows_per_s>
            batch_points[int(parts[1])] = float(parts[2])
        elif key == "len":
            # len <chars> <ops_per_s> <us_per_op>
            length_points[int(parts[1])] = (float(parts[2]), float(parts[3]))
        else:
            vals[key] = " ".join(parts[1:])

    bulk = BulkResult(
        name="ese (pure Rust)",
        kind="native-rust",
        dim=int(vals["dim"]),
        cold_s=float(vals["cold"]),
        hot_s=float(vals["hot"]),
        n_docs=n_docs,
        n_tokens=n_tokens,
        n_chars=n_chars,
        short_qps=float(vals.get("short_qps", 0.0)),
        long_qps=float(vals.get("long_qps", 0.0)),
    )
    batch_res = BatchSweepResult(
        name="ese (pure Rust)", kind="native-rust", points=batch_points
    )
    length_res = LengthSweepResult(
        name="ese (pure Rust)", kind="native-rust", points=length_points
    )
    return bulk, batch_res, length_res


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def fmt_time(s: float) -> str:
    if s < 0.001:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1000:.1f} ms"
    return f"{s:.2f} s"


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def fmt_si(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:,.1f} {unit}".rstrip()
        n /= 1000
    return f"{n:,.1f} T"


# -----------------------------------------------------------------------------
# Renderers
# -----------------------------------------------------------------------------


def render_header(df_bulk: pl.DataFrame, n_total: int) -> None:
    n_docs = df_bulk.height
    n_chars = df_bulk.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df_bulk[TOK_COL].sum())

    header = Text.assemble(
        ("polars-ese gooaq benchmark\n", "bold cyan"),
        (
            f"{n_docs:,} of {n_total:,} GooAQ questions · "
            f"{n_tokens:,} tokens · {n_chars:,} chars\n",
            "dim",
        ),
        (
            "tokens counted with bert-base-uncased wordpiece (ESE's own vocab)\n",
            "dim italic",
        ),
        (
            "bulk sample sized to match the Luxical paper's 100k-doc "
            "throughput comparison",
            "dim italic",
        ),
    )
    console.print(Panel(header, border_style="cyan", padding=(0, 2)))


def render_bulk(results: list[BulkResult]) -> None:
    table = Table(
        title="Bulk throughput (sampled GooAQ corpus)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Backend", style="bold")
    table.add_column("Kind", style="dim")
    table.add_column("Dim", justify="right")
    table.add_column("Cold", justify="right", style="dim")
    table.add_column("Hot", justify="right")
    table.add_column("ms / kT", justify="right", style="bold green")
    table.add_column("tok/s", justify="right", style="green")
    table.add_column("docs/s", justify="right", style="green")
    table.add_column("chars/s", justify="right", style="dim green")
    table.add_column("µs / doc", justify="right", style="cyan")
    table.add_column("", style="yellow")

    for r in results:
        note = "1 run" if r.single_run else ""
        ms_kt = f"{r.ms_per_kt:.4f}" if r.ms_per_kt < 1 else f"{r.ms_per_kt:.3f}"
        table.add_row(
            r.name,
            "Rust" if r.kind == "native-rust" else "Polars",
            str(r.dim),
            fmt_time(r.cold_s),
            fmt_time(r.hot_s),
            ms_kt,
            fmt_si(r.tokens_per_sec),
            fmt_int(r.qps),
            fmt_si(r.chars_per_sec),
            f"{r.us_per_doc:,.0f}",
            note,
        )
    console.print(table)


def render_query_qps(results: list[BulkResult]) -> None:
    if not any(r.short_qps > 0 for r in results):
        return

    table = Table(
        title="Single-query QPS (search-time latency, reciprocal)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
        caption="Polars plugin path is per-call overhead-bound, not compute-bound.",
        caption_style="dim italic",
    )
    table.add_column("Backend", style="bold")
    table.add_column("short (~18 chars)", justify="right", style="green")
    table.add_column("long (~620 chars)", justify="right", style="green")
    table.add_column("overhead vs Rust", justify="right", style="yellow")

    rust_short = next(
        (r.short_qps for r in results if r.kind == "native-rust" and r.short_qps > 0),
        None,
    )

    for r in results:
        if r.short_qps <= 0:
            continue
        overhead = (
            f"{rust_short / r.short_qps:.0f}×"
            if rust_short and r.kind == "polars-plugin"
            else "—"
        )
        table.add_row(
            r.name,
            fmt_int(r.short_qps),
            fmt_int(r.long_qps),
            overhead,
        )
    console.print(table)


def render_batch_sweep(results: list[BatchSweepResult]) -> None:
    if not results:
        return

    sizes = sorted({s for r in results for s in r.points})
    if not sizes:
        return

    table = Table(
        title="Batch-size sweep (rows / sec)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
        caption="Small batches are overhead-bound; large batches approach compute limit.",
        caption_style="dim italic",
    )
    table.add_column("Backend", style="bold")
    for size in sizes:
        table.add_column(f"N={size:,}", justify="right", style="green")

    for r in results:
        row = [r.name]
        for size in sizes:
            val = r.points.get(size)
            row.append(fmt_si(val) if val else "—")
        table.add_row(*row)
    console.print(table)


def render_length_sweep(results: list[LengthSweepResult]) -> None:
    if not results:
        return

    lens = sorted({c for r in results for c in r.points})
    if not lens:
        return

    ops_table = Table(
        title="Length sweep (single-doc ops / sec)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    ops_table.add_column("Backend", style="bold")
    for c in lens:
        ops_table.add_column(f"{c} ch", justify="right", style="green")

    for r in results:
        row = [r.name]
        for c in lens:
            pt = r.points.get(c)
            row.append(fmt_si(pt[0]) if pt else "—")
        ops_table.add_row(*row)
    console.print(ops_table)

    us_table = Table(
        title="Length sweep (µs / op)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
        caption="Per-call overhead dominates short inputs on Polars plugin paths.",
        caption_style="dim italic",
    )
    us_table.add_column("Backend", style="bold")
    for c in lens:
        us_table.add_column(f"{c} ch", justify="right", style="cyan")

    for r in results:
        row = [r.name]
        for c in lens:
            pt = r.points.get(c)
            row.append(f"{pt[1]:,.1f}" if pt else "—")
        us_table.add_row(*row)
    console.print(us_table)


def render_speedup(results: list[BulkResult]) -> None:
    if len(results) < 2:
        return
    fastest = min(r.hot_s for r in results)
    slowest = max(r.hot_s for r in results)
    if fastest == slowest:
        return

    table = Table(
        title="Relative bulk throughput",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Backend", style="bold")
    table.add_column("vs slowest", justify="right", style="green")
    table.add_column("vs fastest", justify="right", style="dim")
    for r in results:
        table.add_row(
            r.name,
            f"{slowest / r.hot_s:.1f}×",
            f"{r.hot_s / fastest:.2f}×",
        )
    console.print(table)


def render_caveats() -> None:
    text = Text.assemble(
        ("Dimensions differ across backends. ", "yellow"),
        (
            "Throughput is dominated by tokenisation and lookup, not projection size. ",
            "dim",
        ),
        (
            "GooAQ is a question corpus — docs are shorter than PEPs, so per-call "
            "overhead shows up more starkly at the low end of the batch sweep.",
            "dim",
        ),
    )
    console.print(Panel(text, border_style="yellow", padding=(0, 2)))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument(
        "--n-bulk",
        type=int,
        default=DEFAULT_N_BULK,
        help=f"Rows to use for bulk throughput (default: {DEFAULT_N_BULK:,})",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Sampling seed for reproducibility"
    )
    args = ap.parse_args()

    full = load_gooaq()
    n_total = full.height
    n_bulk = min(args.n_bulk, n_total)

    df_bulk = full.sample(n=n_bulk, seed=args.seed)
    df_bulk = add_token_counts(df_bulk)

    # Single list for sweeps — avoid re-reading the full corpus.
    sentences = df_bulk[TEXT_COL].to_list()

    render_header(df_bulk, n_total)

    bulk_results: list[BulkResult] = []
    batch_results: list[BatchSweepResult] = []
    length_results: list[LengthSweepResult] = []

    for name, loader in [
        ("polars-ese", backend_ese),
        ("polars-luxical", backend_luxical),
    ]:
        try:
            b, bs, ls = run_plugin_backend(name, loader, df_bulk, sentences)
            bulk_results.append(b)
            batch_results.append(bs)
            length_results.append(ls)
        except Exception as e:
            console.print(f"[red]!! {name} failed:[/red] {e!r}")

    if not args.skip_rust:
        rust = run_rust_backend(df_bulk, sentences)
        if rust is not None:
            b, bs, ls = rust
            bulk_results.insert(0, b)
            batch_results.insert(0, bs)
            length_results.insert(0, ls)

    if not bulk_results:
        console.print("[red]No backends succeeded.[/red]")
        sys.exit(1)

    console.print()
    render_bulk(bulk_results)
    console.print()
    render_query_qps(bulk_results)
    console.print()
    render_batch_sweep(batch_results)
    console.print()
    render_length_sweep(length_results)
    console.print()
    render_speedup(bulk_results)
    console.print()
    render_caveats()


if __name__ == "__main__":
    main()
