"""Benchmark polars-ese against polars-luxical and a pure-Rust ESE baseline.

Usage:
    uv run --group bench python benchmark/run.py
    uv run --group bench python benchmark/run.py --skip-rust
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
from dataset import EMB_COL, LABEL_COL, TEXT_COL, TOK_COL, load_peps
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

QUERY = "Typed dictionaries and mappings"
TOP_K = 10
FAST_THRESHOLD_S = 1.0

SHORT_QUERY = "typed dictionaries"
LONG_QUERY = "typed dictionaries and mappings " * 20  # ~620 chars


@dataclass
class BenchResult:
    name: str
    kind: str  # "polars-plugin" or "native-rust"
    dim: int
    cold_s: float
    hot_s: float
    n_docs: int
    n_tokens: int = 0
    n_chars: int = 0
    short_qps: float = 0.0
    long_qps: float = 0.0
    top_k_peps: list[int] = field(default_factory=list)
    top_1_sim: float = float("nan")
    single_run: bool = False

    @property
    def qps(self) -> float:
        return self.n_docs / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.n_chars / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.n_tokens / self.hot_s if self.hot_s > 0 else 0.0

    @property
    def ms_per_kt(self) -> float:
        """Milliseconds per 1k tokens — the canonical embedding-throughput metric."""
        return (self.hot_s * 1000) / (self.n_tokens / 1000) if self.n_tokens else 0.0

    @property
    def us_per_doc(self) -> float:
        return (self.hot_s * 1e6) / self.n_docs if self.n_docs else 0.0


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------


def backend_ese():
    from polars_ese import DIMENSIONS, embed_text

    def embed(df):
        return df.with_columns(embed_text(TEXT_COL).alias(EMB_COL))

    def retrieve(df):
        return df.ese.retrieve(query=QUERY, embedding_column=EMB_COL, k=TOP_K)

    return embed, retrieve, DIMENSIONS


def backend_luxical(model_id: str = "DatologyAI/luxical-one"):
    from polars_luxical import embed_text, register_model

    register_model(model_id)

    def embed(df):
        return df.with_columns(embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL))

    def retrieve(df):
        return df.luxical.retrieve(
            query=QUERY,
            model_name=model_id,
            embedding_column=EMB_COL,
            k=TOP_K,
        )

    probe = pl.DataFrame({TEXT_COL: ["probe"]}).with_columns(
        embed_text(TEXT_COL, model_id=model_id).alias(EMB_COL)
    )
    return embed, retrieve, probe[EMB_COL].dtype.size


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------


def _time_once(fn):
    t0 = time.perf_counter()
    result = fn()
    _ = result.height
    return time.perf_counter() - t0, result


def measure_query_qps(
    embed_fn: Callable[[pl.DataFrame], pl.DataFrame],
    warmup: int = 100,
    iters: int = 1_000,
) -> tuple[float, float]:
    """Single-string embed QPS at short vs long input.

    For Polars plugin backends this is dominated by per-call overhead
    (Python -> expression engine -> Rust -> Arrow), not compute.
    For the Rust binary this matches the flowercomputers blog-post figures.
    """
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
# Polars plugin harness
# -----------------------------------------------------------------------------


def run_plugin_backend(name: str, loader: Callable, df: pl.DataFrame) -> BenchResult:
    n_docs = df.height
    n_chars = df.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df[TOK_COL].sum())

    with console.status(f"[dim]Initialising {name}…[/dim]", spinner="dots"):
        embed_fn, retrieve_fn, dim = loader()

    with console.status(f"[dim]Cold embed ({name})…[/dim]", spinner="dots"):
        t_cold, embedded = _time_once(lambda: embed_fn(df))

    is_fast = t_cold < FAST_THRESHOLD_S
    warmup, repeats = (2, 5) if is_fast else (0, 1)

    for _ in range(warmup):
        embed_fn(df)

    with console.status(
        f"[dim]Steady-state ({repeats} run{'s' if repeats > 1 else ''})…[/dim]",
        spinner="dots",
    ):
        if repeats > 0:
            t_hot = min(_time_once(lambda: embed_fn(df))[0] for _ in range(repeats))
        else:
            t_hot = t_cold

    with console.status(f"[dim]Single-query QPS ({name})…[/dim]", spinner="dots"):
        short_qps, long_qps = measure_query_qps(embed_fn)

    top = retrieve_fn(embedded).select(LABEL_COL, "similarity")
    return BenchResult(
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
        top_k_peps=top[LABEL_COL].to_list(),
        top_1_sim=float(top["similarity"][0]),
        single_run=(repeats <= 1),
    )


# -----------------------------------------------------------------------------
# Native Rust baseline
# -----------------------------------------------------------------------------


def run_rust_backend(df: pl.DataFrame) -> BenchResult | None:
    """Invoke the Rust bench binary via cargo, parse stdout."""
    n_docs = df.height
    n_chars = df.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df[TOK_COL].sum())

    # Write texts as line-delimited with escaped newlines/tabs.
    txt_path = Path(__file__).parent / "benchmark_data" / "peps_texts.tsv"
    escaped = [
        t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        for t in df[TEXT_COL].to_list()
    ]
    txt_path.write_text("\n".join(escaped))

    with console.status(
        "[dim]Building Rust bench (cargo --release)…[/dim]", spinner="dots"
    ):
        build = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--bin",
                "bench_ese",
                "--manifest-path",
                str(Path(__file__).parent / "rust" / "Cargo.toml"),
            ],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            console.print(
                "[yellow]Rust bench build failed — skipping native backend[/yellow]"
            )
            console.print(Text(build.stderr, style="dim"))
            return None

    binary = Path(__file__).parent / "rust" / "target" / "release" / "bench_ese"

    with console.status("[dim]Running native Rust bench…[/dim]", spinner="dots"):
        proc = subprocess.run(
            [str(binary), str(txt_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            console.print("[yellow]Rust bench failed — skipping[/yellow]")
            console.print(Text(proc.stderr, style="dim"))
            return None

    # Parse stdout: key/value lines.
    vals: dict[str, str] = {}
    for line in proc.stdout.strip().splitlines():
        k, v = line.split(maxsplit=1)
        vals[k] = v

    return BenchResult(
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


# -----------------------------------------------------------------------------
# Rendering
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
    """Format large numbers with SI suffix: 225_000_000 -> '225.0 M'."""
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:,.1f} {unit}".rstrip()
        n /= 1000
    return f"{n:,.1f} T"


def render_header(df: pl.DataFrame) -> None:
    n_docs = df.height
    n_chars = df.select(pl.col(TEXT_COL).str.len_chars().sum()).item()
    n_tokens = int(df[TOK_COL].sum())

    header = Text.assemble(
        ("polars-ese benchmark\n", "bold cyan"),
        (f"{n_docs} PEPs · {n_tokens:,} tokens · {n_chars:,} chars\n", "dim"),
        (
            "tokens counted with bert-base-uncased wordpiece (ESE's own vocab)\n",
            "dim italic",
        ),
        (f'query: "{QUERY}"', "dim italic"),
    )
    console.print(Panel(header, border_style="cyan", padding=(0, 2)))


def render_bulk_throughput(results: list[BenchResult]) -> None:
    table = Table(
        title="Bulk throughput (full PEP corpus)",
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


def render_query_qps(results: list[BenchResult]) -> None:
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


def render_speedup(results: list[BenchResult]) -> None:
    fastest = min(r.hot_s for r in results)
    slowest = max(r.hot_s for r in results)
    if fastest == slowest:
        return

    table = Table(
        title="Relative throughput",
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


def render_retrieval(results: list[BenchResult]) -> None:
    plugins = [r for r in results if r.kind == "polars-plugin"]
    if len(plugins) < 2:
        return

    table = Table(
        title="Top retrievals",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Backend", style="bold")
    table.add_column("Top-1 sim", justify="right", style="cyan")
    table.add_column(f"Top-{TOP_K} PEPs", style="dim")
    for r in plugins:
        table.add_row(
            r.name,
            f"{r.top_1_sim:.3f}",
            " ".join(str(p) for p in r.top_k_peps),
        )
    console.print(table)

    overlap = Table(
        title="Top-K agreement (Jaccard)",
        title_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    overlap.add_column("", style="bold")
    for p in plugins:
        overlap.add_column(p.name, justify="right")
    for a in plugins:
        row = [a.name]
        for b in plugins:
            inter = len(set(a.top_k_peps) & set(b.top_k_peps))
            union = len(set(a.top_k_peps) | set(b.top_k_peps))
            val = inter / union if union else 0.0
            style = (
                "cyan"
                if a.name == b.name
                else ("green" if val > 0.5 else "yellow" if val > 0.25 else "red")
            )
            row.append(f"[{style}]{val:.2f}[/{style}]")
        overlap.add_row(*row)
    console.print(overlap)


def render_caveats() -> None:
    text = Text.assemble(
        ("Dimensions differ across backends. ", "yellow"),
        (
            "Throughput is dominated by tokenisation and lookup, not projection size. ",
            "dim",
        ),
        ("Retrieval quality differences reflect training objective: ", "dim"),
        ("ESE ", "bold"),
        ("is asymmetric query-doc retrieval; ", "dim"),
        ("Luxical ", "bold"),
        ("is symmetric doc-doc similarity.", "dim"),
    )
    console.print(Panel(text, border_style="yellow", padding=(0, 2)))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-rust", action="store_true", help="Skip the native-Rust ESE baseline"
    )
    args = ap.parse_args()

    df = load_peps()
    render_header(df)

    results: list[BenchResult] = []

    for name, loader in [
        ("polars-ese", backend_ese),
        ("polars-luxical", backend_luxical),
    ]:
        try:
            results.append(run_plugin_backend(name, loader, df))
        except Exception as e:
            console.print(f"[red]!! {name} failed:[/red] {e!r}")

    if not args.skip_rust:
        rust = run_rust_backend(df)
        if rust is not None:
            results.insert(0, rust)

    if not results:
        console.print("[red]No backends succeeded.[/red]")
        sys.exit(1)

    console.print()
    render_bulk_throughput(results)
    console.print()
    render_query_qps(results)
    console.print()
    render_speedup(results)
    console.print()
    render_retrieval(results)
    console.print()
    render_caveats()


if __name__ == "__main__":
    main()
