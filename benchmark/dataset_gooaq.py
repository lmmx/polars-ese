from __future__ import annotations

from pathlib import Path
import polars as pl

DATA_DIR = Path(__file__).parent / "benchmark_data" / "gooaq"

TEXT_COL = "question"
EMB_COL = "embedding"

FILES = [
    DATA_DIR / "gooaq_0.parquet",
]

def load_gooaq() -> pl.DataFrame:
    if not DATA_DIR.exists():
        raise SystemExit(
            f"GooAQ dataset missing at {DATA_DIR}.\n"
            f"Run: just bench-prep-gooaq"
        )

    frames: list[pl.DataFrame] = []

    for f in FILES:
        if not f.exists():
            raise SystemExit(f"Missing file: {f}")

        df = pl.read_parquet(f)

        # normalize schema
        if "question" in df.columns:
            df = df.rename({"question": TEXT_COL})
        elif "sentence1" in df.columns:
            df = df.rename({"sentence1": TEXT_COL})
        else:
            raise ValueError(f"Unknown GooAQ schema: {df.columns}")

        frames.append(df.select(TEXT_COL))

    return pl.concat(frames)