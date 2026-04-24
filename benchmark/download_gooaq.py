"""
GooAQ dataset downloader (cache loader)

- uses static HF parquet shard URLs
- downloads only if missing
- stores in benchmark_data/gooaq/
- no datasets library, no requests dependency
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# -----------------------------------------------------------------------------
# STATIC PARQUET SHARDS (same style as Rust benchmark)
# -----------------------------------------------------------------------------

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/0.parquet",
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/1.parquet",
]

CACHE_DIR = Path("benchmark_data/gooaq")


# -----------------------------------------------------------------------------
# LOW-LEVEL DOWNLOAD (no dependencies)
# -----------------------------------------------------------------------------


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(PARQUET_URLS):
        path = CACHE_DIR / f"gooaq_{i}.parquet"

        if not path.exists():
            download(url, path)
        else:
            print(f"cached: {path.name}")

    print(f"\nGooAQ cached in: {CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()
