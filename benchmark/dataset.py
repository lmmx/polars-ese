"""Load the Python PEPs corpus with BERT wordpiece token counts.

Token counts use bert-base-uncased — the same wordpiece vocab ESE is built on,
so the per-kT numbers are methodologically aligned with what ESE is measuring.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

_DATA_DIR = Path(__file__).parent / "benchmark_data"
PEP_DIR = _DATA_DIR / "peps"
PQ_PATH = _DATA_DIR / "peps.parquet"

LABEL_COL = "pep"
TEXT_COL = "text"
EMB_COL = "embedding"
TOK_COL = "token_count"


def _load_rst_files() -> list[dict[str, str]]:
    docs = []
    for path in sorted(PEP_DIR.glob("pep-*.rst")):
        text = path.read_text(errors="replace").strip()
        if text:
            docs.append({LABEL_COL: path.stem.split("-")[1], TEXT_COL: text})
    return docs


def _add_token_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Count BERT wordpiece tokens per doc — the same tokenization ESE uses."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("bert-base-uncased")
    encodings = tok.encode_batch(df[TEXT_COL].to_list(), add_special_tokens=True)
    counts = [len(e.ids) for e in encodings]
    return df.with_columns(pl.Series(TOK_COL, counts))


def load_peps() -> pl.DataFrame:
    if PQ_PATH.exists():
        df = pl.read_parquet(PQ_PATH)
        if TOK_COL in df.columns:
            return df
    else:
        if not PEP_DIR.exists():
            raise SystemExit(
                f"PEP corpus not found at {PEP_DIR}.\n"
                f"Run: bash benchmark/download_peps.sh"
            )
        docs = _load_rst_files()
        df = pl.DataFrame(docs).with_columns(pl.col(LABEL_COL).str.to_integer())

    df = _add_token_counts(df)
    PQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PQ_PATH)
    return df