from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import polars_distance as pld
from polars.api import register_dataframe_namespace
from polars.plugins import register_plugin_function

from polars_ese._polars_ese import DIMENSIONS

from .utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location
    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = ["embed_text", "DIMENSIONS"]


def _plug(expr: IntoExpr, **kwargs) -> pl.Expr:
    func_name = inspect.stack()[1].function
    return register_plugin_function(
        plugin_path=lib,
        function_name=func_name,
        args=parse_into_expr(expr),
        is_elementwise=True,
        kwargs=kwargs,
    )


def embed_text(expr: IntoExpr) -> pl.Expr:
    """Embed text using ESE into a fixed-size Array[f32, DIMENSIONS].

    ESE ships with a single baked-in model (static-retrieval-mrl-en-v1,
    truncated/quantized per build features). No model loading or registry
    required — the weights are in the binary.
    """
    return _plug(expr, _placeholder=None)


@register_dataframe_namespace("ese")
class ESENamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def embed(
        self,
        columns: str | list[str],
        output_column: str = "embedding",
        join_columns: bool = True,
    ) -> pl.DataFrame:
        if isinstance(columns, str):
            columns = [columns]

        if join_columns and len(columns) > 1:
            df = self._df.with_columns(
                pl.concat_str(columns, separator=" ").alias("_text_to_embed"),
            )
            text_col = "_text_to_embed"
        else:
            df = self._df
            text_col = columns[0]

        out = df.with_columns(embed_text(text_col).alias(output_column))
        if join_columns and len(columns) > 1:
            out = out.drop("_text_to_embed")
        return out

    def retrieve(
        self,
        query: str,
        embedding_column: str = "embedding",
        k: int | None = None,
        threshold: float | None = None,
        similarity_metric: str = "cosine",
        add_similarity_column: bool = True,
    ) -> pl.DataFrame:
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found.")

        q = pl.DataFrame({"_q": [query]}).with_columns(
            embed_text("_q").alias("_q_emb"),
        )
        result = self._df.join(q.select("_q_emb"), how="cross")

        if similarity_metric == "cosine":
            sim = 1 - pld.col(embedding_column).dist_arr.cosine("_q_emb")
        elif similarity_metric == "dot":
            sim = pl.col(embedding_column).dot(pl.col("_q_emb"))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        if add_similarity_column:
            result = result.with_columns(sim.alias("similarity"))
        result = result.drop("_q_emb")
        if threshold is not None:
            result = result.filter(pl.col("similarity") >= threshold)
        result = result.sort("similarity", descending=True)
        if k is not None:
            result = result.head(k)
        return result