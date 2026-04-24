#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/benchmark_data/gooaq"
mkdir -p "$TARGET_DIR"

URLS=(
  "https://huggingface.co/datasets/sentence-transformers/gooaq/resolve/main/train-00000-of-00001.parquet"
)

for i in "${!URLS[@]}"; do
  URL="${URLS[$i]}"
  OUT="$TARGET_DIR/gooaq_$i.parquet"

  if [ ! -f "$OUT" ]; then
    echo "Downloading $URL ..."
    curl -L "$URL" -o "$OUT"
  else
    echo "Cached: $OUT"
  fi
done

echo "GooAQ cached in: $TARGET_DIR"