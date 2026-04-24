# Fetch PEP corpus (one-time)
bench-prep:
    bash benchmark/download_peps.sh

# Run three-way embedding benchmark
bench:
    python benchmark/run.py
