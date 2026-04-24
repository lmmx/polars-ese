# Fetch PEP corpus (one-time)
bench-prep:
    bash benchmark/download_peps.sh
    just bench-prep-gooaq

[working-directory: 'benchmark']
bench-prep-gooaq:
    bash download_gooaq.sh

[working-directory: 'benchmark']
bench-gooaq:
    python gooaq.py

# Run three-way embedding benchmark
bench-peps:
    python benchmark/run.py

bench:
    just bench-peps
    just bench-gooaq
