#!/bin/bash
# run_bench.sh — Build and run the p-Spin 2+4 Monte Carlo benchmark
#
# Usage: ./run_bench.sh [--plot] [-warmup w] [-sweeps s] [-dev d]
#
#   --plot   Only build and run the plotter (no GPU benchmark).
#            Use this on your laptop after copying bench_data/*.tsv.
#
#   Without --plot, only the benchmark is built and run (no plots).

set -e

mkdir -p bench_data/plots

if [ "$1" = "--plot" ]; then
    echo "=== Building plotter ==="
    make bench_plot

    echo ""
    echo "=== Generating plots ==="
    bin/plot_benchmark

    echo ""
    echo "=== Done ==="
    echo "Plots: bench_data/plots/*.png"
else
    echo "=== Building benchmark ==="
    make bench

    echo ""
    echo "=== Running benchmark ==="
    bin/benchmark "$@"

    echo ""
    echo "=== Done ==="
    echo "Data: bench_data/*.tsv"
    echo "Run './run_bench.sh --plot' to generate plots."
fi
