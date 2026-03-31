#!/bin/bash
# run_bench.sh — Build and run the p-Spin 2+4 Monte Carlo benchmark
#
# Usage: ./run_bench.sh [--plot] [--dev=D] [--visnu] [-warmup w] [-sweeps s]
#
#   --plot   Only build and run the plotter (no GPU benchmark).
#            Use this on your laptop after copying bench_data/*.tsv.
#   --dev=D  CUDA device index (default: 0).
#   --visnu  Compile for GTX 680 (sm_30).
#
#   Without --plot, only the benchmark is built and run (no plots).

set -e

mkdir -p bench_data/plots

DEV=0
PLOT=0
VISNU=0
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --plot)    PLOT=1 ;;
        --dev=*)   DEV="${arg#*=}" ;;
        --visnu)   VISNU=1 ;;
        *)         EXTRA_ARGS+=("$arg") ;;
    esac
done

MAKE_FLAGS=""
if [ "$VISNU" = "1" ]; then
    MAKE_FLAGS="visnu=1"
fi

if [ "$PLOT" = "1" ]; then
    echo "=== Building plotter ==="
    make bench_plot $MAKE_FLAGS

    # Determine plot directory from GPU name
    PLOTDIR="bench_data/plots"
    if [ -f bench_data/gpu_name.txt ]; then
        GPU_NAME=$(head -1 bench_data/gpu_name.txt | tr ' ' '_')
        if [ -n "$GPU_NAME" ]; then
            PLOTDIR="bench_data/plots_${GPU_NAME}"
        fi
    fi
    mkdir -p "$PLOTDIR"

    echo ""
    echo "=== Generating plots ==="
    bin/plot_benchmark "$PLOTDIR"

    echo ""
    echo "=== Done ==="
    echo "Plots: $PLOTDIR/*.png"
else
    echo "=== Building benchmark ==="
    make bench $MAKE_FLAGS

    echo ""
    echo "=== Running benchmark (dev=$DEV) ==="
    bin/benchmark -dev "$DEV" "${EXTRA_ARGS[@]}"

    echo ""
    echo "=== Done ==="
    echo "Data: bench_data/*.tsv"
    echo "Run './run_bench.sh --plot' to generate plots."
fi
