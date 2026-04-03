#!/bin/bash

# Smoothed Cube visualization test
# Generates plots in smoothed_cube/ showing:
#   - Random samples filling the squircle (Re(a1) vs Re(a2))
#   - Moduli on the superellipse (|a1| vs |a2|)
#   - MC trajectory (2D scatter + 3D splot)
#   - Constraint check: S4 = sum|a_k|^4 = N

N=2
ITER=10
SEED=42
NSAMPLES=10000

for arg in "$@"; do
    case "$arg" in
        --N=*)        N="${arg#*=}" ;;
        --iter=*)     ITER="${arg#*=}" ;;
        --seed=*)     SEED="${arg#*=}" ;;
        --nsamples=*) NSAMPLES="${arg#*=}" ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --N=N           Number of spins (default: $N)"
            echo "  --iter=K        MC sweeps = 2^K (default: $ITER)"
            echo "  --seed=S        RNG seed (default: $SEED)"
            echo "  --nsamples=M    Independent samples for scatter (default: $NSAMPLES)"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

./bin/smoothed_cube -N $N -iter $ITER -seed $SEED -nsamples $NSAMPLES
