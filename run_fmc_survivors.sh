#!/usr/bin/env bash
# ============================================================================
# run_fmc_survivors.sh
#
# Scansione: quante coppie e quadruplette sopravvivono all'FMC in media,
# in funzione di N (numero di spin), con errori jackknife.
#
# Gamma is computed as 1/(2*(N-1)) for each N.
#
# Usage:
#   ./run_fmc_survivors.sh [--nmin=18] [--nmax=120] [--nstep=1] \
#                          [--samples=100] [--seed=42]
# ============================================================================
set -euo pipefail

# ── Default parameters ─────────────────────────────────────────────────────
NMIN=18
NMAX=120
NSTEP=1
SAMPLES=100
SEED=$RANDOM

# ── Parse command-line arguments ───────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --nmin=*)    NMIN="${arg#*=}" ;;
        --nmax=*)    NMAX="${arg#*=}" ;;
        --nstep=*)   NSTEP="${arg#*=}" ;;
        --samples=*) SAMPLES="${arg#*=}" ;;
        --seed=*)    SEED="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --nmin=N       N minimo  (default: 18)"
            echo "  --nmax=N       N massimo (default: 120)"
            echo "  --nstep=S      passo     (default: 1)"
            echo "  --samples=S    # disorder realizations per N (default: 100)"
            echo "  --seed=S       master seed (default: 42)"
            echo "  gamma = 1/(2*(N-1)) is computed automatically for each N"
            exit 0 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Setup ──────────────────────────────────────────────────────────────────
OUTDIR="analysis/fmc_survivors"
mkdir -p "${OUTDIR}/raw"

make test_fmc_survivors

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           FMC survivors scan                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  N range : [%-4d .. %-4d]  step %-3d                    ║\n" "$NMIN" "$NMAX" "$NSTEP"
printf "║  Samples : %-6d                                      ║\n" "$SAMPLES"
echo   "║  Gamma   : 1/(2*(N-1))  (computed per N)              ║"
printf "║  Seed    : %-8s                                    ║\n" "$SEED"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PAIRS_FILE="${OUTDIR}/pairs_vs_N.dat"
QUARTS_FILE="${OUTDIR}/quartets_vs_N.dat"

echo "# N  gamma  mean_pairs_active  jackknife_err  total_pairs" > "$PAIRS_FILE"
echo "# N  gamma  mean_quartets_active  jackknife_err  total_quartets" > "$QUARTS_FILE"

# ── Main loop over N ───────────────────────────────────────────────────────
for N in $(seq "$NMIN" "$NSTEP" "$NMAX"); do
    # gamma = 1 / (2*(N-1))
    GAMMA=$(awk "BEGIN{printf \"%.10f\", 1.0/(2.0*($N - 1.0))}")
    printf "  N=%-4d  gamma=%-12s  " "$N" "$GAMMA"

    RAW="${OUTDIR}/raw/N${N}.dat"
    bin/test_fmc_survivors --N="$N" --gamma="$GAMMA" --samples="$SAMPLES" --seed="$SEED" > "$RAW"

    # Jackknife mean + error
    awk -v N="$N" -v G="$GAMMA" -v pf="$PAIRS_FILE" -v qf="$QUARTS_FILE" '
    !/^#/ {
        n++; p[n] = $2; pt = $3; q[n] = $4; qt = $5
    }
    END {
        sp = 0; sq = 0
        for (i = 1; i <= n; i++) { sp += p[i]; sq += q[i] }
        mp = sp / n;  mq = sq / n

        vp = 0; vq = 0
        for (j = 1; j <= n; j++) {
            jp = (sp - p[j]) / (n - 1)
            jq = (sq - q[j]) / (n - 1)
            vp += (jp - mp)^2
            vq += (jq - mq)^2
        }
        ep = sqrt((n - 1.0) / n * vp)
        eq = sqrt((n - 1.0) / n * vq)

        printf "%d  %.10f  %.6f  %.6f  %d\n", N, G, mp, ep, pt >> pf
        printf "%d  %.10f  %.6f  %.6f  %d\n", N, G, mq, eq, qt >> qf
    }' "$RAW"

    SEED=$RANDOM # Randomize seed for next N to avoid correlations between runs
    # Show progress
    LAST_P=$(tail -1 "$PAIRS_FILE" | awk '{printf "%.1f ± %.1f", $3, $4}')
    LAST_Q=$(tail -1 "$QUARTS_FILE" | awk '{printf "%.1f ± %.1f", $3, $4}')
    
    echo "pairs = ${LAST_P}   quartets = ${LAST_Q}"
done

echo ""
echo "Data files:"
echo "  $PAIRS_FILE"
echo "  $QUARTS_FILE"

# ── Plot 1: Pairs vs N ────────────────────────────────────────────────────
# Columns: 1=N  2=gamma  3=mean_pairs  4=err  5=total
gnuplot <<GNUEOF
set terminal pngcairo enhanced size 1000,700 font "Arial,14"
set output "${OUTDIR}/plot_pairs_vs_N.png"

set xlabel "N  (number of spins)" font ",16"
set ylabel "Surviving pairs - N  (mean)" font ",16"
set title  "FMC: (Pairs surviving - N) vs N   ({/Symbol g} = 1/[2(N-1)],  samples = ${SAMPLES})" font ",18"
set grid
set key top left

plot "${PAIRS_FILE}" using 1:(\$3-\$1):4 with yerrorbars \
        lt rgb "#0072B2" pt 7 ps 0.8 lw 1.5 \
        title "mean - N {/Symbol \261} jackknife", \
     "${PAIRS_FILE}" using 1:(\$5-\$1) with lines \
        lt rgb "#999999" lw 1.2 dt 2 \
        title "C(N,2) - N"
GNUEOF

# ── Plot 2: Quartets vs N ─────────────────────────────────────────────────
gnuplot <<GNUEOF
set terminal pngcairo enhanced size 1000,700 font "Arial,14"
set output "${OUTDIR}/plot_quartets_vs_N.png"

set xlabel "N  (number of spins)" font ",16"
set ylabel "Surviving quartets  (mean)" font ",16"
set title  "FMC: Quartets surviving vs N   ({/Symbol g} = 1/[2(N-1)],  samples = ${SAMPLES})" font ",18"
set grid
set key top left

plot "${QUARTS_FILE}" using 1:3:4 with yerrorbars \
        lt rgb "#D55E00" pt 7 ps 0.8 lw 1.5 \
        title "mean {/Symbol \261} jackknife", \
     "${QUARTS_FILE}" using 1:5 with lines \
        lt rgb "#999999" lw 1.2 dt 2 \
        title "total C(N,4)"
GNUEOF

# ── Plot 3: Pairs vs gamma ────────────────────────────────────────────────
gnuplot <<GNUEOF
set terminal pngcairo enhanced size 1000,700 font "Arial,14"
set output "${OUTDIR}/plot_pairs_vs_gamma.png"

set xlabel "{/Symbol g}  (FMC bandwidth)" font ",16"
set ylabel "Surviving pairs  (mean)" font ",16"
set title  "FMC: Pairs surviving vs {/Symbol g}   ({/Symbol g} = 1/[2(N-1)],  samples = ${SAMPLES})" font ",18"
set grid
set key top left
set logscale x
set format x "10^{%T}"

plot "${PAIRS_FILE}" using 2:3:4 with yerrorbars \
        lt rgb "#0072B2" pt 7 ps 0.8 lw 1.5 \
        title "mean {/Symbol \261} jackknife", \
     "${PAIRS_FILE}" using 2:5 with lines \
        lt rgb "#999999" lw 1.2 dt 2 \
        title "total C(N,2)"
GNUEOF

# ── Plot 4: Quartets vs gamma ─────────────────────────────────────────────
gnuplot <<GNUEOF
set terminal pngcairo enhanced size 1000,700 font "Arial,14"
set output "${OUTDIR}/plot_quartets_vs_gamma.png"

set xlabel "{/Symbol g}  (FMC bandwidth)" font ",16"
set ylabel "Surviving quartets  (mean)" font ",16"
set title  "FMC: Quartets surviving vs {/Symbol g}   ({/Symbol g} = 1/[2(N-1)],  samples = ${SAMPLES})" font ",18"
set grid
set key top left
set logscale x
set format x "10^{%T}"

plot "${QUARTS_FILE}" using 2:3:4 with yerrorbars \
        lt rgb "#D55E00" pt 7 ps 0.8 lw 1.5 \
        title "mean {/Symbol \261} jackknife", \
     "${QUARTS_FILE}" using 2:5 with lines \
        lt rgb "#999999" lw 1.2 dt 2 \
        title "total C(N,4)"
GNUEOF

echo ""
echo "Plots:"
echo "  ${OUTDIR}/plot_pairs_vs_N.png"
echo "  ${OUTDIR}/plot_quartets_vs_N.png"
echo "  ${OUTDIR}/plot_pairs_vs_gamma.png"
echo "  ${OUTDIR}/plot_quartets_vs_gamma.png"
echo ""
echo "Done."
