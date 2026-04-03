#!/bin/bash
# ===========================================================================
# run_tests.sh — Build, run, and visualise the full test suite
#
# Usage:  ./run_tests.sh [--dev=0] [visnu=1]
#
# • --dev=N  selects CUDA device (default 0)
# • Detects GPU name via nvidia-smi
# • Creates  TESTS_<gpu_name>/  output directory
# • Builds & runs all 9 tests, collecting data and logs
# • Generates publication-quality gnuplot plots for each test
# • Produces a one-page summary (summary.txt + summary.png)
#
# Exit code: 0 if all pass, 1 if any fail, 2 if build error.
# ===========================================================================

set -e

# ── parse --dev=N ──────────────────────────────────────────────────────────
DEV=0
MAKE_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --dev=*) DEV="${arg#--dev=}" ;;
        *)       MAKE_ARGS="$MAKE_ARGS $arg" ;;
    esac
done
export CUDA_VISIBLE_DEVICES="$DEV"

# ── colours ────────────────────────────────────────────────────────────────
GREEN='\033[1;32m'
RED='\033[1;31m'
YEL='\033[1;33m'
CYN='\033[1;36m'
NC='\033[0m'

# ── GPU detection ──────────────────────────────────────────────────────────
GPU_RAW=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$GPU_RAW" ]; then
    GPU_RAW="UnknownGPU"
fi
GPU_NAME=$(echo "$GPU_RAW" | sed 's/ /_/g; s/[^A-Za-z0-9_.-]//g')

OUTDIR="TESTS_${GPU_NAME}"
mkdir -p "$OUTDIR"

export TEST_OUTDIR="$OUTDIR"

echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYN}  p-Spin 2+4 :: Test Suite + Validation Plots${NC}"
echo -e "${CYN}  GPU:  ${GPU_RAW}  (device ${DEV})${NC}"
echo -e "${CYN}  Output directory:  ${OUTDIR}/${NC}"
echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo ""

# ── build ──────────────────────────────────────────────────────────────────
echo -e "${YEL}Building tests...  (make tests ${MAKE_ARGS})${NC}"
if ! make tests ${MAKE_ARGS} 2>&1 | tee "${OUTDIR}/build.log"; then
    echo -e "${RED}BUILD FAILED — see ${OUTDIR}/build.log${NC}"
    exit 2
fi
echo ""

# ── test runner ────────────────────────────────────────────────────────────
PASS=0; FAIL=0
declare -a NAMES STATUS

run_one() {
    local name="$1"; shift
    local binary="$1"; shift
    local args="$@"

    NAMES+=("$name")
    echo -e "${CYN}────────────────────────────────────────${NC}"
    echo -e "  Running: ${YEL}${name}${NC}"
    echo -e "  Command: ./${binary} ${args}"
    echo -e "${CYN}────────────────────────────────────────${NC}"

    if ./${binary} ${args} 2>&1 | tee "${OUTDIR}/${name}.log"; then
        STATUS+=("PASS")
        PASS=$((PASS + 1))
    else
        STATUS+=("FAIL")
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

run_one  quartet_index      bin/test_quartet_index      20
run_one  spherical           bin/test_spherical           16 4 500
run_one  delta_e             bin/test_delta_e             10 4 200
run_one  inf_temp            bin/test_inf_temp            12 8 500
run_one  detailed_balance    bin/test_detailed_balance    8 8 500 1000
run_one  fmc_mask            bin/test_fmc_mask            10 1.0 1
run_one  replica_exchange    bin/test_replica_exchange    8 6 65536 5
run_one  sparse_dense        bin/test_sparse_dense        10 4 300
run_one  mean_shift          bin/test_mean_shift          12 0.5

# ── summary text ───────────────────────────────────────────────────────────
{
    echo "═══════════════════════════════════════════════"
    echo "  TEST SUITE SUMMARY"
    echo "  GPU: ${GPU_RAW}"
    echo "  Date: $(date)"
    echo "═══════════════════════════════════════════════"
    for i in "${!NAMES[@]}"; do
        printf "  %-22s  %s\n" "${NAMES[$i]}" "${STATUS[$i]}"
    done
    echo ""
    echo "  ${PASS} passed,  ${FAIL} failed  (total ${#NAMES[@]})"
    echo "═══════════════════════════════════════════════"
} | tee "${OUTDIR}/summary.txt"

# ══════════════════════════════════════════════════════════════════════════
# ── PLOT GENERATION ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${YEL}Generating plots...${NC}"

GP_TERM='set terminal pngcairo size 1200,800 enhanced font "Helvetica,14" lw 2'
GP_GRID='set grid lc rgb "#e0e0e0" lt 1; set border lw 1.5'
COL_PASS='#27ae60'
COL_A='#2980b9'
COL_B='#e74c3c'
COL_FILL='#d5f5e3'
COL_WARN='#f39c12'

# ── 1. Quartet index (parsed from log) ────────────────────────────────────
grep -E '^\s+N=' "${OUTDIR}/quartet_index.log" | \
    sed 's/[^0-9]*//' | awk '{print $1, $3, ($4=="PASS"?1:0)}' \
    > "${OUTDIR}/quartet_index.dat" 2>/dev/null || true

if [ -s "${OUTDIR}/quartet_index.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_quartet_index.png"
${GP_GRID}
set title "Test 1: Quartet Index Round-Trip — GPU: ${GPU_RAW}" font ",16"
set xlabel "N (number of modes)"
set ylabel "C(N,4) quartets tested"
set logscale y
set style fill solid 0.7 border -1
set boxwidth 0.6
set key noautotitle
plot "${OUTDIR}/quartet_index.dat" using 1:2 with boxes lc rgb "${COL_PASS}" title "All round-trips verified"
EOF
echo "  ✓ plot_quartet_index.png"
fi

# ── 2. Spherical constraint ───────────────────────────────────────────────
if [ -s "${OUTDIR}/spherical.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_spherical.png"
${GP_GRID}
set title "Test 2: Spherical Constraint |Σ|a_k|² − N| — GPU: ${GPU_RAW}" font ",16"
set xlabel "MC sweep"
set ylabel "max |Σ|a_k|² − N| across replicas"
set logscale y
set format y "10^{%T}"
set key top right
set yrange [1e-16:*]
plot "${OUTDIR}/spherical.dat" using 1:2 with linespoints \
     pt 7 ps 1.2 lc rgb "${COL_A}" title "constraint error", \
     1e-8 with lines dt 2 lc rgb "${COL_B}" lw 1.5 title "tolerance (10^{-8})"
EOF
echo "  ✓ plot_spherical.png"
fi

# ── 3. Delta-E consistency ────────────────────────────────────────────────
if [ -s "${OUTDIR}/delta_e.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_delta_e.png"
${GP_GRID}
set title "Test 3: ΔE Tracking Accuracy — GPU: ${GPU_RAW}" font ",16"
set xlabel "MC sweep"
set ylabel "max |E_{stored} − E_{full}| / max(|E|,1)"
set logscale y
set format y "10^{%T}"
set yrange [1e-16:*]
set key top right
plot "${OUTDIR}/delta_e.dat" using 1:2 with linespoints \
     pt 7 ps 1.2 lc rgb "${COL_A}" title "energy drift", \
     1e-4 with lines dt 2 lc rgb "${COL_B}" lw 1.5 title "tolerance (10^{-4})"
EOF
echo "  ✓ plot_delta_e.png"
fi

# ── 4. Infinite temperature ───────────────────────────────────────────────
if [ -s "${OUTDIR}/inf_temp.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_inf_temp.png"
set multiplot layout 1,2 title "Test 4: Infinite Temperature (β→0) — GPU: ${GPU_RAW}" font ",16"
${GP_GRID}

# Left panel: acceptance rates
set title "Acceptance rates" font ",14"
set xlabel "Replica"
set ylabel "Acceptance rate"
set yrange [0.99:1.005]
set style fill solid 0.7 border -1
set boxwidth 0.6
set key noautotitle
plot "${OUTDIR}/inf_temp.dat" using 1:2 with boxes lc rgb "${COL_PASS}" title "rate", \
     1.0 with lines dt 2 lc rgb "${COL_B}" lw 2 title "ideal (1.0)"

# Right panel: energy/N
unset yrange
set title "Energy per mode E/N" font ",14"
set xlabel "Replica"
set ylabel "E/N"
set style fill solid 0.5 border -1
plot "${OUTDIR}/inf_temp.dat" using 1:3 with boxes lc rgb "${COL_A}" title "E/N", \
     0 with lines dt 2 lc rgb "${COL_B}" lw 2 title "expected (≈ 0)"

unset multiplot
EOF
echo "  ✓ plot_inf_temp.png"
fi

# ── 5. Detailed balance (convergence) ─────────────────────────────────────
if [ -s "${OUTDIR}/detailed_balance.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_detailed_balance.png"
${GP_GRID}
set title "Test 5: Detailed Balance — Independent Groups Converge — GPU: ${GPU_RAW}" font ",16"
set xlabel "Accumulated samples"
set ylabel "Running mean ⟨E/N⟩"
set key top right
plot "${OUTDIR}/detailed_balance.dat" using 1:2 with lines lw 2.5 lc rgb "${COL_A}" title "Group A (seed 11111)", \
     "${OUTDIR}/detailed_balance.dat" using 1:3 with lines lw 2.5 lc rgb "${COL_B}" title "Group B (seed 99999)"
EOF
echo "  ✓ plot_detailed_balance.png"
fi

# ── 6. FMC mask (parsed from log) ─────────────────────────────────────────
G4_MISMATCH=$(grep -oP 'g4 mask:\s+\K\d+' "${OUTDIR}/fmc_mask.log" 2>/dev/null || echo "?")
G4_QUARTETS=$(grep -oP '\d+(?= quartets)' "${OUTDIR}/fmc_mask.log" 2>/dev/null | head -1 || echo "?")
G4_ACTIVE=$(grep -oP 'GPU=\K\d+' "${OUTDIR}/fmc_mask.log" 2>/dev/null | head -1 || echo "?")
G2_MISMATCH=$(grep -oP 'g2 fmc:\s+\K\d+' "${OUTDIR}/fmc_mask.log" 2>/dev/null || echo "?")
G2_PAIRS=$(grep -oP 'pairs=\K\d+' "${OUTDIR}/fmc_mask.log" 2>/dev/null || echo "?")

cat > "${OUTDIR}/fmc_mask.dat" <<FMCEOF
# type total_tested mismatches
g4_quartets ${G4_QUARTETS:-0} ${G4_MISMATCH:-0}
g2_pairs ${G2_PAIRS:-0} ${G2_MISMATCH:-0}
FMCEOF

if [ "$G4_MISMATCH" != "?" ] && [ "$G2_MISMATCH" != "?" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_fmc_mask.png"
${GP_GRID}
set title "Test 6: FMC Mask — GPU vs CPU Bit-Level Comparison — GPU: ${GPU_RAW}" font ",16"
set xlabel ""
set ylabel "Count"
set style fill solid 0.7 border -1
set boxwidth 0.35
set xtics ("g4 quartets" 0, "g2 pairs" 1)
set key top right
set logscale y
set yrange [0.5:*]

# Data inline
plot "-" using 1:2 with boxes lc rgb "${COL_PASS}" title "Tested", \
     "-" using 1:2 with boxes lc rgb "${COL_B}" title "Mismatches"
0 ${G4_QUARTETS:-1}
1 ${G2_PAIRS:-1}
e
0 ${G4_MISMATCH:-0}.1
1 ${G2_MISMATCH:-0}.1
e
EOF
echo "  ✓ plot_fmc_mask.png"
fi

# ── 7. Replica exchange ───────────────────────────────────────────────────
if [ -s "${OUTDIR}/exchange_rates.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_exchange_rates.png"
${GP_GRID}
set title "Test 7a: Replica Exchange Rates — GPU: ${GPU_RAW}" font ",16"
set xlabel "Temperature T_{high}"
set ylabel "Swap acceptance rate"
set yrange [0:1.05]
set key top right
plot "${OUTDIR}/exchange_rates.dat" using 1:3 with linespoints \
     pt 7 ps 1.8 lc rgb "${COL_A}" lw 2.5 title "measured rate", \
     0 with lines dt 3 lc rgb "#cccccc" notitle, \
     1 with lines dt 3 lc rgb "#cccccc" notitle
EOF
echo "  ✓ plot_exchange_rates.png"
fi

if [ -s "${OUTDIR}/exchange_walk.dat" ]; then
cat > "${OUTDIR}/_walk.gp" <<WALKEOF
set terminal pngcairo size 1200,800 enhanced font "Helvetica,14" lw 2
set output "${OUTDIR}/plot_exchange_walk.png"
set grid lc rgb "#e0e0e0" lt 1
set border lw 1.5
set title "Test 7b: Temperature Random Walk — GPU: ${GPU_RAW}" font ",16"
set xlabel "MC sweep"
set ylabel "Temperature index"
set key top right box opaque
set yrange [-0.5:5.5]
set ytics 0,1,5
plot "< awk 'NR>1 && \$2==0 {print \$1, \$3}' ${OUTDIR}/exchange_walk.dat" u 1:2 w lines lw 1.8 lc rgb "#2980b9" t "replica 0", \
     "< awk 'NR>1 && \$2==1 {print \$1, \$3}' ${OUTDIR}/exchange_walk.dat" u 1:2 w lines lw 1.8 lc rgb "#e74c3c" t "replica 1", \
     "< awk 'NR>1 && \$2==2 {print \$1, \$3}' ${OUTDIR}/exchange_walk.dat" u 1:2 w lines lw 1.8 lc rgb "#27ae60" t "replica 2"
WALKEOF
gnuplot "${OUTDIR}/_walk.gp"
rm -f "${OUTDIR}/_walk.gp"
echo "  ✓ plot_exchange_walk.png"
fi

# ── 8. Sparse MC self-consistency ─────────────────────────────────────────
if [ -s "${OUTDIR}/sparse_dense.dat" ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_sparse_dense.png"
set multiplot layout 1,2 title "Test 8: Sparse MC Self-Consistency — GPU: ${GPU_RAW}" font ",16"
${GP_GRID}

# Left: cube constraint
set title "Smoothed-Cube Constraint" font ",14"
set xlabel "MC sweep"
set ylabel "max |Σ|a_k|⁴ − N|"
set logscale y
set format y "10^{%T}"
set yrange [1e-16:*]
set key top right
plot "${OUTDIR}/sparse_dense.dat" using 1:2 with linespoints \
     pt 7 ps 1.2 lc rgb "${COL_A}" title "cube error", \
     1e-8 with lines dt 2 lc rgb "${COL_B}" lw 1.5 title "tolerance"

# Right: energy tracking
set title "Energy Tracking |E_{stored}−E_{cpu}|/N" font ",14"
set xlabel "MC sweep"
set ylabel "max |ΔE|/N"
set key top right
plot "${OUTDIR}/sparse_dense.dat" using 1:(\$3>0?\$3:1e-16) with linespoints \
     pt 7 ps 1.2 lc rgb "${COL_PASS}" title "energy drift", \
     1e-3 with lines dt 2 lc rgb "${COL_B}" lw 1.5 title "tolerance"

unset multiplot
EOF
echo "  ✓ plot_sparse_dense.png"
fi

# ── 9. Mean shift (parsed from log) ───────────────────────────────────────
grep -E '^\s+(expected|actual)\s+shift' "${OUTDIR}/mean_shift.log" 2>/dev/null | \
    awk '/expected/{exp=$NF} /actual/{act=$NF; print NR/2, exp, act}' \
    > "${OUTDIR}/mean_shift.dat" 2>/dev/null || true

# Better parse: extract triplets of (expected, actual, error) per sub-test
{
    echo "# test_id expected actual error"
    awk '
        /--- g2 shift/       { tid=1 }
        /--- g4 shift ---/   { tid=2 }
        /--- g4 shift with FMC/ { tid=3 }
        /expected shift/     { exp=$NF }
        /actual shift/       { act=$NF }
        /error.*=.*[0-9]/    { err=$NF; sub(/e/,"e",err); if(tid>0) print tid, exp, act, err }
    ' "${OUTDIR}/mean_shift.log"
} > "${OUTDIR}/mean_shift.dat" 2>/dev/null || true

if [ -s "${OUTDIR}/mean_shift.dat" ] && [ "$(wc -l < "${OUTDIR}/mean_shift.dat")" -gt 1 ]; then
gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/plot_mean_shift.png"
${GP_GRID}
set title "Test 9: Mean Coupling Shift J₀ — GPU: ${GPU_RAW}" font ",16"
set xlabel ""
set ylabel "Shift value"
set style fill solid 0.5 border -1
set boxwidth 0.3
set xtics ("g2" 1, "g4" 2, "g4+FMC" 3)
set key top right
plot "${OUTDIR}/mean_shift.dat" using (\$1-0.17):2 with boxes lc rgb "${COL_A}" title "Expected", \
     "${OUTDIR}/mean_shift.dat" using (\$1+0.17):3 with boxes lc rgb "${COL_PASS}" title "Actual"
EOF
echo "  ✓ plot_mean_shift.png"
fi

# ══════════════════════════════════════════════════════════════════════════
# ── SUMMARY FIGURE ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

# Build inline data for summary bar
SUMMARY_DATA=""
for i in "${!NAMES[@]}"; do
    if [ "${STATUS[$i]}" == "PASS" ]; then
        SUMMARY_DATA+="$i 1 \"${NAMES[$i]}\"\n"
    else
        SUMMARY_DATA+="$i 0 \"${NAMES[$i]}\"\n"
    fi
done

gnuplot <<EOF
${GP_TERM}
set output "${OUTDIR}/summary.png"
set title "Test Suite Summary — GPU: ${GPU_RAW} — $(date '+%Y-%m-%d %H:%M')" font ",18"
set xlabel ""
set ylabel ""
set yrange [-0.5:1.5]
set style fill solid 0.85 border -1
set boxwidth 0.7
unset ytics
set grid noytics
set xtics rotate by -35
set key noautotitle
set label 1 "${PASS}/${#NAMES[@]} PASSED" at graph 0.02, graph 0.92 font ",16" tc rgb "${COL_PASS}"

plot "-" using 1:2:(\$2==1?0x27ae60:0xe74c3c):xtic(3) with boxes lc rgb variable
$(echo -e "${SUMMARY_DATA}")
e
EOF
echo "  ✓ summary.png"

# ── final ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo -e "  All outputs in:  ${YEL}${OUTDIR}/${NC}"
echo ""
if [ ${FAIL} -gt 0 ]; then
    echo -e "  ${RED}${FAIL} test(s) FAILED${NC}  —  see logs in ${OUTDIR}/"
    echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
    exit 1
else
    echo -e "  ${GREEN}ALL ${PASS} TESTS PASSED${NC}  ✓"
    echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
    exit 0
fi
