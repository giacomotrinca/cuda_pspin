#!/bin/bash
# ===========================================================================
# run_hyperopt.sh — Hyperparameter optimisation for Sparse Parallel Tempering
#
# For each system size N, determines optimal:
#   iter (MC sweeps = 2^K), TMAX, TMIN, NT, NREP, NSAMPLES
#
# Strategy:
#   Phase 1  Quick pilot (short iter) to calibrate exchange rates vs NT
#   Phase 2  NT scan: find minimum NT giving ~25-35% exchange rate everywhere
#   Phase 3  Equilibration check: verify convergence at optimal NT
#   Phase 4  Emit recommended parameter table
#
# Physics parameters (fixed):
#   J, R (=J/J0), alpha, alpha0, FMC mode, log_temp
#
# Usage:  ./run_hyperopt.sh [OPTIONS]
#   --dev=D          GPU device (default: 0)
#   --J=J            Coupling (default: 2.0)
#   --R=R            J/J0 ratio, 'inf' for pure disorder (default: inf)
#   --alpha=A        4-body fraction (default: 0.5)
#   --alpha0=A0      4-body mean fraction (default: 0.5)
#   --fmc=M          FMC mode 0/1/2 (default: 2)
#   --gamma=G        FMC gamma (default: auto)
#   --nrep=R         Replicas per temperature for pilots (default: 2)
#   --nsamples=S     Disorder samples for pilot (default: 2)
#   --outdir=D       Output directory (default: hyperopt_results)
#   --quick          Reduce scan ranges for faster testing
#   --help           Show this help
# ===========================================================================

set -e
set -o pipefail

# ── Colours ────────────────────────────────────────────────────────────────
GREEN='\033[1;32m'; RED='\033[1;31m'; YEL='\033[1;33m'
CYN='\033[1;36m'; MAG='\033[1;35m'; NC='\033[0m'

# ── Default fixed physics ─────────────────────────────────────────────────
DEV=0
J=2.0
R="inf"
ALPHA=0.5
ALPHA0=0.5
FMC=2
GAMMA=""
NREP_PILOT=2
NSAMPLES_PILOT=2
OUTDIR="hyperopt_results"
QUICK=0

# ── System sizes to study ─────────────────────────────────────────────────
SIZES=(18 24 32 48 56 62 82 96 110 120)

# ── Parse arguments ───────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dev=*)       DEV="${arg#*=}" ;;
        --J=*)         J="${arg#*=}" ;;
        --R=*)         R="${arg#*=}" ;;
        --alpha=*)     ALPHA="${arg#*=}" ;;
        --alpha0=*)    ALPHA0="${arg#*=}" ;;
        --fmc=*)       FMC="${arg#*=}" ;;
        --gamma=*)     GAMMA="${arg#*=}" ;;
        --nrep=*)      NREP_PILOT="${arg#*=}" ;;
        --nsamples=*)  NSAMPLES_PILOT="${arg#*=}" ;;
        --outdir=*)    OUTDIR="${arg#*=}" ;;
        --quick)       QUICK=1 ;;
        --help)
            sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)  echo "Unknown: $arg"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$DEV"

# ── Compute J0 ────────────────────────────────────────────────────────────
if [ "$R" = "inf" ] || [ "$R" = "Inf" ] || [ "$R" = "INF" ]; then
    J0=0.0
else
    J0=$(awk "BEGIN {printf \"%.6g\", $J / $R}")
fi

# ── GPU detection ─────────────────────────────────────────────────────────
GPU_RAW=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM_MB=$(nvidia-smi -i "$DEV" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_RAW" ]; then GPU_RAW="UnknownGPU"; fi
if [ -z "$GPU_MEM_MB" ]; then GPU_MEM_MB=16000; fi

# ── Create output directory ───────────────────────────────────────────────
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
RECOM="$OUTDIR/recommended_params.txt"
LOG="$OUTDIR/hyperopt.log"

# ── Build ─────────────────────────────────────────────────────────────────
echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYN}  p-Spin 2+4 :: Sparse PT Hyperparameter Optimisation     ${NC}"
echo -e "${CYN}  GPU:  ${GPU_RAW}  (device ${DEV}, ~${GPU_MEM_MB} MB free)${NC}"
echo -e "${CYN}  Sizes: ${SIZES[*]}${NC}"
echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YEL}Building sparse PT binary...${NC}"
make pts 2>&1 | tail -3
echo ""

# ── Helper: estimate GPU memory (MB) for a run ────────────────────────────
estimate_mem_mb() {
    local n=$1 nt=$2 nrep=$3
    local total=$((nt * nrep))
    # g2: N*N * 16 bytes
    local mem_g2=$(( n * n * 16 ))
    # sparse quartets: N * 40 bytes (SparseQuartet)
    local mem_sq=$(( n * 40 ))
    # spins: total * N * 16
    local mem_sp=$(( total * n * 16 ))
    # rng: total * 64
    local mem_rng=$(( total * 64 ))
    # aux: total * 40
    local mem_aux=$(( total * 40 ))
    # temporary dense g4 at init: ~N^4/24 * (16+1) ~ rough peak
    # For smoothed cube, n_quartets formula: N*(N-1)*(N-2)*(N-3)/24
    local nq=$(python3 -c "print($n*($n-1)*($n-2)*($n-3)//24)" 2>/dev/null || echo 0)
    local mem_g4_tmp=$(( nq * 17 ))
    local peak=$(( mem_g2 + mem_g4_tmp + mem_sp + mem_rng + mem_aux + mem_sq ))
    echo $(( peak / 1048576 + 1 ))
}

# ── Helper: run a single pilot and extract exchange rates ─────────────────
#    Returns: min_exch_rate  mean_exch_rate  max_exch_rate  ms_per_sweep
run_pilot() {
    local n=$1 nt=$2 nrep=$3 iter_k=$4 tmax=$5 tmin=$6 seed=$7 label=$8
    local mc_iter=$((1 << iter_k))
    local pt_freq=1
    local save_freq=$((mc_iter / 4))
    if [ "$save_freq" -lt 1 ]; then save_freq=1; fi

    local pilotdir="$OUTDIR/pilot_N${n}_NT${nt}"
    mkdir -p "$pilotdir"

    local verbose_flag="-verbose 0"
    local fmc_flag=""
    if [ -n "$FMC" ]; then fmc_flag="-fmc $FMC"; fi
    local gamma_flag=""
    if [ -n "$GAMMA" ]; then gamma_flag="-gamma $GAMMA"; fi

    # Run the binary
    local logfile="$pilotdir/run_S${label}.log"
    ./bin/parallel_tempering_sparse \
        -N "$n" \
        -Tmax "$tmax" -Tmin "$tmin" -NT "$nt" \
        -J "$J" -J0 "$J0" -alpha "$ALPHA" -alpha0 "$ALPHA0" \
        -iter "$mc_iter" \
        -seed "$seed" \
        -nrep "$nrep" \
        -pt_freq "$pt_freq" \
        -save_freq "$save_freq" \
        -label "$label" \
        -dev 0 \
        $fmc_flag $gamma_flag \
        -log_temp \
        $verbose_flag \
        > "$logfile" 2>&1

    # Parse exchange rates from the Summary section
    local min_rate=1.0 max_rate=0.0 sum_rate=0.0 count=0
    while IFS= read -r line; do
        # Lines like: "  T[0]-T[1]  (1.6000-1.4500)  128/256 = 0.5000"
        local rate
        rate=$(echo "$line" | grep -oP '=\s*\K[0-9]+\.[0-9]+' || true)
        if [ -n "$rate" ]; then
            count=$((count + 1))
            sum_rate=$(awk "BEGIN {printf \"%.6f\", $sum_rate + $rate}")
            local cmp_min=$(awk "BEGIN {print ($rate < $min_rate) ? 1 : 0}")
            local cmp_max=$(awk "BEGIN {print ($rate > $max_rate) ? 1 : 0}")
            if [ "$cmp_min" -eq 1 ]; then min_rate=$rate; fi
            if [ "$cmp_max" -eq 1 ]; then max_rate=$rate; fi
        fi
    done < <(grep "T\[" "$logfile" 2>/dev/null || true)

    local mean_rate=0.0
    if [ "$count" -gt 0 ]; then
        mean_rate=$(awk "BEGIN {printf \"%.6f\", $sum_rate / $count}")
    fi

    # Parse ms/sweep from time.txt
    local datadir="data/PTS_N${n}_NT${nt}_NR${nrep}_S${label}"
    local ms_per_sweep=0.0
    if [ -f "$datadir/time.txt" ]; then
        ms_per_sweep=$(awk 'NR==2 {print $NF}' "$datadir/time.txt" 2>/dev/null || echo "0.0")
    fi

    # Parse MC acceptance at coldest temperature from energy_accept.txt
    local mc_acc_cold=0.0
    if [ -f "$datadir/energy_accept.txt" ]; then
        # Last data line, last temperature index (NT-1), second-to-last column = acc0
        mc_acc_cold=$(tail -1 "$datadir/energy_accept.txt" | awk '{print $(NF)}' 2>/dev/null || echo "0.0")
    fi

    echo "$min_rate $mean_rate $max_rate $ms_per_sweep $mc_acc_cold"
}

# ── Helper: heuristic starting parameters for size N ──────────────────────
get_heuristic_params() {
    local n=$1
    # Temperature window: scale with coupling
    local tmax="1.6"
    local tmin="0.05"

    # NT heuristic: NT ~ 4 * sqrt(N), clamped to [16, 128]
    local nt
    nt=$(python3 -c "import math; print(max(16, min(128, int(4*math.sqrt($n)))))")

    # iter heuristic: 2^K where K scales with N
    local iter_k
    if [ "$n" -le 32 ]; then
        iter_k=14
    elif [ "$n" -le 64 ]; then
        iter_k=16
    elif [ "$n" -le 96 ]; then
        iter_k=17
    else
        iter_k=18
    fi

    # NREP heuristic for production: scales mildly
    local nrep_prod
    if [ "$n" -le 32 ]; then
        nrep_prod=16
    elif [ "$n" -le 64 ]; then
        nrep_prod=8
    elif [ "$n" -le 96 ]; then
        nrep_prod=4
    else
        nrep_prod=2
    fi

    # NSAMPLES heuristic for production
    local ns_prod
    if [ "$n" -le 32 ]; then
        ns_prod=100
    elif [ "$n" -le 64 ]; then
        ns_prod=50
    elif [ "$n" -le 96 ]; then
        ns_prod=30
    else
        ns_prod=20
    fi

    echo "$tmax $tmin $nt $iter_k $nrep_prod $ns_prod"
}

# ══════════════════════════════════════════════════════════════════════════
#                            MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

echo -e "${CYN}Fixed physics:${NC}"
printf "  %-16s %s\n" "J" "$J"
printf "  %-16s %s\n" "R = J/J0" "$R"
printf "  %-16s %s\n" "J0 (computed)" "$J0"
printf "  %-16s %s\n" "alpha" "$ALPHA"
printf "  %-16s %s\n" "alpha0" "$ALPHA0"
printf "  %-16s %s\n" "FMC mode" "$FMC"
printf "  %-16s %s\n" "T schedule" "geometric (log_temp)"
printf "  %-16s %s\n" "pilot nrep" "$NREP_PILOT"
printf "  %-16s %s\n" "pilot nsamples" "$NSAMPLES_PILOT"
echo ""

# Header for recommendation file
cat > "$RECOM" <<'EOF'
# ═══════════════════════════════════════════════════════════════════════════
# Recommended hyperparameters for Sparse Parallel Tempering (p-Spin 2+4)
# Generated by run_hyperopt.sh
# ═══════════════════════════════════════════════════════════════════════════
#
# N      TMAX    TMIN    NT   iter_K  sweeps      NREP  NSAMPLES  pt_freq  save_freq  exch_rate_min  exch_rate_mean  ms/sweep
EOF

# Also write machine-readable TSV
RECOM_TSV="$OUTDIR/recommended_params.tsv"
printf "N\tTMAX\tTMIN\tNT\titer_K\tsweeps\tNREP\tNSAMPLES\tpt_freq\tsave_freq\texch_rate_min\texch_rate_mean\tms_per_sweep\n" > "$RECOM_TSV"

for N in "${SIZES[@]}"; do
    echo -e "${MAG}══════════════════════════════════════════════════════════${NC}"
    echo -e "${MAG}  N = ${N}${NC}"
    echo -e "${MAG}══════════════════════════════════════════════════════════${NC}"

    read TMAX TMIN NT_HEUR ITER_K NREP_PROD NS_PROD <<< $(get_heuristic_params $N)

    echo -e "  Heuristic: TMAX=${TMAX} TMIN=${TMIN} NT=${NT_HEUR} iter=2^${ITER_K} NREP_prod=${NREP_PROD} NSAMPLES=${NS_PROD}"

    # ── Phase 1: Memory check ─────────────────────────────────────────────
    mem_est=$(estimate_mem_mb $N $NT_HEUR $NREP_PILOT)
    echo -e "  Memory estimate (pilot): ~${mem_est} MB  (GPU free: ~${GPU_MEM_MB} MB)"

    if [ "$mem_est" -gt "$((GPU_MEM_MB * 9 / 10))" ]; then
        echo -e "  ${RED}SKIP: estimated memory exceeds 90% of GPU free memory${NC}"
        printf "# N=%d  SKIPPED (memory)\n" "$N" >> "$RECOM"
        continue
    fi

    # ── Phase 2: NT scan — find optimal NT ────────────────────────────────
    echo -e "\n  ${YEL}Phase 2: NT scan${NC}"

    # Generate NT values to test: from NT_HEUR/2 up to NT_HEUR*2, step ~20%
    NT_MIN_SCAN=$((NT_HEUR * 2 / 3))
    if [ "$NT_MIN_SCAN" -lt 8 ]; then NT_MIN_SCAN=8; fi
    NT_MAX_SCAN=$((NT_HEUR * 2))
    if [ "$NT_MAX_SCAN" -gt 200 ]; then NT_MAX_SCAN=200; fi

    if [ "$QUICK" -eq 1 ]; then
        NT_VALUES=($NT_MIN_SCAN $NT_HEUR $NT_MAX_SCAN)
    else
        # Generate ~6-8 values spaced geometrically
        NT_VALUES=()
        n_steps=6
        for i in $(seq 0 $((n_steps - 1))); do
            val=$(python3 -c "import math; print(int(round($NT_MIN_SCAN * ($NT_MAX_SCAN/$NT_MIN_SCAN)**($i/($n_steps-1)))))")
            # Avoid duplicate or too-close values
            if [ ${#NT_VALUES[@]} -eq 0 ] || [ "$val" -gt "${NT_VALUES[-1]}" ]; then
                NT_VALUES+=("$val")
            fi
        done
    fi

    echo -e "  NT values to test: ${NT_VALUES[*]}"

    # Short pilot iter for NT scan (2^10 = 1024 sweeps, enough for exchange rate statistics)
    PILOT_ITER_K=10
    if [ "$N" -gt 64 ]; then PILOT_ITER_K=11; fi

    BEST_NT=0
    BEST_MEAN_RATE=0.0
    BEST_MIN_RATE=0.0
    TARGET_MIN_RATE=0.20   # target: min exchange rate >= 20%
    TARGET_MAX_RATE=0.50   # don't waste temps if min rate > 50%

    SCAN_LOG="$OUTDIR/NT_scan_N${N}.tsv"
    printf "NT\tmin_rate\tmean_rate\tmax_rate\tms_per_sweep\tmc_acc_cold\n" > "$SCAN_LOG"

    for NT_TRY in "${NT_VALUES[@]}"; do
        # Memory check for this NT
        mem_try=$(estimate_mem_mb $N $NT_TRY $NREP_PILOT)
        if [ "$mem_try" -gt "$((GPU_MEM_MB * 9 / 10))" ]; then
            echo -e "    NT=$NT_TRY: ${RED}skip (memory ~${mem_try} MB)${NC}"
            continue
        fi

        # Run pilot for each disorder sample, average results
        sum_min=0.0; sum_mean=0.0; sum_max=0.0; sum_ms=0.0; sum_mc=0.0
        pilot_ok=0
        for S in $(seq 0 $((NSAMPLES_PILOT - 1))); do
            seed=$((42 + S * 1000 + N * 100))
            result=$(run_pilot $N $NT_TRY $NREP_PILOT $PILOT_ITER_K "$TMAX" "$TMIN" $seed $S 2>/dev/null || echo "0.0 0.0 0.0 0.0 0.0")
            read r_min r_mean r_max r_ms r_mc <<< "$result"
            sum_min=$(awk "BEGIN {printf \"%.6f\", $sum_min + $r_min}")
            sum_mean=$(awk "BEGIN {printf \"%.6f\", $sum_mean + $r_mean}")
            sum_max=$(awk "BEGIN {printf \"%.6f\", $sum_max + $r_max}")
            sum_ms=$(awk "BEGIN {printf \"%.4f\", $sum_ms + $r_ms}")
            sum_mc=$(awk "BEGIN {printf \"%.6f\", $sum_mc + $r_mc}")
            pilot_ok=$((pilot_ok + 1))
        done

        if [ "$pilot_ok" -eq 0 ]; then
            echo -e "    NT=$NT_TRY: ${RED}all pilots failed${NC}"
            continue
        fi

        avg_min=$(awk "BEGIN {printf \"%.4f\", $sum_min / $pilot_ok}")
        avg_mean=$(awk "BEGIN {printf \"%.4f\", $sum_mean / $pilot_ok}")
        avg_max=$(awk "BEGIN {printf \"%.4f\", $sum_max / $pilot_ok}")
        avg_ms=$(awk "BEGIN {printf \"%.4f\", $sum_ms / $pilot_ok}")
        avg_mc=$(awk "BEGIN {printf \"%.4f\", $sum_mc / $pilot_ok}")

        printf "%d\t%s\t%s\t%s\t%s\t%s\n" "$NT_TRY" "$avg_min" "$avg_mean" "$avg_max" "$avg_ms" "$avg_mc" >> "$SCAN_LOG"

        # Status colour based on min exchange rate
        local_col="$RED"
        meets_target=$(awk "BEGIN {print ($avg_min >= $TARGET_MIN_RATE) ? 1 : 0}")
        if [ "$meets_target" -eq 1 ]; then local_col="$GREEN"; fi

        echo -e "    NT=${NT_TRY}: min_exch=${local_col}${avg_min}${NC}  mean_exch=${avg_mean}  ms/sw=${avg_ms}  mc_cold=${avg_mc}"

        # Select: smallest NT where min_rate >= TARGET
        if [ "$meets_target" -eq 1 ]; then
            if [ "$BEST_NT" -eq 0 ]; then
                # First NT that meets target — this is the optimum (smallest)
                BEST_NT=$NT_TRY
                BEST_MEAN_RATE=$avg_mean
                BEST_MIN_RATE=$avg_min
                BEST_MS=$avg_ms
            fi
        fi
    done

    # If no NT met the target, use the largest tried
    if [ "$BEST_NT" -eq 0 ]; then
        BEST_NT=${NT_VALUES[-1]}
        echo -e "  ${YEL}WARNING: No NT reached target min_rate=${TARGET_MIN_RATE}. Using largest NT=${BEST_NT}${NC}"
        # Re-read its rates from the scan log
        last_line=$(tail -1 "$SCAN_LOG")
        BEST_MIN_RATE=$(echo "$last_line" | awk '{print $2}')
        BEST_MEAN_RATE=$(echo "$last_line" | awk '{print $3}')
        BEST_MS=$(echo "$last_line" | awk '{print $5}')
    fi

    echo -e "  ${GREEN}Optimal NT = ${BEST_NT}  (min_exch=${BEST_MIN_RATE}, mean_exch=${BEST_MEAN_RATE})${NC}"

    # ── Phase 3: Equilibration check ──────────────────────────────────────
    echo -e "\n  ${YEL}Phase 3: Equilibration check${NC}"

    # Run a longer simulation at the optimal NT
    EQUIL_ITER_K=$((PILOT_ITER_K + 3))  # 8x longer
    if [ "$EQUIL_ITER_K" -gt "$ITER_K" ]; then EQUIL_ITER_K=$ITER_K; fi
    equil_sweeps=$((1 << EQUIL_ITER_K))

    echo -e "  Running equilibration pilot: iter=2^${EQUIL_ITER_K} (${equil_sweeps} sweeps), NT=${BEST_NT}"

    equil_seed=$((12345 + N * 7))
    equil_result=$(run_pilot $N $BEST_NT $NREP_PILOT $EQUIL_ITER_K "$TMAX" "$TMIN" $equil_seed 99 2>/dev/null || echo "0.0 0.0 0.0 0.0 0.0")
    read eq_min eq_mean eq_max eq_ms eq_mc <<< "$equil_result"

    # Check energy drift at coldest T: compare first-half vs second-half average
    equil_datadir="data/PTS_N${N}_NT${BEST_NT}_NR${NREP_PILOT}_S99"
    equil_ok="?"
    if [ -f "$equil_datadir/energy_accept.txt" ]; then
        # Extract energy at coldest T (highest Tidx = NT-1)
        coldest_idx=$((BEST_NT - 1))
        cold_energies=$(grep -v "^#" "$equil_datadir/energy_accept.txt" | awk -v t="$coldest_idx" '$2 == t {print $4}')
        n_lines=$(echo "$cold_energies" | wc -l)
        if [ "$n_lines" -gt 4 ]; then
            half=$((n_lines / 2))
            e_first=$(echo "$cold_energies" | head -$half | awk '{s+=$1; n++} END {if(n>0) printf "%.6f", s/n; else print "NaN"}')
            e_second=$(echo "$cold_energies" | tail -$half | awk '{s+=$1; n++} END {if(n>0) printf "%.6f", s/n; else print "NaN"}')
            drift=$(awk "BEGIN {d = ($e_first - $e_second); if(d<0) d=-d; printf \"%.6f\", d}")
            drift_rel=$(awk "BEGIN {
                ref = ($e_second < 0) ? -$e_second : $e_second
                if (ref < 1e-10) ref = 1e-10
                printf \"%.4f\", $drift / ref
            }")

            echo -e "  E_cold/N: first_half=${e_first}  second_half=${e_second}  |drift|=${drift}  rel=${drift_rel}"

            converged=$(awk "BEGIN {print ($drift_rel < 0.05) ? 1 : 0}")
            if [ "$converged" -eq 1 ]; then
                equil_ok="CONVERGED"
                echo -e "  ${GREEN}Equilibration: OK (relative drift < 5%)${NC}"
            else
                equil_ok="NOT_CONVERGED"
                echo -e "  ${YEL}Equilibration: drift ${drift_rel} > 5% — need more sweeps for production${NC}"
                # Suggest doubling iter
                ITER_K=$((ITER_K + 1))
                echo -e "  ${YEL}  → Bumping recommended iter to 2^${ITER_K}${NC}"
            fi
        else
            equil_ok="TOO_FEW_POINTS"
            echo -e "  ${YEL}Too few data points for drift analysis${NC}"
        fi
    fi

    # ── Phase 4: Final recommendation ─────────────────────────────────────
    SWEEPS_PROD=$((1 << ITER_K))
    PT_FREQ_PROD=1
    SAVE_FREQ_PROD=64

    echo -e "\n  ${CYN}── Recommendation for N=${N} ──${NC}"
    printf "  %-18s %s\n" "TMAX" "$TMAX"
    printf "  %-18s %s\n" "TMIN" "$TMIN"
    printf "  %-18s %d\n" "NT" "$BEST_NT"
    printf "  %-18s 2^%d = %d\n" "iter" "$ITER_K" "$SWEEPS_PROD"
    printf "  %-18s %d\n" "NREP" "$NREP_PROD"
    printf "  %-18s %d\n" "NSAMPLES" "$NS_PROD"
    printf "  %-18s %d\n" "pt_freq" "$PT_FREQ_PROD"
    printf "  %-18s %d\n" "save_freq" "$SAVE_FREQ_PROD"
    printf "  %-18s %s\n" "min_exch_rate" "$BEST_MIN_RATE"
    printf "  %-18s %s\n" "mean_exch_rate" "$BEST_MEAN_RATE"
    printf "  %-18s %s ms\n" "ms/sweep" "$BEST_MS"
    printf "  %-18s %s\n" "equil_status" "$equil_ok"
    echo ""

    # Write to recommendation file
    printf "  %-6d  %-6s  %-6s  %-4d  %-6d  %-10d  %-4d  %-8d  %-8d  %-10d  %-14s  %-14s  %s\n" \
        "$N" "$TMAX" "$TMIN" "$BEST_NT" "$ITER_K" "$SWEEPS_PROD" \
        "$NREP_PROD" "$NS_PROD" "$PT_FREQ_PROD" "$SAVE_FREQ_PROD" \
        "$BEST_MIN_RATE" "$BEST_MEAN_RATE" "$BEST_MS" >> "$RECOM"

    # TSV row
    printf "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n" \
        "$N" "$TMAX" "$TMIN" "$BEST_NT" "$ITER_K" "$SWEEPS_PROD" \
        "$NREP_PROD" "$NS_PROD" "$PT_FREQ_PROD" "$SAVE_FREQ_PROD" \
        "$BEST_MIN_RATE" "$BEST_MEAN_RATE" "$BEST_MS" >> "$RECOM_TSV"

    # Clean up pilot data to save disk space
    rm -rf data/PTS_N${N}_NT*_NR${NREP_PILOT}_S*

done

# ══════════════════════════════════════════════════════════════════════════
#  Generate launch script for production runs
# ══════════════════════════════════════════════════════════════════════════

LAUNCH="$OUTDIR/launch_production.sh"
cat > "$LAUNCH" <<'HEADER'
#!/bin/bash
# ===========================================================================
# launch_production.sh — Auto-generated by run_hyperopt.sh
#
# Launches production Sparse PT campaigns with optimised hyperparameters.
# Uses the sparse model with geometric (log) temperature schedule.
#
# Usage:  ./launch_production.sh [--dev=D] [--dry-run]
# ===========================================================================

set -e
DEV=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        --dev=*) DEV="${arg#*=}" ;;
        --dry-run) DRY=1 ;;
    esac
done
export CUDA_VISIBLE_DEVICES="$DEV"

HEADER

# Append a launch command for each N
while IFS=$'\t' read -r LN LTMAX LTMIN LNT LITERK LSWEEPS LNREP LNSAMPLES LPTF LSAVEF LEXMIN LEXMEAN LMS; do
    [ "$LN" = "N" ] && continue  # skip header
    cat >> "$LAUNCH" <<ENTRY
echo "════════════════════════════════════════"
echo "  N=${LN}  NT=${LNT}  iter=2^${LITERK}  NREP=${LNREP}  NSAMPLES=${LNSAMPLES}"
echo "════════════════════════════════════════"
if [ "\$DRY" -eq 0 ]; then
    ./scan_samples_pt \\
        --size=${LN} --tmax=${LTMAX} --tmin=${LTMIN} --nt=${LNT} \\
        --J=${J} --R=${R} --alpha=${ALPHA} --alpha0=${ALPHA0} \\
        --iter=${LITERK} --nrep=${LNREP} --nsamples=${LNSAMPLES} \\
        --pt_freq=${LPTF} --save_freq=${LSAVEF} \\
        --fmc=${FMC} --log-temp --sparse --dev=\$DEV --show-output=1
fi
echo ""

ENTRY
done < "$RECOM_TSV"

chmod +x "$LAUNCH"

echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYN}  HYPERPARAMETER OPTIMISATION COMPLETE${NC}"
echo -e "${CYN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Results:     ${GREEN}${RECOM}${NC}"
echo -e "  TSV:         ${GREEN}${RECOM_TSV}${NC}"
echo -e "  NT scans:    ${GREEN}${OUTDIR}/NT_scan_N*.tsv${NC}"
echo -e "  Launch:      ${GREEN}${LAUNCH}${NC}"
echo ""
echo -e "  ${YEL}Review the recommendation table, then run:${NC}"
echo -e "    ${CYN}./${LAUNCH}${NC}"
echo -e "  or with dry-run first:"
echo -e "    ${CYN}./${LAUNCH} --dry-run${NC}"
echo ""

# Print summary table to stdout
echo ""
echo "  N      TMAX    TMIN    NT   iter_K  sweeps      NREP  NSAMPLES  exch_min  exch_mean  ms/sw"
echo "  ─────  ──────  ──────  ───  ──────  ──────────  ────  ────────  ────────  ─────────  ─────"
while IFS=$'\t' read -r LN LTMAX LTMIN LNT LITERK LSWEEPS LNREP LNSAMPLES LPTF LSAVEF LEXMIN LEXMEAN LMS; do
    [ "$LN" = "N" ] && continue
    printf "  %-6s  %-6s  %-6s  %-3s  %-6s  %-10s  %-4s  %-8s  %-8s  %-9s  %s\n" \
        "$LN" "$LTMAX" "$LTMIN" "$LNT" "$LITERK" "$LSWEEPS" "$LNREP" "$LNSAMPLES" "$LEXMIN" "$LEXMEAN" "$LMS"
done < "$RECOM_TSV"
echo ""
