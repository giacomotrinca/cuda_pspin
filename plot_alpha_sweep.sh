#!/bin/bash

# Plot salient observables as a function of alpha from analysis_pt output.
# Reads equilibrium_data_mean.dat and parisi_glass_observables.dat
# from each analysis/{prefix}_N{N}_a{alpha}_R{R}_a0{alpha0}_NT{NT}_NR{nrep}/ directory.

# Default parameters — must match what was used in scan_alpha_pt
SIZE=18
NT=30
J=2.0
R="inf"
ALPHA0=0.5
NREP=16
SPARSE=0
ALPHA_LIST="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
OUTDIR="alpha_sweep"  # output directory for the sweep plots

for arg in "$@"; do
    case "$arg" in
        --size=*)    SIZE="${arg#*=}" ;;
        --nt=*)      NT="${arg#*=}" ;;
        --J=*)       J="${arg#*=}" ;;
        --R=*)       R="${arg#*=}" ;;
        --alpha0=*)  ALPHA0="${arg#*=}" ;;
        --nrep=*)    NREP="${arg#*=}" ;;
        --sparse)    SPARSE=1 ;;
        --alphas=*)  ALPHA_LIST="${arg#*=}" ;;
        --outdir=*)  OUTDIR="${arg#*=}" ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --size=N    --nt=N    --J=J    --R=R    --alpha0=A0    --nrep=N"
            echo "  --sparse    --alphas='0.0 0.2 0.5 0.8 1.0'    --outdir=DIR"
            exit 0
            ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

PREFIX="PT"
[ "$SPARSE" -eq 1 ] && PREFIX="PTS"

# Format float for directory names (same logic as C fmt_param)
fmt_param() { awk '{ s=sprintf("%.6g", $1); if (index(s,".")==0 && index(s,"e")==0) s=s".0"; print s }'; }

ALPHA0_TAG=$(echo "$ALPHA0" | fmt_param)
if [ "$R" = "inf" ] || [ "$R" = "Inf" ] || [ "$R" = "INF" ]; then
    J0=0.0
    R_TAG="inf"
else
    J0=$(awk "BEGIN {printf \"%.6g\", $J / $R}")
    R_TAG=$(echo "$R" | fmt_param)
fi

# Build analysis dir path for a given alpha
analysis_dir() {
    local a_tag
    a_tag=$(echo "$1" | fmt_param)
    echo "analysis/${PREFIX}_N${SIZE}_a${a_tag}_R${R_TAG}_a0${ALPHA0_TAG}_NT${NT}_NR${NREP}"
}

mkdir -p "$OUTDIR"

# ──────────────────────────────────────────────────────────────
# 1) Collect summary data: for each alpha, extract key quantities
# ──────────────────────────────────────────────────────────────
SUMMARY="$OUTDIR/alpha_summary.dat"
cat > "$SUMMARY" <<'HEADER'
# Alpha sweep summary
# Columns:
#  1: alpha
#  2: Cv_peak    (max specific heat)
#  3: Cv_err     (error at peak)
#  4: T_Cv_peak  (temperature of Cv peak)
#  5: E_lowT     (energy at lowest T)
#  6: E_lowT_err
#  7: chi_lowT   (SG susceptibility at lowest T)
#  8: chi_lowT_err
#  9: g4_lowT    (Binder param at lowest T)
# 10: g4_lowT_err
# 11: A_lowT     (non-self-averaging param at lowest T)
# 12: A_lowT_err
HEADER

for ALPHA in $ALPHA_LIST; do
    ADIR=$(analysis_dir "$ALPHA")
    EQFILE="$ADIR/equilibrium_data_mean.dat"
    GLFILE="$ADIR/parisi_glass_observables.dat"

    if [ ! -f "$EQFILE" ]; then
        echo "WARNING: $EQFILE not found — skipping alpha=$ALPHA"
        continue
    fi

    # Extract Cv peak (column 6/7) and lowest-T energy (column 2/3)
    # equilibrium_data_mean.dat: T  E  E_err  A  A_err  Cv  Cv_err
    read -r Cv_peak Cv_err T_peak <<< $(awk '!/^#/ {
        if ($6+0 > max+0) { max=$6; err=$7; tpk=$1 }
    } END { printf "%.8e %.8e %.8f", max, err, tpk }' "$EQFILE")

    read -r E_low E_low_err <<< $(awk '!/^#/ { if (NR==1 || $1+0 < tmin+0) { tmin=$1; e=$2; ee=$3 } }
        END { printf "%.8e %.8e", e, ee }' "$EQFILE")

    # Glass observables: T  chi  chi_err  g4  g4_err  A  A_err
    chi_low="NaN"; chi_low_err="NaN"
    g4_low="NaN"; g4_low_err="NaN"
    A_low="NaN"; A_low_err="NaN"
    if [ -f "$GLFILE" ]; then
        read -r chi_low chi_low_err g4_low g4_low_err A_low A_low_err <<< $(awk '!/^#/ {
            if (NR==1 || $1+0 < tmin+0) { tmin=$1; c=$2; ce=$3; g=$4; ge=$5; a=$6; ae=$7 }
        } END { printf "%.8e %.8e %.8e %.8e %.8e %.8e", c, ce, g, ge, a, ae }' "$GLFILE")
    fi

    printf "%.6g\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ALPHA" "$Cv_peak" "$Cv_err" "$T_peak" "$E_low" "$E_low_err" \
        "$chi_low" "$chi_low_err" "$g4_low" "$g4_low_err" "$A_low" "$A_low_err" >> "$SUMMARY"
done

echo "Written $SUMMARY"

# ──────────────────────────────────────────────────────────────
# 2) Collect per-temperature data for all alphas (for multi-curve plots)
# ──────────────────────────────────────────────────────────────
for ALPHA in $ALPHA_LIST; do
    ADIR=$(analysis_dir "$ALPHA")
    EQFILE="$ADIR/equilibrium_data_mean.dat"
    GLFILE="$ADIR/parisi_glass_observables.dat"
    PQFILE="$ADIR/parisi_overlap.dat"
    IFOFILE="$ADIR/ifo_overlap.dat"
    A_TAG=$(echo "$ALPHA" | fmt_param)

    # Copy equilibrium data
    if [ -f "$EQFILE" ]; then
        cp "$EQFILE" "$OUTDIR/eq_a${A_TAG}.dat"
    fi
    # Copy glass observables
    if [ -f "$GLFILE" ]; then
        cp "$GLFILE" "$OUTDIR/glass_a${A_TAG}.dat"
    fi
    # Copy Parisi overlap P(q)
    if [ -f "$PQFILE" ]; then
        cp "$PQFILE" "$OUTDIR/pq_a${A_TAG}.dat"
    fi
    # Copy IFO overlap P(C)
    if [ -f "$IFOFILE" ]; then
        cp "$IFOFILE" "$OUTDIR/ifo_a${A_TAG}.dat"
    fi
done

# ──────────────────────────────────────────────────────────────
# 3) gnuplot
# ──────────────────────────────────────────────────────────────

NALPHA=$(echo "$ALPHA_LIST" | wc -w)

# --- Plot: summary vs alpha ---
gnuplot -persist <<GNUPLOT_SUMMARY
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 1600,1200
set output "${OUTDIR}/alpha_sweep_summary.png"

set multiplot layout 2,2 title "N = ${SIZE},  R = ${R},  {/Symbol a}_0 = ${ALPHA0}" font "Helvetica-Bold,16"

set xlabel "{/Symbol a}" font "Helvetica,14"
set style data linespoints
set pointsize 1.2
set bars small
set grid lt 1 lc rgb "#E0E0E0" dt '.'

# (a) Cv peak
set ylabel "C_v^{peak}" font "Helvetica,14"
plot "${SUMMARY}" u 1:2:3 w yerrorbars pt 7 lc rgb "#2166AC" notitle

# (b) T of Cv peak
set ylabel "T_{peak}" font "Helvetica,14"
plot "${SUMMARY}" u 1:4 w linespoints pt 7 lc rgb "#B2182B" notitle

# (c) g_4 (lowest T)
set ylabel "g_4(T_{min})" font "Helvetica,14"
plot "${SUMMARY}" u 1:9:10 w yerrorbars pt 7 lc rgb "#1B7837" notitle

# (d) A (lowest T)
set ylabel "A(T_{min})" font "Helvetica,14"
plot "${SUMMARY}" u 1:11:12 w yerrorbars pt 7 lc rgb "#762A83" notitle

unset multiplot
GNUPLOT_SUMMARY
echo "Written ${OUTDIR}/alpha_sweep_summary.png"

# --- Plot: Cv(T) for all alpha ---
gnuplot -persist <<GNUPLOT_CV
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/Cv_vs_T.png"

set xlabel "T" font "Helvetica,14"
set ylabel "C_v" font "Helvetica,14"
set title "Specific heat  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'
set key right top font "Helvetica,11" box lw 0.5

set style data linespoints
set pointsize 0.7

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/eq_a${A_TAG}.dat"
    if [ -f "$FILE" ]; then
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:6:7 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        else
            echo "replot '${FILE}' u 1:6:7 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_CV
echo "Written ${OUTDIR}/Cv_vs_T.png"

# --- Plot: chi(T) for all alpha ---
gnuplot -persist <<GNUPLOT_CHI
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/chi_vs_T.png"

set xlabel "T" font "Helvetica,14"
set ylabel "{/Symbol c}_{SG}" font "Helvetica,14"
set title "SG susceptibility  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'
set key right top font "Helvetica,11" box lw 0.5

set style data linespoints
set pointsize 0.7

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/glass_a${A_TAG}.dat"
    if [ -f "$FILE" ]; then
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:2:3 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        else
            echo "replot '${FILE}' u 1:2:3 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_CHI
echo "Written ${OUTDIR}/chi_vs_T.png"

# --- Plot: g4(T) for all alpha ---
gnuplot -persist <<GNUPLOT_G4
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/g4_vs_T.png"

set xlabel "T" font "Helvetica,14"
set ylabel "g_4" font "Helvetica,14"
set title "Binder parameter  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'
set key right top font "Helvetica,11" box lw 0.5

set style data linespoints
set pointsize 0.7
set yrange [0:1]

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/glass_a${A_TAG}.dat"
    if [ -f "$FILE" ]; then
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:4:5 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        else
            echo "replot '${FILE}' u 1:4:5 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_G4
echo "Written ${OUTDIR}/g4_vs_T.png"

# --- Plot: Energy(T) for all alpha ---
gnuplot -persist <<GNUPLOT_E
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/E_vs_T.png"

set xlabel "T" font "Helvetica,14"
set ylabel "E / N" font "Helvetica,14"
set title "Energy per spin  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'
set key left bottom font "Helvetica,11" box lw 0.5

set style data linespoints
set pointsize 0.7

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/eq_a${A_TAG}.dat"
    if [ -f "$FILE" ]; then
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:2:3 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        else
            echo "replot '${FILE}' u 1:2:3 w yerrorbars title '{/Symbol a} = ${ALPHA}' lt $((idx+1))"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_E
echo "Written ${OUTDIR}/E_vs_T.png"

# ──────────────────────────────────────────────────────────────
# 4) P(Q) at three temperatures for all alphas (alpha on colorbar)
# ──────────────────────────────────────────────────────────────

# Determine min/max alpha for colorbar
ALPHA_MIN=$(echo "$ALPHA_LIST" | tr ' ' '\n' | sort -g | head -1)
ALPHA_MAX=$(echo "$ALPHA_LIST" | tr ' ' '\n' | sort -g | tail -1)

# Extract available temperatures from the first alpha's overlap file
FIRST_ALPHA=$(echo "$ALPHA_LIST" | awk '{print $1}')
FIRST_A_TAG=$(echo "$FIRST_ALPHA" | fmt_param)
FIRST_PQ="${OUTDIR}/pq_a${FIRST_A_TAG}.dat"

if [ -f "$FIRST_PQ" ]; then
    # Get unique temperatures from column 4 of non-comment, non-empty lines
    TEMPS_AVAIL=$(awk '!/^#/ && NF>=4 {print $4}' "$FIRST_PQ" | sort -gu)
    T_LOW=$(echo "$TEMPS_AVAIL" | head -1)
    T_HIGH=$(echo "$TEMPS_AVAIL" | tail -1)
    NT_AVAIL=$(echo "$TEMPS_AVAIL" | wc -l)
    MID_IDX=$(( (NT_AVAIL + 1) / 2 ))
    T_MID=$(echo "$TEMPS_AVAIL" | sed -n "${MID_IDX}p")

    echo ""
    echo "P(Q) plots: T_low=$T_LOW  T_mid=$T_MID  T_high=$T_HIGH"

    # Utility: extract lines with a given temperature (column 4) from an overlap file
    # Usage: extract_temp_block FILE TEMP > output
    # We match the temperature column to 6 decimals

    for TSEL_LABEL in low mid high; do
        case "$TSEL_LABEL" in
            low)  TSEL="$T_LOW" ;;
            mid)  TSEL="$T_MID" ;;
            high) TSEL="$T_HIGH" ;;
        esac

        # Extract per-alpha data at this temperature
        for ALPHA in $ALPHA_LIST; do
            A_TAG=$(echo "$ALPHA" | fmt_param)
            PQSRC="${OUTDIR}/pq_a${A_TAG}.dat"
            if [ -f "$PQSRC" ]; then
                awk -v T="$TSEL" '!/^#/ && NF>=4 && $4==T {print $1, $2, $3}' "$PQSRC" \
                    > "${OUTDIR}/pq_a${A_TAG}_T${TSEL_LABEL}.dat"
            fi
        done

        gnuplot -persist <<GNUPLOT_PQ
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/PQ_T${TSEL_LABEL}.png"

set xlabel "q" font "Helvetica,14"
set ylabel "P(q)" font "Helvetica,14"
set title "P(q) at T = ${TSEL}  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'

set palette defined (0 '#2166AC', 0.25 '#67A9CF', 0.5 '#8BBD57', 0.75 '#E78A62', 1.0 '#B2182B')
set cbrange [${ALPHA_MIN}:${ALPHA_MAX}]
set cblabel "{/Symbol a}" font "Helvetica,14" offset 1,0
set cbtics font "Helvetica,12" scale 0.4
set colorbox vertical user origin 0.88, 0.15 size 0.025, 0.7
set rmargin 14

set style data lines
set key off

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/pq_a${A_TAG}_T${TSEL_LABEL}.dat"
    if [ -f "$FILE" ] && [ -s "$FILE" ]; then
        # Map alpha to palette fraction
        if [ "$ALPHA_MAX" = "$ALPHA_MIN" ]; then
            FRAC="0.5"
        else
            FRAC=$(awk "BEGIN {printf \"%.6f\", ($ALPHA - $ALPHA_MIN) / ($ALPHA_MAX - $ALPHA_MIN)}")
        fi
        # Interpolate color from palette
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:2 w lines lw 2 lc palette frac ${FRAC} notitle"
        else
            echo "replot '${FILE}' u 1:2 w lines lw 2 lc palette frac ${FRAC} notitle"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_PQ
        echo "Written ${OUTDIR}/PQ_T${TSEL_LABEL}.png"
    done
else
    echo "No parisi_overlap.dat found — skipping P(Q) plots"
fi

# ──────────────────────────────────────────────────────────────
# 5) IFO P(C) at three temperatures for all alphas (alpha on colorbar)
# ──────────────────────────────────────────────────────────────

FIRST_IFO="${OUTDIR}/ifo_a${FIRST_A_TAG}.dat"

if [ -f "$FIRST_IFO" ]; then
    # Get temperatures from IFO file
    ITEMPS_AVAIL=$(awk '!/^#/ && NF>=4 {print $4}' "$FIRST_IFO" | sort -gu)
    IT_LOW=$(echo "$ITEMPS_AVAIL" | head -1)
    IT_HIGH=$(echo "$ITEMPS_AVAIL" | tail -1)
    INT_AVAIL=$(echo "$ITEMPS_AVAIL" | wc -l)
    IMID_IDX=$(( (INT_AVAIL + 1) / 2 ))
    IT_MID=$(echo "$ITEMPS_AVAIL" | sed -n "${IMID_IDX}p")

    echo ""
    echo "IFO P(C) plots: T_low=$IT_LOW  T_mid=$IT_MID  T_high=$IT_HIGH"

    for TSEL_LABEL in low mid high; do
        case "$TSEL_LABEL" in
            low)  TSEL="$IT_LOW" ;;
            mid)  TSEL="$IT_MID" ;;
            high) TSEL="$IT_HIGH" ;;
        esac

        for ALPHA in $ALPHA_LIST; do
            A_TAG=$(echo "$ALPHA" | fmt_param)
            IFOSRC="${OUTDIR}/ifo_a${A_TAG}.dat"
            if [ -f "$IFOSRC" ]; then
                awk -v T="$TSEL" '!/^#/ && NF>=4 && $4==T {print $1, $2, $3}' "$IFOSRC" \
                    > "${OUTDIR}/ifo_a${A_TAG}_T${TSEL_LABEL}.dat"
            fi
        done

        gnuplot -persist <<GNUPLOT_IFO
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/IFO_T${TSEL_LABEL}.png"

set xlabel "C" font "Helvetica,14"
set ylabel "P(C)" font "Helvetica,14"
set title "P(C) [IFO] at T = ${TSEL}  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'

set palette defined (0 '#2166AC', 0.25 '#67A9CF', 0.5 '#8BBD57', 0.75 '#E78A62', 1.0 '#B2182B')
set cbrange [${ALPHA_MIN}:${ALPHA_MAX}]
set cblabel "{/Symbol a}" font "Helvetica,14" offset 1,0
set cbtics font "Helvetica,12" scale 0.4
set colorbox vertical user origin 0.88, 0.15 size 0.025, 0.7
set rmargin 14

set style data lines
set key off

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/ifo_a${A_TAG}_T${TSEL_LABEL}.dat"
    if [ -f "$FILE" ] && [ -s "$FILE" ]; then
        if [ "$ALPHA_MAX" = "$ALPHA_MIN" ]; then
            FRAC="0.5"
        else
            FRAC=$(awk "BEGIN {printf \"%.6f\", ($ALPHA - $ALPHA_MIN) / ($ALPHA_MAX - $ALPHA_MIN)}")
        fi
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:2 w lines lw 2 lc palette frac ${FRAC} notitle"
        else
            echo "replot '${FILE}' u 1:2 w lines lw 2 lc palette frac ${FRAC} notitle"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_IFO
        echo "Written ${OUTDIR}/IFO_T${TSEL_LABEL}.png"
    done
else
    echo "No ifo_overlap.dat found — skipping IFO plots"
fi

# ──────────────────────────────────────────────────────────────
# 6) Cv vs T for all alphas with alpha on colorbar
# ──────────────────────────────────────────────────────────────

gnuplot -persist <<GNUPLOT_CV_CB
set encoding utf8
set terminal pngcairo enhanced font "Helvetica,14" size 900,650
set output "${OUTDIR}/Cv_vs_T_colorbar.png"

set xlabel "T" font "Helvetica,14"
set ylabel "C_v" font "Helvetica,14"
set title "Specific heat  (N=${SIZE}, R=${R})" font "Helvetica-Bold,15"
set grid lt 1 lc rgb "#E0E0E0" dt '.'
set key off

set palette defined (0 '#2166AC', 0.25 '#67A9CF', 0.5 '#8BBD57', 0.75 '#E78A62', 1.0 '#B2182B')
set cbrange [${ALPHA_MIN}:${ALPHA_MAX}]
set cblabel "{/Symbol a}" font "Helvetica,14" offset 1,0
set cbtics font "Helvetica,12" scale 0.4
set colorbox vertical user origin 0.88, 0.15 size 0.025, 0.7
set rmargin 14

set style data linespoints
set pointsize 0.7

$(
idx=0
for ALPHA in $ALPHA_LIST; do
    A_TAG=$(echo "$ALPHA" | fmt_param)
    FILE="${OUTDIR}/eq_a${A_TAG}.dat"
    if [ -f "$FILE" ]; then
        if [ "$ALPHA_MAX" = "$ALPHA_MIN" ]; then
            FRAC="0.5"
        else
            FRAC=$(awk "BEGIN {printf \"%.6f\", ($ALPHA - $ALPHA_MIN) / ($ALPHA_MAX - $ALPHA_MIN)}")
        fi
        if [ $idx -eq 0 ]; then
            echo "plot '${FILE}' u 1:6:7 w yerrorbars pt 7 ps 0.5 lc palette frac ${FRAC} notitle"
        else
            echo "replot '${FILE}' u 1:6:7 w yerrorbars pt 7 ps 0.5 lc palette frac ${FRAC} notitle"
        fi
        idx=$((idx+1))
    fi
done
)
GNUPLOT_CV_CB
echo "Written ${OUTDIR}/Cv_vs_T_colorbar.png"

echo ""
echo "Done — all plots in ${OUTDIR}/"
