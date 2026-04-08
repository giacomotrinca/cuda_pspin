#!/usr/bin/env bash
# Plot FMC survivors data: 6 plots, publication-quality style
set -euo pipefail

DIR="analysis/fmc_survivors"
PF="${DIR}/pairs_vs_N.dat"
QF="${DIR}/quartets_vs_N.dat"

# ── common style block injected into each gnuplot heredoc ──
read -r -d '' STYLE <<'STYLE_END' || true
set terminal pngcairo enhanced size 1100,750 font "Helvetica,15"
set border lw 1.5
set grid back lc rgb "#E0E0E0" lw 0.8 lt 1
set tics nomirror
set mxtics
set mytics
set key spacing 1.5 samplen 3 width 1 font ",14"
set bmargin 4.5
set lmargin 10
set rmargin 3
set tmargin 3
STYLE_END

# ── Plot 1: (Pairs - N) vs N ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_pairs_vs_N.png"
set xlabel "N" font ",17"
set ylabel "<P_{surv}> - N" font ",17"
set title  "FMC surviving pairs - N  vs  N      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top left opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "%g"
set xrange [15:135]
set yrange [7:80]

f(x) = a*x + b
fit f(x) "${PF}" using 1:(\$3-\$1):(\$4) yerrors via a, b

set print "${DIR}/fit_pairs_vs_N.txt"
print sprintf("pairs - N = %.4f * N + (%.4f)", a, b)
set print

plot "${PF}" using 1:(\$3-\$1):(\$4) with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#1B6AC9" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.3f N %+.2f", a, b)
EOF

# ── Plot 2: Quartets vs N ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_quartets_vs_N.png"
set xlabel "N" font ",17"
set ylabel "<Q_{surv}>" font ",17"
set title  "FMC surviving quartets  vs  N      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top left opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "10^{%T}"
set xrange [15:135]
set yrange [250:200000]

f(x) = a * x**b
a = 1.0; b = 3.0
fit f(x) "${QF}" using 1:3:4 yerrors via a, b

set print "${DIR}/fit_quartets_vs_N.txt"
print sprintf("quartets = %.6f * N^(%.4f)", a, b)
set print

plot "${QF}" using 1:3:4 with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#E8590C" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.4f  N^{%.3f}", a, b)
EOF

# ── Plot 3: (Pairs - N) vs gamma ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_pairs_vs_gamma.png"
set xlabel "{/Symbol g}" font ",17"
set ylabel "<P_{surv}> - N" font ",17"
set title  "FMC surviving pairs - N  vs  {/Symbol g}      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top right opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "%g"
set xrange [0.003:0.04]
set yrange [7:80]

f(x) = a/x + b
a = 1.0; b = 0.0
fit f(x) "${PF}" using 2:(\$3-\$1):(\$4) yerrors via a, b

set print "${DIR}/fit_pairs_vs_gamma.txt"
print sprintf("pairs - N = %.4f / gamma + (%.4f)", a, b)
set print

plot "${PF}" using 2:(\$3-\$1):(\$4) with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#1B6AC9" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.3f / {/Symbol g} %+.1f", a, b)
EOF

# ── Plot 4: Quartets vs gamma ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_quartets_vs_gamma.png"
set xlabel "{/Symbol g}" font ",17"
set ylabel "<Q_{surv}>" font ",17"
set title  "FMC surviving quartets  vs  {/Symbol g}      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top right opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "10^{%T}"
set xrange [0.003:0.04]
set yrange [250:200000]

f(x) = a * x**(-c)
a = 1.0; c = 3.0
fit f(x) "${QF}" using 2:3:4 yerrors via a, c

set print "${DIR}/fit_quartets_vs_gamma.txt"
print sprintf("quartets = %.6f * gamma^(-%.4f)", a, c)
set print

plot "${QF}" using 2:3:4 with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#E8590C" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.4f  {/Symbol g}^{-%.3f}", a, c)
EOF

# ── Plot 5: (Pairs - N)/N vs N ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_pairs_over_N_vs_N.png"
set xlabel "N" font ",17"
set ylabel "(<P_{surv}> - N) / N" font ",17"
set title  "(Pairs - N)/N  vs  N      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top right opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "%g"
set xrange [15:135]
set yrange [0.45:0.55]

f(x) = a + b/x
a = 0.5; b = 0.0
fit f(x) "${PF}" using 1:((\$3-\$1)/\$1):(\$4/\$1) yerrors via a, b

set print "${DIR}/fit_pairs_over_N_vs_N.txt"
print sprintf("(pairs - N)/N = %.6f + (%.4f)/N", a, b)
set print

plot "${PF}" using 1:((\$3-\$1)/\$1):(\$4/\$1) with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#1B6AC9" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.4f %+.2f/N", a, b)
EOF

# ── Plot 6: Quartets/N vs N ──
gnuplot <<EOF
${STYLE}
set output "${DIR}/plot_quartets_over_N_vs_N.png"
set xlabel "N" font ",17"
set ylabel "<Q_{surv}> / N" font ",17"
set title  "Quartets/N  vs  N      ({/Symbol g} = 1/[2(N{-}1)])" font ",17"
set key top left opaque box lc rgb "#808080" lw 0.5
set logscale xy
set format x "%g"
set format y "%g"
set xrange [15:135]
set yrange [13:1500]

f(x) = a * x**b
a = 0.05; b = 2.0
fit f(x) "${QF}" using 1:(\$3/\$1):(\$4/\$1) yerrors via a, b

set print "${DIR}/fit_quartets_over_N_vs_N.txt"
print sprintf("quartets/N = %.6f * N^(%.4f)", a, b)
set print

plot "${QF}" using 1:(\$3/\$1):(\$4/\$1) with yerrorbars \
        pt 7 ps 0.6 lw 1.2 lc rgb "#E8590C" title "MC data  (100 samples)", \
     f(x) lw 2.5 dt 2 lc rgb "#2D8B2D" \
        title sprintf("fit:  %.4f  N^{%.3f}", a, b)
EOF

echo "Done:"
echo "  ${DIR}/plot_pairs_vs_N.png"
echo "  ${DIR}/plot_quartets_vs_N.png"
echo "  ${DIR}/plot_pairs_vs_gamma.png"
echo "  ${DIR}/plot_quartets_vs_gamma.png"
echo "  ${DIR}/plot_pairs_over_N_vs_N.png"
echo "  ${DIR}/plot_quartets_over_N_vs_N.png"
