// plot_benchmark.cpp — Reads benchmark TSV data and generates sciplot figures
//
// Reads bench_data/scaling_N.tsv, scaling_nrep.tsv, scaling_nrep_N32.tsv
// and produces publication-quality plots in bench_data/plots/.

#include <sciplot/sciplot.hpp>
using namespace sciplot;

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct Row {
    int N, nrep, nsweeps;
    double time_ms, ms_per_sweep, sweeps_per_sec, spin_updates_per_ns, mem_MB;
    double acceptance_rate, total_throughput;
};

static std::vector<Row> read_tsv(const char* path) {
    std::vector<Row> rows;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); return rows; }
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        Row r;
        r.acceptance_rate = 0; r.total_throughput = 0;
        std::istringstream ss(line);
        ss >> r.N >> r.nrep >> r.nsweeps >> r.time_ms >> r.ms_per_sweep
           >> r.sweeps_per_sec >> r.spin_updates_per_ns >> r.mem_MB;
        // New columns (optional for backward compat)
        ss >> r.acceptance_rate >> r.total_throughput;
        if (ss || r.N > 0) rows.push_back(r);
    }
    return rows;
}

// Okabe-Ito colorblind-safe palette
static const char* COL_BLUE   = "#0072B2";
static const char* COL_VERM   = "#D55E00";
static const char* COL_TEAL   = "#009E73";
static const char* COL_AMBER  = "#E69F00";
static const char* COL_ROSE   = "#CC79A7";
static const char* COL_SKY    = "#56B4E9";

static void setup_plot(Plot2D& plot, bool has_legend = false) {
    plot.fontName("Helvetica");
    plot.fontSize(15);
    // L-shaped border (bottom + left only)
    plot.gnuplot("set border 3 lw 1.4 lc rgb '#2D2D2D'");
    // Subtle grid behind data
    plot.gnuplot("set style line 100 lt 1 lc rgb '#E8E8E8' lw 0.6");
    plot.gnuplot("set grid back ls 100");
    // Tics: outward, bottom-left only
    plot.gnuplot("set tics nomirror out scale 0.6");
    plot.gnuplot("set tics font 'Helvetica,12'");
    // Margins
    plot.gnuplot("set lmargin 12");
    plot.gnuplot(has_legend ? "set rmargin 5" : "set rmargin 4");
    plot.gnuplot("set tmargin 2");
    plot.gnuplot("set bmargin 4.5");
    // Key/legend styling
    if (has_legend) {
        plot.gnuplot("set key opaque box lc rgb '#CCCCCC' lw 0.5");
        plot.gnuplot("set key spacing 1.3");
    } else {
        plot.gnuplot("set key off");
    }
}

// ---- Power-law fitting utilities ----

struct FitResult {
    double A, alpha, A_err, alpha_err, chi2, R2;
    int ndof;
    bool valid;
};

// Fit y = A * x^alpha via log-log linear regression (points with x >= x_cutoff)
static FitResult fit_power_law(const std::vector<double>& x, const std::vector<double>& y,
                                double x_cutoff) {
    FitResult fr = {0,0,0,0,0,0,0,false};
    std::vector<double> lx, ly;
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] >= x_cutoff && y[i] > 0) {
            lx.push_back(log(x[i]));
            ly.push_back(log(y[i]));
        }
    }
    int n = (int)lx.size();
    if (n < 3) return fr;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += lx[i]; sy += ly[i];
        sxx += lx[i]*lx[i]; sxy += lx[i]*ly[i];
    }
    double D = n*sxx - sx*sx;
    if (fabs(D) < 1e-30) return fr;
    double m = (n*sxy - sx*sy) / D;
    double b = (sy - m*sx) / n;
    double ss_res = 0, ss_tot = 0, mean_ly = sy / n;
    for (int i = 0; i < n; i++) {
        double ri = ly[i] - (b + m*lx[i]);
        ss_res += ri*ri;
        ss_tot += (ly[i] - mean_ly)*(ly[i] - mean_ly);
    }
    fr.alpha = m; fr.A = exp(b);
    fr.ndof = n - 2; fr.chi2 = ss_res;
    fr.R2 = (ss_tot > 0) ? 1.0 - ss_res/ss_tot : 0.0;
    if (n > 2) {
        double s2 = ss_res / (n - 2);
        fr.alpha_err = sqrt(s2 * n / D);
        fr.A_err = fr.A * sqrt(s2 * sxx / D);
    }
    fr.valid = true;
    return fr;
}

// Generate smooth power-law curve (log-spaced x)
static void fit_curve_log(const FitResult& fr, double x0, double x1, int npts,
                           std::vector<double>& xc, std::vector<double>& yc) {
    xc.resize(npts); yc.resize(npts);
    double lx0 = log(x0), lx1 = log(x1);
    for (int i = 0; i < npts; i++) {
        double t = (double)i / (npts - 1);
        xc[i] = exp(lx0 + t * (lx1 - lx0));
        yc[i] = fr.A * pow(xc[i], fr.alpha);
    }
}

// Add power-law fit curve and label to a plot
static void add_power_fit(Plot2D& plot, const std::vector<double>& x, const std::vector<double>& y,
                           double x_cutoff, const char* xname, const char* col,
                           int label_id, double gx, double gy, const char* align) {
    FitResult fr = fit_power_law(x, y, x_cutoff);
    if (!fr.valid) return;
    double xmax = x[0];
    for (size_t i = 1; i < x.size(); i++) if (x[i] > xmax) xmax = x[i];
    std::vector<double> xf, yf;
    fit_curve_log(fr, x_cutoff * 0.85, xmax * 1.1, 100, xf, yf);
    plot.drawCurve(xf, yf).lineColor(col).lineWidth(1.5).dashType(2).label("");
    char lbl[1024];
    snprintf(lbl, sizeof(lbl),
        "set label %d \""
        "fit (%s {/Symbol \\263} %.0f):\\n"
        "  y = A {/Symbol \\267} %s^{{/Symbol a}}\\n"
        "  A = %.3e {/Symbol \\261} %.1e\\n"
        "  {/Symbol a} = %.3f {/Symbol \\261} %.3f\\n"
        "  {/Symbol c}^2/dof = %.2e\\n"
        "  R^2 = %.6f"
        "\" at graph %.2f, graph %.2f %s font 'Courier,10' tc rgb '#555555'",
        label_id, xname, x_cutoff, xname,
        fr.A, fr.A_err, fr.alpha, fr.alpha_err,
        fr.ndof > 0 ? fr.chi2/fr.ndof : 0.0, fr.R2,
        gx, gy, align);
    plot.gnuplot(lbl);
}

// Add constant (saturation) fit line and label
static void add_constant_fit(Plot2D& plot, const std::vector<double>& x, const std::vector<double>& y,
                              double x_cutoff, const char* col, const char* tag,
                              int label_id, double gx, double gy, const char* align) {
    double s = 0; int cnt = 0;
    for (size_t i = 0; i < x.size(); i++)
        if (x[i] >= x_cutoff) { s += y[i]; cnt++; }
    if (cnt < 2) return;
    double mean = s / cnt;
    double ss = 0;
    for (size_t i = 0; i < x.size(); i++)
        if (x[i] >= x_cutoff) ss += (y[i] - mean)*(y[i] - mean);
    double stdev = sqrt(ss / (cnt - 1));
    double xmax = x[0];
    for (size_t i = 1; i < x.size(); i++) if (x[i] > xmax) xmax = x[i];
    std::vector<double> xf = {x_cutoff * 0.85, xmax * 1.1};
    std::vector<double> yf = {mean, mean};
    plot.drawCurve(xf, yf).lineColor(col).lineWidth(1.5).dashType(2).label("");
    char lbl[1024];
    snprintf(lbl, sizeof(lbl),
        "set label %d \""
        "fit %s (n_{rep} {/Symbol \\263} %.0f):\\n"
        "  y = A (const)\\n"
        "  A = %.1f {/Symbol \\261} %.1f"
        "\" at graph %.2f, graph %.2f %s font 'Courier,10' tc rgb '#555555'",
        label_id, tag, x_cutoff,
        mean, stdev,
        gx, gy, align);
    plot.gnuplot(lbl);
}

int main(int argc, char** argv) {
    const char* plotdir = (argc > 1) ? argv[1] : "bench_data/plots";

    // --- Read data ---
    auto dN    = read_tsv("bench_data/scaling_N.tsv");
    auto dR    = read_tsv("bench_data/scaling_nrep.tsv");
    auto dR32  = read_tsv("bench_data/scaling_nrep_N32.tsv");

    if (dN.empty() && dR.empty() && dR32.empty()) {
        fprintf(stderr, "No benchmark data found. Run bin/benchmark first.\n");
        return 1;
    }

    // ========== Plot 1: Sweeps/s vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vS;
        for (auto& r : dN) { vN.push_back(r.N); vS.push_back(r.sweeps_per_sec); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.drawCurveWithPoints(vN, vS).lineColor(COL_BLUE).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);
        add_power_fit(plot, vN, vS, 18, "N", COL_BLUE, 1, 0.95, 0.90, "right");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/sweeps_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 2: ms/sweep vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vT;
        for (auto& r : dN) { vN.push_back(r.N); vT.push_back(r.ms_per_sweep); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("ms / sweep");
        plot.legend().hide();
        plot.drawCurveWithPoints(vN, vT).lineColor(COL_VERM).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);
        add_power_fit(plot, vN, vT, 18, "N", COL_VERM, 1, 0.05, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/time_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 2b: Log-log ms/sweep vs N (power-law) ==========
    if (!dN.empty()) {
        std::vector<double> vN, vT;
        for (auto& r : dN) { vN.push_back(r.N); vT.push_back(r.ms_per_sweep); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("ms / sweep");
        plot.legend().hide();
        plot.gnuplot("set logscale xy 10");
        plot.gnuplot("set format x '10^{%L}'");
        plot.gnuplot("set format y '10^{%L}'");
        plot.drawCurveWithPoints(vN, vT).lineColor(COL_VERM).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);
        add_power_fit(plot, vN, vT, 18, "N", COL_VERM, 1, 0.05, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/time_vs_N_loglog.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 3: Spin updates / ns vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vU;
        for (auto& r : dN) { vN.push_back(r.N); vU.push_back(r.spin_updates_per_ns); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("spin-pair updates / ns");
        plot.legend().hide();
        plot.drawCurveWithPoints(vN, vU).lineColor(COL_TEAL).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);
        add_power_fit(plot, vN, vU, 18, "N", COL_TEAL, 1, 0.95, 0.90, "right");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/spin_updates_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 4: GPU memory vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vM;
        for (auto& r : dN) { vN.push_back(r.N); vM.push_back(r.mem_MB); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("GPU memory (MB)");
        plot.legend().hide();
        plot.gnuplot("set logscale y 10");
        plot.drawCurveWithPoints(vN, vM).lineColor(COL_AMBER).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);
        add_power_fit(plot, vN, vM, 18, "N", COL_AMBER, 1, 0.05, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/memory_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 4b: Acceptance rate vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vA;
        for (auto& r : dN) { vN.push_back(r.N); vA.push_back(r.acceptance_rate); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique N}");
        plot.ylabel("acceptance rate");
        plot.legend().hide();
        plot.drawCurveWithPoints(vN, vA).lineColor(COL_ROSE).lineWidth(2)
            .label("").pointType(7).pointSize(1.2);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/acceptance_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 5: Sweeps/s vs nrep (N=18) ==========
    if (!dR.empty()) {
        std::vector<double> vR, vS;
        for (auto& r : dR) { vR.push_back(r.nrep); vS.push_back(r.sweeps_per_sec); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique n}_{rep}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurveWithPoints(vR, vS).lineColor(COL_BLUE).lineWidth(2)
            .label("N = 18").pointType(7).pointSize(1.2);
        add_power_fit(plot, vR, vS, 32, "n_{rep}", COL_BLUE, 1, 0.55, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/sweeps_vs_nrep_N18.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 6: ms/sweep vs nrep (N=18) ==========
    if (!dR.empty()) {
        std::vector<double> vR, vT;
        for (auto& r : dR) { vR.push_back(r.nrep); vT.push_back(r.ms_per_sweep); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique n}_{rep}");
        plot.ylabel("ms / sweep");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurveWithPoints(vR, vT).lineColor(COL_VERM).lineWidth(2)
            .label("N = 18").pointType(7).pointSize(1.2);
        add_power_fit(plot, vR, vT, 32, "n_{rep}", COL_VERM, 1, 0.05, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/time_vs_nrep_N18.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 7: Sweeps/s vs nrep (N=32) ==========
    if (!dR32.empty()) {
        std::vector<double> vR, vS;
        for (auto& r : dR32) { vR.push_back(r.nrep); vS.push_back(r.sweeps_per_sec); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Helvetica-Oblique n}_{rep}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurveWithPoints(vR, vS).lineColor(COL_BLUE).lineWidth(2)
            .label("N = 32").pointType(7).pointSize(1.2);
        add_power_fit(plot, vR, vS, 32, "n_{rep}", COL_BLUE, 1, 0.55, 0.90, "left");

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/sweeps_vs_nrep_N32.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 8: Combined scaling overview ==========
    if (!dN.empty() && !dR.empty()) {
        // Left: sweeps/s vs N ; Right: sweeps/s vs nrep
        Plot2D p1;
        setup_plot(p1);
        p1.xlabel("{/Helvetica-Oblique N}");
        p1.ylabel("sweeps / s");
        p1.legend().hide();
        {
            std::vector<double> vN, vS;
            for (auto& r : dN) { vN.push_back(r.N); vS.push_back(r.sweeps_per_sec); }
            p1.drawCurveWithPoints(vN, vS).lineColor(COL_BLUE).lineWidth(2)
                .label("").pointType(7).pointSize(1.2);
        }

        Plot2D p2;
        setup_plot(p2, true);
        p2.xlabel("{/Helvetica-Oblique n}_{rep}");
        p2.ylabel("sweeps / s");
        p2.gnuplot("set logscale x 2");
        {
            std::vector<double> vR18, vS18, vR32, vS32;
            for (auto& r : dR)   { vR18.push_back(r.nrep); vS18.push_back(r.sweeps_per_sec); }
            for (auto& r : dR32) { vR32.push_back(r.nrep); vS32.push_back(r.sweeps_per_sec); }
            p2.drawCurveWithPoints(vR18, vS18).lineColor(COL_BLUE).lineWidth(2)
                .label("N = 18").pointType(7).pointSize(1.2);
            if (!vR32.empty())
                p2.drawCurveWithPoints(vR32, vS32).lineColor(COL_VERM).lineWidth(2)
                    .label("N = 32").pointType(5).pointSize(1.2);
            p2.legend().atTopRight().fontSize(12);
        }

        Figure fig = {{p1, p2}};
        Canvas canvas = {{fig}};
        canvas.size(1600, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/benchmark_overview.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 9: Total throughput vs nrep ==========
    if (!dR.empty()) {
        Plot2D plot;
        setup_plot(plot, true);
        plot.xlabel("{/Helvetica-Oblique n}_{rep}");
        plot.ylabel("{/Helvetica-Oblique n}_{rep} {/Symbol \\264} sweeps / s");
        plot.gnuplot("set logscale xy 2");
        {
            std::vector<double> vR, vT;
            for (auto& r : dR) { vR.push_back(r.nrep); vT.push_back(r.total_throughput); }
            plot.drawCurveWithPoints(vR, vT).lineColor(COL_BLUE).lineWidth(2)
                .label("N = 18").pointType(7).pointSize(1.2);
            add_constant_fit(plot, vR, vT, 128, COL_BLUE, "N=18", 1, 0.95, 0.60, "right");
        }
        if (!dR32.empty()) {
            std::vector<double> vR, vT;
            for (auto& r : dR32) { vR.push_back(r.nrep); vT.push_back(r.total_throughput); }
            plot.drawCurveWithPoints(vR, vT).lineColor(COL_VERM).lineWidth(2)
                .label("N = 32").pointType(5).pointSize(1.2);
            add_constant_fit(plot, vR, vT, 128, COL_VERM, "N=32", 2, 0.95, 0.20, "right");
        }
        plot.legend().atTopLeft().fontSize(12);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/total_throughput_vs_nrep.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 10: Parallel efficiency vs nrep ==========
    if (!dR.empty() && dR[0].nrep == 1) {
        double base18 = dR[0].sweeps_per_sec;
        Plot2D plot;
        setup_plot(plot, true);
        plot.xlabel("{/Helvetica-Oblique n}_{rep}");
        plot.ylabel("parallel efficiency");
        plot.gnuplot("set logscale x 2");
        plot.gnuplot("set yrange [0:*]");
        {
            std::vector<double> vR, vE;
            for (auto& r : dR) {
                vR.push_back(r.nrep);
                vE.push_back(r.sweeps_per_sec / base18);
            }
            plot.drawCurveWithPoints(vR, vE).lineColor(COL_BLUE).lineWidth(2)
                .label("N = 18").pointType(7).pointSize(1.2);
            add_power_fit(plot, vR, vE, 32, "n_{rep}", COL_BLUE, 1, 0.95, 0.70, "right");
        }
        if (!dR32.empty() && dR32[0].nrep == 1) {
            double base32 = dR32[0].sweeps_per_sec;
            std::vector<double> vR, vE;
            for (auto& r : dR32) {
                vR.push_back(r.nrep);
                vE.push_back(r.sweeps_per_sec / base32);
            }
            plot.drawCurveWithPoints(vR, vE).lineColor(COL_VERM).lineWidth(2)
                .label("N = 32").pointType(5).pointSize(1.2);
            add_power_fit(plot, vR, vE, 32, "n_{rep}", COL_VERM, 2, 0.95, 0.40, "right");
        }
        // Ideal line at 1.0
        plot.gnuplot("set arrow from graph 0,first 1 to graph 1,first 1 nohead lc rgb '#999999' lw 1 dt 4");
        plot.legend().atBottomLeft().fontSize(12);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(900, 600);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/parallel_efficiency.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    printf("Plots complete.\n");
    return 0;
}
