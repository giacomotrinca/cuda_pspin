// plot_benchmark.cpp — Reads benchmark TSV data and generates sciplot figures
//
// Reads bench_data/scaling_N.tsv, scaling_nrep.tsv, scaling_nrep_N32.tsv
// and produces publication-quality plots in bench_data/plots/.

#include <sciplot/sciplot.hpp>
using namespace sciplot;

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct Row {
    int N, nrep, nsweeps;
    double time_ms, ms_per_sweep, sweeps_per_sec, spin_updates_per_ns, mem_MB;
};

static std::vector<Row> read_tsv(const char* path) {
    std::vector<Row> rows;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); return rows; }
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        Row r;
        std::istringstream ss(line);
        char tab;
        ss >> r.N >> r.nrep >> r.nsweeps >> r.time_ms >> r.ms_per_sweep
           >> r.sweeps_per_sec >> r.spin_updates_per_ns >> r.mem_MB;
        if (ss) rows.push_back(r);
    }
    return rows;
}

static void setup_plot(Plot2D& plot) {
    plot.fontName("Times");
    plot.fontSize(18);
    plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
}

int main() {
    const char* plotdir = "bench_data/plots";

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
        plot.xlabel("{/Times-Italic N}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.drawCurve(vN, vS).lineColor("#1A33CC").lineWidth(2.5)
            .label("").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
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
        plot.xlabel("{/Times-Italic N}");
        plot.ylabel("ms / sweep");
        plot.legend().hide();
        plot.drawCurve(vN, vT).lineColor("#CC1A19").lineWidth(2.5)
            .label("").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/time_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 3: Spin updates / ns vs N ==========
    if (!dN.empty()) {
        std::vector<double> vN, vU;
        for (auto& r : dN) { vN.push_back(r.N); vU.push_back(r.spin_updates_per_ns); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Times-Italic N}");
        plot.ylabel("spin-pair updates / ns");
        plot.legend().hide();
        plot.drawCurve(vN, vU).lineColor("#1AB580").lineWidth(2.5)
            .label("").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
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
        plot.xlabel("{/Times-Italic N}");
        plot.ylabel("GPU memory (MB)");
        plot.legend().hide();
        plot.gnuplot("set logscale y 10");
        plot.drawCurve(vN, vM).lineColor("#CC9919").lineWidth(2.5)
            .label("").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/memory_vs_N.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 5: Sweeps/s vs nrep (N=18) ==========
    if (!dR.empty()) {
        std::vector<double> vR, vS;
        for (auto& r : dR) { vR.push_back(r.nrep); vS.push_back(r.sweeps_per_sec); }

        Plot2D plot;
        setup_plot(plot);
        plot.xlabel("{/Times-Italic n}_{rep}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurve(vR, vS).lineColor("#1A33CC").lineWidth(2.5)
            .label("N = 18").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
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
        plot.xlabel("{/Times-Italic n}_{rep}");
        plot.ylabel("ms / sweep");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurve(vR, vT).lineColor("#CC1A19").lineWidth(2.5)
            .label("N = 18").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
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
        plot.xlabel("{/Times-Italic n}_{rep}");
        plot.ylabel("sweeps / s");
        plot.legend().hide();
        plot.gnuplot("set logscale x 2");
        plot.drawCurve(vR, vS).lineColor("#1A33CC").lineWidth(2.5)
            .label("N = 32").pointType(7).pointSize(1.5);

        Figure fig = {{plot}};
        Canvas canvas = {{fig}};
        canvas.size(1200, 800);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/sweeps_vs_nrep_N32.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    // ========== Plot 8: Combined scaling overview ==========
    if (!dN.empty() && !dR.empty()) {
        // Left: sweeps/s vs N ; Right: sweeps/s vs nrep
        Plot2D p1;
        setup_plot(p1);
        p1.xlabel("{/Times-Italic N}");
        p1.ylabel("sweeps / s");
        p1.legend().hide();
        {
            std::vector<double> vN, vS;
            for (auto& r : dN) { vN.push_back(r.N); vS.push_back(r.sweeps_per_sec); }
            p1.drawCurve(vN, vS).lineColor("#1A33CC").lineWidth(2.5)
                .label("").pointType(7).pointSize(1.5);
        }

        Plot2D p2;
        setup_plot(p2);
        p2.xlabel("{/Times-Italic n}_{rep}");
        p2.ylabel("sweeps / s");
        p2.gnuplot("set logscale x 2");
        {
            std::vector<double> vR18, vS18, vR32, vS32;
            for (auto& r : dR)   { vR18.push_back(r.nrep); vS18.push_back(r.sweeps_per_sec); }
            for (auto& r : dR32) { vR32.push_back(r.nrep); vS32.push_back(r.sweeps_per_sec); }
            p2.drawCurve(vR18, vS18).lineColor("#1A33CC").lineWidth(2.5)
                .label("N = 18").pointType(7).pointSize(1.5);
            if (!vR32.empty())
                p2.drawCurve(vR32, vS32).lineColor("#CC1A19").lineWidth(2.5)
                    .label("N = 32").pointType(5).pointSize(1.5);
            p2.legend().atTopRight().fontSize(14);
        }

        Figure fig = {{p1, p2}};
        Canvas canvas = {{fig}};
        canvas.size(2400, 900);
        char pf[512]; snprintf(pf, sizeof(pf), "%s/benchmark_overview.png", plotdir);
        canvas.save(pf);
        printf("  Written %s\n", pf);
    }

    printf("Plots complete.\n");
    return 0;
}
