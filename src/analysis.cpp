#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <regex.h>
#include <sys/stat.h>
#include <sciplot/sciplot.hpp>

static std::string temp_color(double frac) {
    double r, g, b;
    if (frac < 0.33) {
        double t = frac / 0.33;
        r = 0.1 * t;         g = 0.2 + 0.5 * t;  b = 0.8 - 0.3 * t;
    } else if (frac < 0.66) {
        double t = (frac - 0.33) / 0.33;
        r = 0.1 + 0.7 * t;  g = 0.7 - 0.1 * t;  b = 0.5 - 0.4 * t;
    } else {
        double t = (frac - 0.66) / 0.34;
        r = 0.8 + 0.2 * t;  g = 0.6 - 0.5 * t;  b = 0.1 - 0.05 * t;
    }
    char hex[16];
    snprintf(hex, sizeof(hex), "#%02X%02X%02X",
             (int)(r*255), (int)(g*255), (int)(b*255));
    return hex;
}

// ================================================================
// Intensity spectrum helpers
// ================================================================

// Read frequencies from frequencies.txt in a data directory
static std::vector<double> read_frequencies(const char* datadir, int N) {
    char freqfile[512];
    snprintf(freqfile, sizeof(freqfile), "%s/frequencies.txt", datadir);
    std::vector<double> omega(N);
    FILE* ff = fopen(freqfile, "r");
    if (ff) {
        char line[256];
        while (fgets(line, sizeof(line), ff)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            int idx; double w;
            if (sscanf(line, "%d %lf", &idx, &w) == 2 && idx >= 0 && idx < N)
                omega[idx] = w;
        }
        fclose(ff);
    } else {
        for (int k = 0; k < N; k++)
            omega[k] = (N > 1) ? (double)k / (double)(N - 1) : 0.0;
    }
    return omega;
}

// Read a binary config (2N doubles: re0 im0 re1 im1 ...) and compute
// normalized intensities I_k = |a_k|^2 / sum_j |a_j|^2
static bool read_config_intensities(const char* filename, int N,
                                    std::vector<double>& Ik) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    std::vector<double> buf(2 * N);
    size_t nr = fread(buf.data(), sizeof(double), 2 * N, f);
    fclose(f);
    if ((int)nr != 2 * N) return false;
    Ik.resize(N);
    double total = 0;
    for (int k = 0; k < N; k++) {
        double re = buf[2*k], im = buf[2*k + 1];
        Ik[k] = re * re + im * im;
        total += Ik[k];
    }
    if (total > 0)
        for (int k = 0; k < N; k++) Ik[k] /= total;
    return true;
}

// Find config files matching conf_r{rep}_iter*.bin in confdir
static std::vector<std::string> find_configs_mc(const char* confdir, int rep) {
    std::vector<std::string> files;
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "conf_r%d_iter", rep);
    int plen = (int)strlen(prefix);
    DIR* dir = opendir(confdir);
    if (!dir) return files;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, prefix, plen) == 0) {
            const char* rest = ent->d_name + plen;
            if (strstr(rest, ".bin")) {
                char fp[768];
                snprintf(fp, sizeof(fp), "%s/%s", confdir, ent->d_name);
                files.push_back(fp);
            }
        }
    }
    closedir(dir);
    return files;
}

// ================================================================

struct DataPoint {
    int iter;
    std::vector<double> energy;  // E/N per replica
    std::vector<double> acc;     // acceptance per replica
};

// Read energy_accept.txt from a directory, return data points
static std::vector<DataPoint> read_data(const char* datadir, int nrep) {
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/energy_accept.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) return {};

    std::vector<DataPoint> data;
    char line[4096];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        DataPoint dp;
        dp.energy.resize(nrep);
        dp.acc.resize(nrep);

        char* tok = strtok(line, " \t\n");
        if (!tok) continue;
        dp.iter = atoi(tok);

        bool ok = true;
        for (int r = 0; r < nrep; r++) {
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            dp.energy[r] = atof(tok);
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            dp.acc[r] = atof(tok);
        }
        if (!ok) continue;
        data.push_back(dp);
    }
    fclose(fin);
    return data;
}

// Compute mean of last half of a time series for each replica
// Returns vector of size nrep
static std::vector<double> last_half_mean(const std::vector<DataPoint>& data, int nrep) {
    int M = (int)data.size();
    int start = M / 2;
    if (start >= M) start = 0;
    int n = M - start;
    std::vector<double> means(nrep, 0.0);
    for (int r = 0; r < nrep; r++) {
        double s = 0.0;
        for (int i = start; i < M; i++)
            s += data[i].energy[r];
        means[r] = s / n;
    }
    return means;
}

// Find all sample directories matching data/N{N}_NR{nrep}_S*
static std::vector<int> find_samples(int N, int nrep) {
    std::vector<int> labels;
    DIR* dir = opendir("data");
    if (!dir) return labels;

    char pat[128];
    snprintf(pat, sizeof(pat), "N%d_NR%d_S", N, nrep);
    int plen = strlen(pat);

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, pat, plen) == 0) {
            int label = atoi(ent->d_name + plen);
            // Verify it parses back correctly
            char check[128];
            snprintf(check, sizeof(check), "N%d_NR%d_S%d", N, nrep, label);
            if (strcmp(ent->d_name, check) == 0)
                labels.push_back(label);
        }
    }
    closedir(dir);
    std::sort(labels.begin(), labels.end());
    return labels;
}

static void mkdir_p(const char* path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>] [-label <l>] [-datadir <path>] [-T <temperature>] [--plot]\n", prog);
    fprintf(stderr, "  Without -label: averages over all samples S0, S1, ...\n");
    fprintf(stderr, "  With -label L:  analyzes single sample SL only\n");
    fprintf(stderr, "  -T: simulation temperature (default 1.0, used for spectrum output)\n");
    exit(1);
}

// ================================================================
// Block averaging for a single sample
// ================================================================
struct Block { int start; int end; };

static std::vector<Block> build_blocks(int M) {
    std::vector<Block> blocks;
    int pos = M;
    int bsize = M / 2;
    if (bsize < 1) bsize = 1;
    while (pos > 0) {
        int bstart = pos - bsize;
        if (bstart < 0) bstart = 0;
        blocks.push_back({bstart, pos});
        pos = bstart;
        bsize /= 2;
        if (bsize < 1) bsize = 1;
    }
    std::reverse(blocks.begin(), blocks.end());
    return blocks;
}

static void single_sample_analysis(const char* datadir, const std::vector<DataPoint>& data, int nrep) {
    int M = (int)data.size();
    auto blocks = build_blocks(M);
    int nblocks = (int)blocks.size();

    char outfile[512];
    snprintf(outfile, sizeof(outfile), "%s/block_energy.txt", datadir);
    FILE* fout = fopen(outfile, "w");
    if (!fout) {
        fprintf(stderr, "Cannot open %s for writing\n", outfile);
        return;
    }

    fprintf(fout, "# Block averaging of E/N (doubling block sizes, last block = last half)\n");
    fprintf(fout, "# Errors: jackknife\n");
    fprintf(fout, "# Columns: 1:block  2:iter_start  3:iter_end  4:n_samples");
    int col = 5;
    for (int r = 0; r < nrep; r++) {
        fprintf(fout, "  %d:<E%d/N>  %d:err%d", col, r, col+1, r);
        col += 2;
    }
    fprintf(fout, "\n");

    printf("  Block   iter_range          samples");
    for (int r = 0; r < nrep; r++) printf("    <E%d/N>      err%d   ", r, r);
    printf("\n");
    printf("  --------------------------------------------------------------");
    for (int r = 0; r < nrep; r++) printf("------------------------");
    printf("\n");

    for (int b = 0; b < nblocks; b++) {
        int s = blocks[b].start;
        int e = blocks[b].end;
        int n = e - s;

        fprintf(fout, "%d\t%d\t%d\t%d", b, data[s].iter, data[e-1].iter, n);
        printf("  %-6d  [%7d, %7d]  %6d ", b, data[s].iter, data[e-1].iter, n);

        for (int r = 0; r < nrep; r++) {
            double sum_full = 0.0;
            for (int i = s; i < e; i++)
                sum_full += data[i].energy[r];
            double mean_full = sum_full / n;

            double jk_sum2 = 0.0;
            for (int j = s; j < e; j++) {
                double jk_mean = (sum_full - data[j].energy[r]) / (n - 1);
                double diff = jk_mean - mean_full;
                jk_sum2 += diff * diff;
            }
            double jk_err = sqrt((double)(n - 1) / n * jk_sum2);

            fprintf(fout, "\t%.8f\t%.8f", mean_full, jk_err);
            printf("  %10.6f  %10.6f", mean_full, jk_err);
        }

        fprintf(fout, "\n");
        printf("\n");
    }

    fclose(fout);
    printf("  Block averages written to %s\n", outfile);
}

int main(int argc, char** argv) {
    int N = 0;
    int nrep = 1;
    int label = -1;
    double T_sim = 1.0;
    std::string datadir_override;
    bool do_plot = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-label") == 0 && i + 1 < argc)
            label = atoi(argv[++i]);
        else if (strcmp(argv[i], "-datadir") == 0 && i + 1 < argc)
            datadir_override = argv[++i];
        else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc)
            T_sim = atof(argv[++i]);
        else if (strcmp(argv[i], "--plot") == 0)
            do_plot = true;
        else usage(argv[0]);
    }
    if (N < 4) usage(argv[0]);

    // ================================================================
    // Single-sample mode: -label given or -datadir given
    // ================================================================
    if (label >= 0 || !datadir_override.empty()) {
        char datadir[256];
        if (!datadir_override.empty())
            snprintf(datadir, sizeof(datadir), "%s", datadir_override.c_str());
        else
            snprintf(datadir, sizeof(datadir), "data/N%d_NR%d_S%d", N, nrep, label);

        auto data = read_data(datadir, nrep);
        if (data.empty()) {
            fprintf(stderr, "No data found in %s\n", datadir);
            return 1;
        }
        printf("\n");
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║           p-Spin 2+4 :: Analysis (MC)            ║\n");
        printf("╚══════════════════════════════════════════════════╝\n");
        printf("  %-22s single\n", "mode");
        printf("  %-22s %s\n", "directory", datadir);
        printf("  %-22s %d\n", "data points", (int)data.size());
        printf("  %-22s %d\n\n", "replicas", nrep);
        single_sample_analysis(datadir, data, nrep);
        return 0;
    }

    // ================================================================
    // Multi-sample mode: average over all S0, S1, S2, ...
    // ================================================================
    auto labels = find_samples(N, nrep);
    if (labels.empty()) {
        fprintf(stderr, "No sample directories found for N%d_NR%d_S*\n", N, nrep);
        return 1;
    }

    int nsamples = (int)labels.size();
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║           p-Spin 2+4 :: Analysis (MC)            ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  %-22s %d\n", "N", N);
    printf("  %-22s %d\n", "nrep", nrep);
    printf("  %-22s %d\n\n", "samples", nsamples);

    // Per-sample: compute last-half mean E/N for each replica
    // E_sr[s][r] = mean energy of sample s, replica r (last half of time series)
    std::vector<std::vector<double>> E_sr(nsamples);

    for (int s = 0; s < nsamples; s++) {
        char sdir[256];
        snprintf(sdir, sizeof(sdir), "data/N%d_NR%d_S%d", N, nrep, labels[s]);
        auto data = read_data(sdir, nrep);
        if (data.empty()) {
            fprintf(stderr, "Warning: no data in %s, skipping\n", sdir);
            continue;
        }
        E_sr[s] = last_half_mean(data, nrep);

        printf("\n── Sample S%d ──────────────────────────────────────\n", labels[s]);
        single_sample_analysis(sdir, data, nrep);
        printf("\n");
    }

    // Remove samples that had no data
    std::vector<std::vector<double>> E_valid;
    std::vector<int> labels_valid;
    for (int s = 0; s < nsamples; s++) {
        if (!E_sr[s].empty()) {
            E_valid.push_back(E_sr[s]);
            labels_valid.push_back(labels[s]);
        }
    }
    int S = (int)E_valid.size();
    if (S == 0) {
        fprintf(stderr, "No valid samples\n");
        return 1;
    }

    // ================================================================
    // Sample average with jackknife errors
    // ================================================================
    // For each replica r:
    //   full mean = (1/S) Σ_s E_sr
    //   jackknife: leave out sample j, mean_j = (1/(S-1)) Σ_{s≠j} E_sr
    //   jk_err = sqrt((S-1)/S * Σ_j (mean_j - full_mean)^2)
    //
    // Also compute replica-averaged quantity:
    //   E_s = (1/R) Σ_r E_sr   (sample-level average over replicas)
    //   then jackknife over samples
    // ================================================================

    printf("\n── Sample Averages ────────────────────────────────\n");
    printf("  %-22s %d\n\n", "jackknife samples", S);

    // Per-replica sample averages
    printf("Per-replica averages:\n");
    printf("  %6s  %12s  %12s\n", "rep", "<E/N>", "jk_err");
    printf("  ------------------------------------\n");

    // Output file
    char sumfile[256];
    snprintf(sumfile, sizeof(sumfile), "data/N%d_NR%d_sample_avg.txt", N, nrep);
    FILE* fsum = fopen(sumfile, "w");
    if (fsum) {
        fprintf(fsum, "# Sample-averaged E/N with jackknife errors\n");
        fprintf(fsum, "# N=%d  nrep=%d  nsamples=%d\n", N, nrep, S);
        fprintf(fsum, "# Per-replica:\n");
        fprintf(fsum, "# replica  <E/N>  jk_err\n");
    }

    std::vector<double> rep_means(nrep), rep_errs(nrep);
    for (int r = 0; r < nrep; r++) {
        double sum_full = 0.0;
        for (int s = 0; s < S; s++)
            sum_full += E_valid[s][r];
        double mean_full = sum_full / S;

        double jk_sum2 = 0.0;
        for (int j = 0; j < S; j++) {
            double jk_mean = (sum_full - E_valid[j][r]) / (S - 1);
            double diff = jk_mean - mean_full;
            jk_sum2 += diff * diff;
        }
        double jk_err = (S > 1) ? sqrt((double)(S - 1) / S * jk_sum2) : 0.0;

        rep_means[r] = mean_full;
        rep_errs[r] = jk_err;
        printf("  %6d  %12.8f  %12.8f\n", r, mean_full, jk_err);
        if (fsum) fprintf(fsum, "%d\t%.8f\t%.8f\n", r, mean_full, jk_err);
    }

    // Replica-averaged, then sample-averaged
    // E_s = (1/R) Σ_r E_sr
    std::vector<double> E_s(S);
    for (int s = 0; s < S; s++) {
        double sum = 0.0;
        for (int r = 0; r < nrep; r++)
            sum += E_valid[s][r];
        E_s[s] = sum / nrep;
    }

    double sum_full = 0.0;
    for (int s = 0; s < S; s++)
        sum_full += E_s[s];
    double mean_all = sum_full / S;

    double jk_sum2 = 0.0;
    for (int j = 0; j < S; j++) {
        double jk_mean = (sum_full - E_s[j]) / (S - 1);
        double diff = jk_mean - mean_all;
        jk_sum2 += diff * diff;
    }
    double jk_err_all = (S > 1) ? sqrt((double)(S - 1) / S * jk_sum2) : 0.0;

    printf("\n── Result ──────────────────────────────────────────\n");
    printf("  %-22s %.8f +/- %.8f\n", "[E/N]", mean_all, jk_err_all);

    if (fsum) {
        fprintf(fsum, "# Replica + sample averaged:\n");
        fprintf(fsum, "# <E/N>  jk_err\n");
        fprintf(fsum, "%.8f\t%.8f\n", mean_all, jk_err_all);
        fclose(fsum);
        printf("\nSample averages written to %s\n", sumfile);
    }

    // ================================================================
    // Intensity Spectrum
    // ================================================================
    {
        // Read frequencies from first valid sample
        char refdir[256];
        snprintf(refdir, sizeof(refdir), "data/N%d_NR%d_S%d",
                 N, nrep, labels_valid[0]);
        auto omega = read_frequencies(refdir, N);

        // Per-sample mean spectrum
        // spec_s[s][k] = <I_k> averaged over (replica, iteration) configs
        std::vector<std::vector<double>> spec_s(S, std::vector<double>(N, 0.0));

        for (int s = 0; s < S; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/N%d_NR%d_S%d",
                     N, nrep, labels_valid[s]);
            char confdir[512];
            snprintf(confdir, sizeof(confdir), "%s/configs", sdir);

            int nconfigs = 0;
            for (int r = 0; r < nrep; r++) {
                auto cfiles = find_configs_mc(confdir, r);
                for (auto& cf : cfiles) {
                    std::vector<double> Ik;
                    if (read_config_intensities(cf.c_str(), N, Ik)) {
                        for (int k = 0; k < N; k++)
                            spec_s[s][k] += Ik[k];
                        nconfigs++;
                    }
                }
            }
            if (nconfigs > 0) {
                for (int k = 0; k < N; k++)
                    spec_s[s][k] /= nconfigs;
            }
            printf("  Sample S%d: spectrum computed (%d configs)\n",
                   labels_valid[s], nconfigs);
        }

        // Output directory
        char outdir[256];
        snprintf(outdir, sizeof(outdir), "analysis/MC_N%d_NR%d", N, nrep);
        mkdir_p(outdir);

        // Write intensity_spectrum.dat (single temperature block)
        char specfile[512];
        snprintf(specfile, sizeof(specfile), "%s/intensity_spectrum.dat", outdir);
        FILE* fsp = fopen(specfile, "w");
        if (fsp) {
            fprintf(fsp, "# Intensity spectrum: I_k = |a_k|^2 / sum_j |a_j|^2\n");
            fprintf(fsp, "# N=%d nrep=%d nsamples=%d T=%.8f\n", N, nrep, S, T_sim);
            fprintf(fsp, "# Columns: frequency  Intensity  Error_jk  Temperature\n");

            // Sort index by frequency
            std::vector<int> order(N);
            for (int k = 0; k < N; k++) order[k] = k;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return omega[a] < omega[b]; });

            fprintf(fsp, "\n");
            for (int ik = 0; ik < N; ik++) {
                int k = order[ik];
                double fI = 0;
                for (int s = 0; s < S; s++)
                    fI += spec_s[s][k];
                fI /= S;

                double jI = 0;
                for (int j = 0; j < S; j++) {
                    double lI = 0;
                    for (int s = 0; s < S; s++) {
                        if (s == j) continue;
                        lI += spec_s[s][k];
                    }
                    lI /= (S - 1);
                    jI += (lI - fI) * (lI - fI);
                }
                jI = sqrt((S - 1.0) / S * jI);

                fprintf(fsp, "%.12f\t%.8e\t%.8e\t%.8f\n",
                        omega[k], fI, jI, T_sim);
            }
            fclose(fsp);
            printf("  Written %s\n", specfile);
        }
    }

    // ================================================================
    // Plotting (if --plot)
    // ================================================================
    if (do_plot) {
        using namespace sciplot;
        printf("\n── Generating plots ────────────────────────────────\n");

        char outdir[256];
        snprintf(outdir, sizeof(outdir), "analysis/MC_N%d_NR%d", N, nrep);

        char plotdir[512];
        snprintf(plotdir, sizeof(plotdir), "%s/plots", outdir);
        mkdir_p(plotdir);

        char specfile[512];
        snprintf(specfile, sizeof(specfile), "%s/intensity_spectrum.dat", outdir);
        FILE* f = fopen(specfile, "r");
        if (f) {
            struct SpecLine { double omega, I, err, T; };
            std::vector<std::vector<SpecLine>> blocks;
            std::vector<SpecLine> cur;
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                if (line[0] == '#') continue;
                if (line[0] == '\n' || line[0] == '\r') {
                    if (!cur.empty()) { blocks.push_back(cur); cur.clear(); }
                    continue;
                }
                SpecLine sl;
                if (sscanf(line, "%lf %lf %lf %lf", &sl.omega, &sl.I, &sl.err, &sl.T) == 4)
                    cur.push_back(sl);
            }
            if (!cur.empty()) blocks.push_back(cur);
            fclose(f);

            if (!blocks.empty()) {
                Plot2D plot;
                plot.xlabel("Frequency {/Symbol w}");
                plot.ylabel("Emission intensity  I_k");
                plot.fontName("Helvetica");
                plot.fontSize(16);
                plot.legend().hide();
                plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 0.8 dt 3");
                plot.gnuplot("set border lw 1.5");
                plot.gnuplot("set tics font 'Helvetica,13'");
                int nb = (int)blocks.size();
                double Tmin = blocks.back()[0].T;
                double Tmax = blocks.front()[0].T;
                plot.gnuplot("set palette defined (0 '#1A33CC', 0.33 '#1AB580', 0.66 '#CC9919', 1.0 '#FF1A0D')");
                char cbr[128];
                snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin, Tmax);
                plot.gnuplot(cbr);
                plot.gnuplot("set cblabel 'T' font 'Helvetica,14'");
                plot.gnuplot("set colorbox");
                for (int bi = 0; bi < nb; bi++) {
                    auto& bl = blocks[bi];
                    int n = (int)bl.size();
                    std::vector<double> vx(n), vy(n);
                    for (int i = 0; i < n; i++) { vx[i] = bl[i].omega; vy[i] = bl[i].I; }
                    double frac = (nb > 1) ? 1.0 - (double)bi / (nb - 1) : 0.5;
                    plot.drawCurve(vx, vy)
                        .lineColor(temp_color(frac))
                        .lineWidth(2)
                        .label("");
                }
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(1800, 1200);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/intensity_spectrum.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }
    }

    return 0;
}
