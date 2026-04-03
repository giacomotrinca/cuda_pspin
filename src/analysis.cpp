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
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>] [-label <l>] [-datadir <path>] [-T <temperature>] [-nbins_spec <B>] [--plot]\n", prog);
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
    int nbins_spec = 0;
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
        else if (strcmp(argv[i], "-nbins_spec") == 0 && i + 1 < argc)
            nbins_spec = atoi(argv[++i]);
        else if (strcmp(argv[i], "--plot") == 0)
            do_plot = true;
        else usage(argv[0]);
    }
    if (N < 4) usage(argv[0]);
    if (nbins_spec <= 0) nbins_spec = N;

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
    // History block averaging (doubling blocks)
    // ================================================================
    // For each doubling block (sizes 1, 2, 4, ..., M/2), compute mean energy
    // and acceptance. Jackknife over samples, average over replicas.
    // Output: history_nr{r}.dat, history_mean.dat
    // Columns: sweep_block  energy  err  accept_mc  err
    {
        char outdir_h[256];
        snprintf(outdir_h, sizeof(outdir_h), "analysis/MC_N%d_NR%d", N, nrep);
        mkdir_p(outdir_h);

        // Re-read all sample data
        std::vector<std::vector<DataPoint>> all_data(S);
        int M_ref = 0;
        for (int s = 0; s < S; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/N%d_NR%d_S%d", N, nrep, labels_valid[s]);
            all_data[s] = read_data(sdir, nrep);
            if (s == 0) M_ref = (int)all_data[s].size();
        }

        // Count doubling blocks
        int nblocks = 0;
        { int sz = 1; int pos = 0; while (pos < M_ref) { pos += sz; sz *= 2; nblocks++; } }

        printf("\n── History block averaging ─────────────────────────\n");
        printf("  data points (ref): %d   doubling blocks: %d\n", M_ref, nblocks);

        // Per-replica history files
        for (int r = 0; r < nrep; r++) {
            char outfile[512];
            snprintf(outfile, sizeof(outfile), "%s/history_nr%d.dat", outdir_h, r);
            FILE* fout = fopen(outfile, "w");
            if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); continue; }

            fprintf(fout, "# History block averaging (doubling blocks)\n");
            fprintf(fout, "# N=%d nrep=%d replica=%d nsamples=%d T=%.8f\n", N, nrep, r, S, T_sim);
            fprintf(fout, "# Columns: sweep_block  energy  err  accept_mc  err\n");

            int bsize = 1, pos = 0;
            for (int b = 0; b < nblocks && pos < M_ref; b++) {
                int bend = pos + bsize;
                if (bend > M_ref) bend = M_ref;
                int sweep_end = all_data[0][bend - 1].iter;

                std::vector<double> sE(S, 0), sA(S, 0);
                for (int s = 0; s < S; s++) {
                    int Ms = (int)all_data[s].size();
                    int p = pos, be = bend;
                    if (be > Ms) be = Ms;
                    if (p >= Ms) continue;
                    int nn = be - p;
                    double se = 0, sa = 0;
                    for (int i = p; i < be; i++) {
                        se += all_data[s][i].energy[r];
                        sa += all_data[s][i].acc[r];
                    }
                    sE[s] = se / nn;
                    sA[s] = sa / nn;
                }

                double fE = 0, fA = 0;
                for (int s = 0; s < S; s++) { fE += sE[s]; fA += sA[s]; }
                fE /= S; fA /= S;

                double jE = 0, jA = 0;
                for (int j = 0; j < S; j++) {
                    double lE = 0, lA = 0;
                    for (int s = 0; s < S; s++) {
                        if (s == j) continue;
                        lE += sE[s]; lA += sA[s];
                    }
                    lE /= (S - 1); lA /= (S - 1);
                    jE += (lE - fE) * (lE - fE);
                    jA += (lA - fA) * (lA - fA);
                }
                jE = sqrt((S - 1.0) / S * jE);
                jA = sqrt((S - 1.0) / S * jA);

                fprintf(fout, "%d\t%.8f\t%.8f\t%.5f\t%.5f\n", sweep_end, fE, jE, fA, jA);

                pos = bend;
                bsize *= 2;
            }
            fclose(fout);
            printf("  Written %s\n", outfile);
        }

        // Replica-averaged history file
        {
            char outfile[512];
            snprintf(outfile, sizeof(outfile), "%s/history_mean.dat", outdir_h);
            FILE* fout = fopen(outfile, "w");
            if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); }
            else {
                fprintf(fout, "# History block averaging (doubling blocks) — replica averaged\n");
                fprintf(fout, "# N=%d nrep=%d nsamples=%d T=%.8f\n", N, nrep, S, T_sim);
                fprintf(fout, "# Columns: sweep_block  energy  err  accept_mc  err\n");

                int bsize = 1, pos = 0;
                int b = 0;
                while (pos < M_ref) {
                    int bend = pos + bsize;
                    if (bend > M_ref) bend = M_ref;
                    int sweep_end = all_data[0][bend - 1].iter;

                    std::vector<double> smE(S, 0), smA(S, 0);
                    for (int s = 0; s < S; s++) {
                        int Ms = (int)all_data[s].size();
                        int p = pos, be = bend;
                        if (be > Ms) be = Ms;
                        if (p >= Ms) continue;
                        int nn = be - p;
                        for (int r = 0; r < nrep; r++) {
                            double se = 0, sa = 0;
                            for (int i = p; i < be; i++) {
                                se += all_data[s][i].energy[r];
                                sa += all_data[s][i].acc[r];
                            }
                            smE[s] += se / nn;
                            smA[s] += sa / nn;
                        }
                        smE[s] /= nrep;
                        smA[s] /= nrep;
                    }

                    double fE = 0, fA = 0;
                    for (int s = 0; s < S; s++) { fE += smE[s]; fA += smA[s]; }
                    fE /= S; fA /= S;

                    double jE = 0, jA = 0;
                    for (int j = 0; j < S; j++) {
                        double lE = 0, lA = 0;
                        for (int s = 0; s < S; s++) {
                            if (s == j) continue;
                            lE += smE[s]; lA += smA[s];
                        }
                        lE /= (S - 1); lA /= (S - 1);
                        jE += (lE - fE) * (lE - fE);
                        jA += (lA - fA) * (lA - fA);
                    }
                    jE = sqrt((S - 1.0) / S * jE);
                    jA = sqrt((S - 1.0) / S * jA);

                    fprintf(fout, "%d\t%.8f\t%.8f\t%.5f\t%.5f\n", sweep_end, fE, jE, fA, jA);

                    pos = bend;
                    bsize *= 2;
                    b++;
                }
                fclose(fout);
                printf("  Written %s\n", outfile);
            }
        }
    }

    // ================================================================
    // Intensity Spectrum (rebinned)
    // ================================================================
    {
        printf("\n── Intensity spectrum (nbins_spec=%d) ──────────────\n", nbins_spec);

        // Read per-sample frequencies
        std::vector<std::vector<double>> omega_s(S);
        for (int s = 0; s < S; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/N%d_NR%d_S%d",
                     N, nrep, labels_valid[s]);
            omega_s[s] = read_frequencies(sdir, N);
        }

        // Rebinned spectrum: spec_bin_s[s][b]
        double dw = 1.0 / nbins_spec;
        std::vector<std::vector<double>> spec_bin_s(S, std::vector<double>(nbins_spec, 0.0));

        for (int s = 0; s < S; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/N%d_NR%d_S%d",
                     N, nrep, labels_valid[s]);
            char confdir[512];
            snprintf(confdir, sizeof(confdir), "%s/configs", sdir);

            int nconfigs = 0;
            std::vector<double> bin_acc(nbins_spec, 0.0);
            for (int r = 0; r < nrep; r++) {
                auto cfiles = find_configs_mc(confdir, r);
                for (auto& cf : cfiles) {
                    std::vector<double> Ik;
                    if (read_config_intensities(cf.c_str(), N, Ik)) {
                        for (int k = 0; k < N; k++) {
                            int b = (int)(omega_s[s][k] / dw);
                            if (b < 0) b = 0;
                            if (b >= nbins_spec) b = nbins_spec - 1;
                            bin_acc[b] += Ik[k];
                        }
                        nconfigs++;
                    }
                }
            }
            if (nconfigs > 0) {
                for (int b = 0; b < nbins_spec; b++)
                    spec_bin_s[s][b] = bin_acc[b] / nconfigs;
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
            fprintf(fsp, "# Rebinned intensity spectrum: I(omega) on uniform grid [0,1]\n");
            fprintf(fsp, "# N=%d nrep=%d nsamples=%d T=%.8f nbins_spec=%d\n",
                    N, nrep, S, T_sim, nbins_spec);
            fprintf(fsp, "# Columns: omega_center  Intensity  Error_jk  Temperature\n");

            fprintf(fsp, "\n");
            for (int b = 0; b < nbins_spec; b++) {
                double wc = (b + 0.5) * dw;

                double fI = 0;
                for (int s = 0; s < S; s++)
                    fI += spec_bin_s[s][b];
                fI /= S;

                double jI = 0;
                for (int j = 0; j < S; j++) {
                    double lI = 0;
                    for (int s = 0; s < S; s++) {
                        if (s == j) continue;
                        lI += spec_bin_s[s][b];
                    }
                    lI /= (S - 1);
                    jI += (lI - fI) * (lI - fI);
                }
                jI = sqrt((S - 1.0) / S * jI);

                fprintf(fsp, "%.12f\t%.8e\t%.8e\t%.8f\n",
                        wc, fI, jI, T_sim);
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

        char plotdir[384];
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
                plot.xlabel("{/Times-Italic {/Symbol w}}");
                plot.ylabel("{/Times-Italic I}_{/Times-Italic k}");
                plot.fontName("Times");
                plot.fontSize(18);
                plot.legend().hide();
                int nb = (int)blocks.size();
                double Tmin = blocks.back()[0].T;
                double Tmax = blocks.front()[0].T;
                plot.gnuplot("set palette defined (0 '#1A33CC', 0.33 '#1AB580', 0.66 '#CC9919', 1.0 '#FF1A0D')");
                char cbr[128];
                snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin, Tmax);
                plot.gnuplot(cbr);
                plot.gnuplot("set cblabel '{/Times-Italic T}' font 'Times,16'");
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

        // --- History plots (from history_mean.dat) ---
        {
            char histfile[512];
            snprintf(histfile, sizeof(histfile), "%s/history_mean.dat", outdir);
            FILE* fh = fopen(histfile, "r");
            if (fh) {
                struct HistLine { int sweep; double E, Eerr, A, Aerr; };
                std::vector<HistLine> hdata;
                char line[512];
                while (fgets(line, sizeof(line), fh)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    HistLine hl;
                    if (sscanf(line, "%d %lf %lf %lf %lf",
                               &hl.sweep, &hl.E, &hl.Eerr, &hl.A, &hl.Aerr) == 5)
                        hdata.push_back(hl);
                }
                fclose(fh);

                if (!hdata.empty()) {
                    int nh = (int)hdata.size();

                    auto make_history_plot = [&](const char* ylabel_str,
                                                 int val_col,
                                                 const char* outname) {
                        Plot2D plot;
                        plot.xlabel("sweep");
                        plot.ylabel(ylabel_str);
                        plot.fontName("Times");
                        plot.fontSize(18);
                        plot.legend().hide();
                        plot.gnuplot("set logscale x 2");
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");

                        std::vector<double> vx(nh), vy(nh), vlo(nh), vhi(nh);
                        for (int i = 0; i < nh; i++) {
                            vx[i] = hdata[i].sweep;
                            double v = (val_col == 0) ? hdata[i].E : hdata[i].A;
                            double e = (val_col == 0) ? hdata[i].Eerr : hdata[i].Aerr;
                            vy[i] = v;
                            vlo[i] = v - e;
                            vhi[i] = v + e;
                        }
                        plot.drawCurvesFilled(vx, vlo, vhi)
                            .fillColor("#4393c3").fillIntensity(0.35).fillTransparent()
                            .lineColor("#4393c3").lineWidth(0).labelNone();
                        plot.drawCurve(vx, vy)
                            .lineColor("#2166ac").lineWidth(2.5).label("");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1800, 1200);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/%s", plotdir, outname);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    };

                    make_history_plot("{/Times-Italic E} / {/Times-Italic N}", 0, "energy_history.png");
                    make_history_plot("MC acceptance", 1, "acceptance_history.png");
                }
            }
        }
    }

    return 0;
}
