#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <regex.h>

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

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>] [-label <l>] [-datadir <path>]\n", prog);
    fprintf(stderr, "  Without -label: averages over all samples S0, S1, ...\n");
    fprintf(stderr, "  With -label L:  analyzes single sample SL only\n");
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
    std::string datadir_override;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-label") == 0 && i + 1 < argc)
            label = atoi(argv[++i]);
        else if (strcmp(argv[i], "-datadir") == 0 && i + 1 < argc)
            datadir_override = argv[++i];
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
        printf("=== Single sample analysis: %s (%d points, %d replicas) ===\n\n", datadir, (int)data.size(), nrep);
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
    printf("========================================\n");
    printf("Multi-sample analysis: N=%d  nrep=%d\n", N, nrep);
    printf("Found %d samples: S%d", nsamples, labels[0]);
    for (int i = 1; i < nsamples; i++) printf(", S%d", labels[i]);
    printf("\n");
    printf("========================================\n\n");

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

        printf("--- Sample S%d (%s, %d points) ---\n", labels[s], sdir, (int)data.size());
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

    printf("========================================\n");
    printf("SAMPLE AVERAGES (jackknife, %d samples)\n", S);
    printf("========================================\n\n");

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

    printf("\nReplica + sample averaged:\n");
    printf("  [E/N] = %.8f +/- %.8f\n", mean_all, jk_err_all);

    if (fsum) {
        fprintf(fsum, "# Replica + sample averaged:\n");
        fprintf(fsum, "# <E/N>  jk_err\n");
        fprintf(fsum, "%.8f\t%.8f\n", mean_all, jk_err_all);
        fclose(fsum);
        printf("\nSample averages written to %s\n", sumfile);
    }

    return 0;
}
