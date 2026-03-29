// Analysis of parallel tempering data.
//
// Reads energy_accept.txt and exchanges.txt from data/PT_N{N}_NT{NT}_NR{nrep}_S{0,1,...}.
// For each temperature, uses the second half of the time series to compute
// mean energy, acceptance, and specific heat (jackknife errors over samples).
//
// Output directory: analysis/PT_N{N}_NT{NT}_NR{nrep}/
//   equilibrium_data_nr{r}.dat  — per-replica results
//   equilibrium_data_mean.dat   — replica-averaged results
//   exchange_rates.dat          — exchange acceptance rates between adjacent temperatures
//
// Columns (equilibrium_data):
//   Temperature  Energy_mean  Energy_err_jk  Acceptance_mean  Acceptance_err_jk  Cv  Cv_err_jk
//
// Specific heat: Cv = N * (<e^2> - <e>^2) / T^2,  e = E/N.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sys/stat.h>
#include <dirent.h>
#include <sciplot/sciplot.hpp>

// Color interpolation for temperature gradient (blue=cold, red=hot)
static std::string temp_color(double frac) {
    // Smooth blue -> teal -> gold -> red
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

// Read a binary config and return raw complex amplitudes (re, im) pairs
static bool read_config_spins(const char* filename, int N,
                              std::vector<double>& re, std::vector<double>& im) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    std::vector<double> buf(2 * N);
    size_t nr = fread(buf.data(), sizeof(double), 2 * N, f);
    fclose(f);
    if ((int)nr != 2 * N) return false;
    re.resize(N);
    im.resize(N);
    for (int k = 0; k < N; k++) {
        re[k] = buf[2*k];
        im[k] = buf[2*k + 1];
    }
    return true;
}

// Find all iteration numbers for a given (tidx, rep) in the config directory
static std::vector<int> find_config_iters(const char* confdir, int tidx, int rep) {
    std::vector<int> iters;
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "conf_T%d_r%d_iter", tidx, rep);
    int plen = (int)strlen(prefix);
    DIR* dir = opendir(confdir);
    if (!dir) return iters;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, prefix, plen) == 0) {
            int it = atoi(ent->d_name + plen);
            if (it > 0) iters.push_back(it);
        }
    }
    closedir(dir);
    std::sort(iters.begin(), iters.end());
    return iters;
}

// Find config files matching conf_T{tidx}_r{rep}_iter*.bin in confdir
static std::vector<std::string> find_configs_pt(const char* confdir,
                                                int tidx, int rep) {
    std::vector<std::string> files;
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "conf_T%d_r%d_iter", tidx, rep);
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
struct Row {
    int sweep;
    int tidx;
    double T;
    std::vector<double> energy;
    std::vector<double> acc;
};

// One exchange measurement row
struct ExRow {
    int sweep;
    int tidx;
    double T_high, T_low;
    long long n_acc, n_prop;
    double rate;
};

// Per-temperature data for one sample
struct TempBlock {
    double T;
    int tidx;
    std::vector<int> sweeps;
    std::vector<std::vector<double>> energy; // [measurement][replica]
    std::vector<std::vector<double>> acc;
};

// Per-temperature-pair exchange data for one sample (cumulative)
struct ExBlock {
    int tidx;
    double T_high, T_low;
    long long total_acc, total_prop;
};

// Per-temperature-pair exchange data, per-sweep (for history block averaging)
struct ExSweepBlock {
    int tidx;
    double T_high, T_low;
    std::vector<int> sweeps;
    std::vector<long long> n_acc;   // per sweep
    std::vector<long long> n_prop;  // per sweep
};

// Read energy_accept.txt, return grouped by temperature index
static std::vector<TempBlock> read_pt_data(const char* datadir, int nrep) {
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/energy_accept.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) return {};

    std::vector<Row> rows;
    char line[4096];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        Row row;
        row.energy.resize(nrep);
        row.acc.resize(nrep);

        char* tok = strtok(line, " \t\n");
        if (!tok) continue;
        row.sweep = atoi(tok);

        tok = strtok(nullptr, " \t\n");
        if (!tok) continue;
        row.tidx = atoi(tok);

        tok = strtok(nullptr, " \t\n");
        if (!tok) continue;
        row.T = atof(tok);

        bool ok = true;
        for (int r = 0; r < nrep; r++) {
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            row.energy[r] = atof(tok);
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            row.acc[r] = atof(tok);
        }
        if (!ok) continue;
        rows.push_back(row);
    }
    fclose(fin);

    // Group by temperature index
    std::map<int, int> tidx_to_block;
    std::vector<TempBlock> blocks;

    for (auto& row : rows) {
        auto it = tidx_to_block.find(row.tidx);
        int idx;
        if (it == tidx_to_block.end()) {
            idx = (int)blocks.size();
            tidx_to_block[row.tidx] = idx;
            TempBlock tb;
            tb.T = row.T;
            tb.tidx = row.tidx;
            blocks.push_back(tb);
        } else {
            idx = it->second;
        }
        blocks[idx].sweeps.push_back(row.sweep);
        blocks[idx].energy.push_back(row.energy);
        blocks[idx].acc.push_back(row.acc);
    }

    // Sort by temperature index
    std::sort(blocks.begin(), blocks.end(),
              [](const TempBlock& a, const TempBlock& b) { return a.tidx < b.tidx; });

    return blocks;
}

// Read exchanges.txt, return cumulative exchange stats per temperature pair
static std::vector<ExBlock> read_exchange_data(const char* datadir) {
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/exchanges.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) return {};

    std::map<int, ExBlock> exmap;
    char line[1024];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int sweep, tidx;
        double T_high, T_low, rate;
        long long n_acc, n_prop;
        if (sscanf(line, "%d %d %lf %lf %lld %lld %lf",
                   &sweep, &tidx, &T_high, &T_low, &n_acc, &n_prop, &rate) < 7)
            continue;

        auto it = exmap.find(tidx);
        if (it == exmap.end()) {
            ExBlock eb;
            eb.tidx = tidx;
            eb.T_high = T_high;
            eb.T_low = T_low;
            eb.total_acc = n_acc;
            eb.total_prop = n_prop;
            exmap[tidx] = eb;
        } else {
            it->second.total_acc += n_acc;
            it->second.total_prop += n_prop;
        }
    }
    fclose(fin);

    std::vector<ExBlock> exvec;
    for (auto& kv : exmap) exvec.push_back(kv.second);
    std::sort(exvec.begin(), exvec.end(),
              [](const ExBlock& a, const ExBlock& b) { return a.tidx < b.tidx; });
    return exvec;
}

// Read exchanges.txt, return per-sweep exchange data per temperature pair
static std::vector<ExSweepBlock> read_exchange_sweep_data(const char* datadir) {
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/exchanges.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) return {};

    std::map<int, int> tidx_to_block;
    std::vector<ExSweepBlock> blocks;
    char line[1024];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int sweep, tidx;
        double T_high, T_low, rate;
        long long n_acc, n_prop;
        if (sscanf(line, "%d %d %lf %lf %lld %lld %lf",
                   &sweep, &tidx, &T_high, &T_low, &n_acc, &n_prop, &rate) < 7)
            continue;

        auto it = tidx_to_block.find(tidx);
        int idx;
        if (it == tidx_to_block.end()) {
            idx = (int)blocks.size();
            tidx_to_block[tidx] = idx;
            ExSweepBlock eb;
            eb.tidx = tidx;
            eb.T_high = T_high;
            eb.T_low = T_low;
            blocks.push_back(eb);
        } else {
            idx = it->second;
        }
        blocks[idx].sweeps.push_back(sweep);
        blocks[idx].n_acc.push_back(n_acc);
        blocks[idx].n_prop.push_back(n_prop);
    }
    fclose(fin);

    std::sort(blocks.begin(), blocks.end(),
              [](const ExSweepBlock& a, const ExSweepBlock& b) { return a.tidx < b.tidx; });
    return blocks;
}

// Find all sample directories matching data/PT_N{N}_NT{NT}_NR{nrep}_S*
static std::vector<int> find_samples(int N, int NT, int nrep) {
    std::vector<int> labels;
    DIR* dir = opendir("data");
    if (!dir) return labels;

    char pat[128];
    snprintf(pat, sizeof(pat), "PT_N%d_NT%d_NR%d_S", N, NT, nrep);
    int plen = (int)strlen(pat);

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, pat, plen) == 0) {
            int label = atoi(ent->d_name + plen);
            char check[128];
            snprintf(check, sizeof(check), "PT_N%d_NT%d_NR%d_S%d", N, NT, nrep, label);
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

// Per-sample, per-temperature observables from second half of time series
struct SampleObs {
    double e_mean;
    double e2_mean;
    double a_mean;
};

static SampleObs compute_obs(const TempBlock& tb, int replica) {
    int M = (int)tb.energy.size();
    int start = M / 2;
    if (start >= M) start = 0;
    int n = M - start;

    double se = 0, se2 = 0, sa = 0;
    for (int i = start; i < M; i++) {
        double e = tb.energy[i][replica];
        se += e;
        se2 += e * e;
        sa += tb.acc[i][replica];
    }
    return { se / n, se2 / n, sa / n };
}

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s -N <N> -NT <NT> [-nrep <nrep>] [-nbins <nbins>] [-nthreads <t>] [--plot]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0, NT = 0, nrep = 1, nbins = 100, nthreads = 0;
    bool do_plot = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-NT") == 0 && i + 1 < argc)
            NT = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nbins") == 0 && i + 1 < argc)
            nbins = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nthreads") == 0 && i + 1 < argc)
            nthreads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--plot") == 0)
            do_plot = true;
        else usage(argv[0]);
    }
    if (nthreads <= 0)
        nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (N < 4 || NT < 2) usage(argv[0]);

    auto labels = find_samples(N, NT, nrep);
    if (labels.empty()) {
        fprintf(stderr, "No sample directories found for data/PT_N%d_NT%d_NR%d_S*\n", N, NT, nrep);
        return 1;
    }
    int nsamples = (int)labels.size();

    char outdir[512];
    snprintf(outdir, sizeof(outdir), "analysis/PT_N%d_NT%d_NR%d", N, NT, nrep);
    mkdir_p(outdir);

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║           p-Spin 2+4 :: Analysis (PT)            ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  %-22s %d\n", "N", N);
    printf("  %-22s %d\n", "NT", NT);
    printf("  %-22s %d\n", "nrep", nrep);
    printf("  %-22s %d\n", "samples", nsamples);
    printf("  %-22s %s/\n\n", "output", outdir);

    // Read all sample data
    std::vector<std::vector<TempBlock>> all_data(nsamples);
    std::vector<std::vector<ExBlock>> all_ex(nsamples);
    std::vector<std::vector<ExSweepBlock>> all_ex_sweep(nsamples);
    for (int s = 0; s < nsamples; s++) {
        char sdir[512];
        snprintf(sdir, sizeof(sdir), "data/PT_N%d_NT%d_NR%d_S%d", N, NT, nrep, labels[s]);
        all_data[s] = read_pt_data(sdir, nrep);
        all_ex[s] = read_exchange_data(sdir);
        all_ex_sweep[s] = read_exchange_sweep_data(sdir);
        if (all_data[s].empty())
            fprintf(stderr, "Warning: no data in %s\n", sdir);
    }

    // Use first non-empty sample as reference
    int ref = -1;
    for (int s = 0; s < nsamples; s++) {
        if (!all_data[s].empty()) { ref = s; break; }
    }
    if (ref < 0) {
        fprintf(stderr, "No valid data found\n");
        return 1;
    }
    int ntemps = (int)all_data[ref].size();

    // ================================================================
    // Per-replica output files
    // ================================================================
    for (int r = 0; r < nrep; r++) {
        char outfile[512];
        snprintf(outfile, sizeof(outfile), "%s/equilibrium_data_nr%d.dat", outdir, r);
        FILE* fout = fopen(outfile, "w");
        if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); continue; }

        fprintf(fout, "# Temperature\tEnergy_mean\tEnergy_err_jk\tAcceptance_mean\tAcceptance_err_jk\tCv\tCv_err_jk\n");
        fprintf(fout, "# N=%d NT=%d nrep=%d replica=%d nsamples=%d\n", N, NT, nrep, r, nsamples);
        fprintf(fout, "# Energy = E/N. Cv = N*(<e^2>-<e>^2)/T^2. Jackknife over %d samples.\n", nsamples);

        for (int ti = 0; ti < ntemps; ti++) {
            double T = all_data[ref][ti].T;

            std::vector<double> sE(nsamples), sC(nsamples), sA(nsamples);
            for (int s = 0; s < nsamples; s++) {
                if (ti >= (int)all_data[s].size()) continue;
                SampleObs obs = compute_obs(all_data[s][ti], r);
                sE[s] = obs.e_mean;
                sA[s] = obs.a_mean;
                sC[s] = N * (obs.e2_mean - obs.e_mean * obs.e_mean) / (T * T);
            }

            double fE = 0, fA = 0, fC = 0;
            for (int s = 0; s < nsamples; s++) { fE += sE[s]; fA += sA[s]; fC += sC[s]; }
            fE /= nsamples; fA /= nsamples; fC /= nsamples;

            double jE = 0, jA = 0, jC = 0;
            for (int j = 0; j < nsamples; j++) {
                double lE = 0, lA = 0, lC = 0;
                for (int s = 0; s < nsamples; s++) {
                    if (s == j) continue;
                    lE += sE[s]; lA += sA[s]; lC += sC[s];
                }
                lE /= (nsamples - 1); lA /= (nsamples - 1); lC /= (nsamples - 1);
                jE += (lE - fE) * (lE - fE);
                jA += (lA - fA) * (lA - fA);
                jC += (lC - fC) * (lC - fC);
            }
            jE = sqrt((nsamples - 1.0) / nsamples * jE);
            jA = sqrt((nsamples - 1.0) / nsamples * jA);
            jC = sqrt((nsamples - 1.0) / nsamples * jC);

            fprintf(fout, "%.8f\t%.8f\t%.8f\t%.5f\t%.5f\t%.8f\t%.8f\n",
                    T, fE, jE, fA, jA, fC, jC);
        }
        fclose(fout);
        printf("  Written %s\n", outfile);
    }

    // ================================================================
    // Replica-averaged output file
    // ================================================================
    {
        char outfile[512];
        snprintf(outfile, sizeof(outfile), "%s/equilibrium_data_mean.dat", outdir);
        FILE* fout = fopen(outfile, "w");
        if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }

        fprintf(fout, "# Temperature\tEnergy_mean\tEnergy_err_jk\tAcceptance_mean\tAcceptance_err_jk\tCv\tCv_err_jk\n");
        fprintf(fout, "# N=%d NT=%d nrep=%d nsamples=%d\n", N, NT, nrep, nsamples);
        fprintf(fout, "# Replica-averaged, then jackknife over %d samples. Cv = N*(<e^2>-<e>^2)/T^2.\n", nsamples);

        for (int ti = 0; ti < ntemps; ti++) {
            double T = all_data[ref][ti].T;

            std::vector<double> smE(nsamples, 0), smA(nsamples, 0), smC(nsamples, 0);
            for (int s = 0; s < nsamples; s++) {
                if (ti >= (int)all_data[s].size()) continue;
                for (int r = 0; r < nrep; r++) {
                    SampleObs obs = compute_obs(all_data[s][ti], r);
                    smE[s] += obs.e_mean;
                    smA[s] += obs.a_mean;
                    smC[s] += N * (obs.e2_mean - obs.e_mean * obs.e_mean) / (T * T);
                }
                smE[s] /= nrep;
                smA[s] /= nrep;
                smC[s] /= nrep;
            }

            double fE = 0, fA = 0, fC = 0;
            for (int s = 0; s < nsamples; s++) { fE += smE[s]; fA += smA[s]; fC += smC[s]; }
            fE /= nsamples; fA /= nsamples; fC /= nsamples;

            double jE = 0, jA = 0, jC = 0;
            for (int j = 0; j < nsamples; j++) {
                double lE = 0, lA = 0, lC = 0;
                for (int s = 0; s < nsamples; s++) {
                    if (s == j) continue;
                    lE += smE[s]; lA += smA[s]; lC += smC[s];
                }
                lE /= (nsamples - 1); lA /= (nsamples - 1); lC /= (nsamples - 1);
                jE += (lE - fE) * (lE - fE);
                jA += (lA - fA) * (lA - fA);
                jC += (lC - fC) * (lC - fC);
            }
            jE = sqrt((nsamples - 1.0) / nsamples * jE);
            jA = sqrt((nsamples - 1.0) / nsamples * jA);
            jC = sqrt((nsamples - 1.0) / nsamples * jC);

            fprintf(fout, "%.8f\t%.8f\t%.8f\t%.5f\t%.5f\t%.8f\t%.8f\n",
                    T, fE, jE, fA, jA, fC, jC);
        }
        fclose(fout);
        printf("  Written %s\n", outfile);
    }

    // ================================================================
    // Exchange rates (averaged over samples)
    // ================================================================
    {
        char outfile[512];
        snprintf(outfile, sizeof(outfile), "%s/exchange_rates.dat", outdir);
        FILE* fout = fopen(outfile, "w");
        if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }

        fprintf(fout, "# Tidx\tT_high\tT_low\trate_mean\trate_err_jk\n");
        fprintf(fout, "# N=%d NT=%d nrep=%d nsamples=%d\n", N, NT, nrep, nsamples);

        // Determine number of exchange pairs from reference
        int nex = 0;
        for (int s = 0; s < nsamples; s++) {
            if ((int)all_ex[s].size() > nex)
                nex = (int)all_ex[s].size();
        }

        for (int ei = 0; ei < nex; ei++) {
            // Compute per-sample exchange rate
            std::vector<double> sR(nsamples, 0.0);
            double T_high = 0, T_low = 0;
            int tidx = ei;
            int n_valid = 0;

            for (int s = 0; s < nsamples; s++) {
                if (ei >= (int)all_ex[s].size()) continue;
                auto& ex = all_ex[s][ei];
                T_high = ex.T_high;
                T_low = ex.T_low;
                tidx = ex.tidx;
                sR[s] = (ex.total_prop > 0)
                    ? (double)ex.total_acc / ex.total_prop : 0.0;
                n_valid++;
            }
            if (n_valid == 0) continue;

            double fR = 0;
            for (int s = 0; s < nsamples; s++) fR += sR[s];
            fR /= nsamples;

            double jR = 0;
            for (int j = 0; j < nsamples; j++) {
                double lR = 0;
                for (int s = 0; s < nsamples; s++) {
                    if (s == j) continue;
                    lR += sR[s];
                }
                lR /= (nsamples - 1);
                jR += (lR - fR) * (lR - fR);
            }
            jR = sqrt((nsamples - 1.0) / nsamples * jR);

            fprintf(fout, "%d\t%.8f\t%.8f\t%.6f\t%.6f\n",
                    tidx, T_high, T_low, fR, jR);

            printf("  Exchange T[%d]-T[%d] (%.4f-%.4f): rate=%.4f +/- %.4f\n",
                   tidx, tidx + 1, T_high, T_low, fR, jR);
        }
        fclose(fout);
        printf("  Written %s\n", outfile);
    }

    // ================================================================
    // History block averaging
    // ================================================================
    // For each temperature, tile the time series into doubling blocks:
    //   block 0: measurements [0, 1)            size 1
    //   block 1: measurements [1, 3)            size 2
    //   block 2: measurements [3, 7)            size 4
    //   ...
    //   block K: measurements [M/2, M)          size M/2  (last block = second half)
    //
    // For each block compute mean energy, error, mean acceptance MC, error,
    // mean acceptance PT (exchange rate), error — jackknife over samples.
    //
    // Output:
    //   history_nr{r}.dat   — per-replica
    //   history_mean.dat    — replica-averaged
    // Columns: sweep_block  energy  err  accept_mc  err  accept_pt  err  temperature
    // NT blocks separated by blank lines.
    {
        // Determine number of doubling blocks from reference sample
        int M_ref = (int)all_data[ref][0].energy.size();
        int nblocks = 0;
        { int sz = 1; int pos = 0; while (pos < M_ref) { pos += sz; sz *= 2; nblocks++; } }

        // Per-replica history files
        for (int r = 0; r < nrep; r++) {
            char outfile[512];
            snprintf(outfile, sizeof(outfile), "%s/history_nr%d.dat", outdir, r);
            FILE* fout = fopen(outfile, "w");
            if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); continue; }

            fprintf(fout, "# History block averaging (doubling blocks)\n");
            fprintf(fout, "# N=%d NT=%d nrep=%d replica=%d nsamples=%d\n", N, NT, nrep, r, nsamples);
            fprintf(fout, "# Columns: sweep_block  energy  err  accept_mc  err  accept_pt  err  temperature\n");
            fprintf(fout, "# %d temperature blocks separated by blank lines\n", ntemps);

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                if (ti > 0) fprintf(fout, "\n");

                int M = (int)all_data[ref][ti].energy.size();
                int bsize = 1, pos = 0;
                for (int b = 0; b < nblocks && pos < M; b++) {
                    int bend = pos + bsize;
                    if (bend > M) bend = M;
                    int sweep_end = all_data[ref][ti].sweeps[bend - 1];

                    // Per-sample observables for this block
                    std::vector<double> sE(nsamples, 0), sA(nsamples, 0), sPT(nsamples, 0);
                    for (int s = 0; s < nsamples; s++) {
                        if (ti >= (int)all_data[s].size()) continue;
                        auto& tb = all_data[s][ti];
                        int Ms = (int)tb.energy.size();
                        int p = pos, be = bend;
                        if (be > Ms) be = Ms;
                        if (p >= Ms) continue;
                        int nn = be - p;
                        double se = 0, sa = 0;
                        for (int i = p; i < be; i++) {
                            se += tb.energy[i][r];
                            sa += tb.acc[i][r];
                        }
                        sE[s] = se / nn;
                        sA[s] = sa / nn;

                        // Exchange rate for this block (use temperature pair tidx = ti)
                        if (ti < (int)all_ex_sweep[s].size()) {
                            auto& esb = all_ex_sweep[s][ti];
                            int Mex = (int)esb.sweeps.size();
                            if (p < Mex) {
                                int be_ex = (be < Mex) ? be : Mex;
                                long long acc_sum = 0, prop_sum = 0;
                                for (int i = p; i < be_ex; i++) {
                                    acc_sum += esb.n_acc[i];
                                    prop_sum += esb.n_prop[i];
                                }
                                sPT[s] = (prop_sum > 0) ? (double)acc_sum / prop_sum : 0.0;
                            }
                        }
                    }

                    // Jackknife
                    double fE = 0, fA = 0, fPT = 0;
                    for (int s = 0; s < nsamples; s++) { fE += sE[s]; fA += sA[s]; fPT += sPT[s]; }
                    fE /= nsamples; fA /= nsamples; fPT /= nsamples;

                    double jE = 0, jA = 0, jPT = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lE = 0, lA = 0, lPT = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lE += sE[s]; lA += sA[s]; lPT += sPT[s];
                        }
                        lE /= (nsamples - 1); lA /= (nsamples - 1); lPT /= (nsamples - 1);
                        jE += (lE - fE) * (lE - fE);
                        jA += (lA - fA) * (lA - fA);
                        jPT += (lPT - fPT) * (lPT - fPT);
                    }
                    jE = sqrt((nsamples - 1.0) / nsamples * jE);
                    jA = sqrt((nsamples - 1.0) / nsamples * jA);
                    jPT = sqrt((nsamples - 1.0) / nsamples * jPT);

                    fprintf(fout, "%d\t%.8f\t%.8f\t%.5f\t%.5f\t%.6f\t%.6f\t%.8f\n",
                            sweep_end, fE, jE, fA, jA, fPT, jPT, T);

                    pos = bend;
                    bsize *= 2;
                }
            }
            fclose(fout);
            printf("  Written %s\n", outfile);
        }

        // Replica-averaged history file
        {
            char outfile[512];
            snprintf(outfile, sizeof(outfile), "%s/history_mean.dat", outdir);
            FILE* fout = fopen(outfile, "w");
            if (!fout) { fprintf(stderr, "Cannot open %s\n", outfile); }
            else {
                fprintf(fout, "# History block averaging (doubling blocks) — replica averaged\n");
                fprintf(fout, "# N=%d NT=%d nrep=%d nsamples=%d\n", N, NT, nrep, nsamples);
                fprintf(fout, "# Columns: sweep_block  energy  err  accept_mc  err  accept_pt  err  temperature\n");
                fprintf(fout, "# %d temperature blocks separated by blank lines\n", ntemps);

                for (int ti = 0; ti < ntemps; ti++) {
                    double T = all_data[ref][ti].T;
                    if (ti > 0) fprintf(fout, "\n");

                    int M = (int)all_data[ref][ti].energy.size();
                    int bsize = 1, pos = 0;
                    int b = 0;
                    while (pos < M) {
                        int bend = pos + bsize;
                        if (bend > M) bend = M;
                        int sweep_end = all_data[ref][ti].sweeps[bend - 1];

                        // Per-sample: average over replicas
                        std::vector<double> smE(nsamples, 0), smA(nsamples, 0), smPT(nsamples, 0);
                        for (int s = 0; s < nsamples; s++) {
                            if (ti >= (int)all_data[s].size()) continue;
                            auto& tb = all_data[s][ti];
                            int Ms = (int)tb.energy.size();
                            int p = pos, be = bend;
                            if (be > Ms) be = Ms;
                            if (p >= Ms) continue;
                            int nn = be - p;
                            for (int r = 0; r < nrep; r++) {
                                double se = 0, sa = 0;
                                for (int i = p; i < be; i++) {
                                    se += tb.energy[i][r];
                                    sa += tb.acc[i][r];
                                }
                                smE[s] += se / nn;
                                smA[s] += sa / nn;
                            }
                            smE[s] /= nrep;
                            smA[s] /= nrep;

                            // Exchange rate averaged over replicas (use tidx = ti)
                            if (ti < (int)all_ex_sweep[s].size()) {
                                auto& esb = all_ex_sweep[s][ti];
                                int Mex = (int)esb.sweeps.size();
                                if (p < Mex) {
                                    int be_ex = (be < Mex) ? be : Mex;
                                    long long acc_sum = 0, prop_sum = 0;
                                    for (int i = p; i < be_ex; i++) {
                                        acc_sum += esb.n_acc[i];
                                        prop_sum += esb.n_prop[i];
                                    }
                                    smPT[s] = (prop_sum > 0) ? (double)acc_sum / prop_sum : 0.0;
                                }
                            }
                        }

                        // Jackknife
                        double fE = 0, fA = 0, fPT = 0;
                        for (int s = 0; s < nsamples; s++) { fE += smE[s]; fA += smA[s]; fPT += smPT[s]; }
                        fE /= nsamples; fA /= nsamples; fPT /= nsamples;

                        double jE = 0, jA = 0, jPT = 0;
                        for (int j = 0; j < nsamples; j++) {
                            double lE = 0, lA = 0, lPT = 0;
                            for (int s = 0; s < nsamples; s++) {
                                if (s == j) continue;
                                lE += smE[s]; lA += smA[s]; lPT += smPT[s];
                            }
                            lE /= (nsamples - 1); lA /= (nsamples - 1); lPT /= (nsamples - 1);
                            jE += (lE - fE) * (lE - fE);
                            jA += (lA - fA) * (lA - fA);
                            jPT += (lPT - fPT) * (lPT - fPT);
                        }
                        jE = sqrt((nsamples - 1.0) / nsamples * jE);
                        jA = sqrt((nsamples - 1.0) / nsamples * jA);
                        jPT = sqrt((nsamples - 1.0) / nsamples * jPT);

                        fprintf(fout, "%d\t%.8f\t%.8f\t%.5f\t%.5f\t%.6f\t%.6f\t%.8f\n",
                                sweep_end, fE, jE, fA, jA, fPT, jPT, T);

                        pos = bend;
                        bsize *= 2;
                        b++;
                    }
                }
                fclose(fout);
                printf("  Written %s\n", outfile);
            }
        }
    }

    // ================================================================
    // Intensity Spectrum
    // ================================================================
    {
        // Read frequencies from reference sample
        char refdir[512];
        snprintf(refdir, sizeof(refdir), "data/PT_N%d_NT%d_NR%d_S%d",
                 N, NT, nrep, labels[ref]);
        auto omega = read_frequencies(refdir, N);

        // Per-sample, per-temperature mean spectrum
        // spec_s[s][ti][k] = <I_k> averaged over (replica, iteration) configs
        std::vector<std::vector<std::vector<double>>> spec_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[512];
            snprintf(sdir, sizeof(sdir), "data/PT_N%d_NT%d_NR%d_S%d",
                     N, NT, nrep, labels[s]);
            char confdir[512];
            snprintf(confdir, sizeof(confdir), "%s/configs", sdir);

            spec_s[s].resize(ntemps, std::vector<double>(N, 0.0));

            for (int ti = 0; ti < ntemps; ti++) {
                int tidx = all_data[ref][ti].tidx;
                int nconfigs = 0;

                for (int r = 0; r < nrep; r++) {
                    auto cfiles = find_configs_pt(confdir, tidx, r);
                    for (auto& cf : cfiles) {
                        std::vector<double> Ik;
                        if (read_config_intensities(cf.c_str(), N, Ik)) {
                            for (int k = 0; k < N; k++)
                                spec_s[s][ti][k] += Ik[k];
                            nconfigs++;
                        }
                    }
                }

                if (nconfigs > 0) {
                    for (int k = 0; k < N; k++)
                        spec_s[s][ti][k] /= nconfigs;
                }
            }
            printf("  Sample S%d: spectrum computed\n", labels[s]);
        }

        // Write intensity_spectrum.dat: NT blocks, jackknife over samples
        char specfile[512];
        snprintf(specfile, sizeof(specfile), "%s/intensity_spectrum.dat", outdir);
        FILE* fsp = fopen(specfile, "w");
        if (fsp) {
            fprintf(fsp, "# Intensity spectrum: I_k = |a_k|^2 / sum_j |a_j|^2\n");
            fprintf(fsp, "# N=%d NT=%d nrep=%d nsamples=%d\n", N, NT, nrep, nsamples);
            fprintf(fsp, "# Columns: frequency  Intensity  Error_jk  Temperature\n");
            fprintf(fsp, "# NT blocks separated by blank lines\n");

            // Sort index by frequency
            std::vector<int> order(N);
            for (int k = 0; k < N; k++) order[k] = k;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return omega[a] < omega[b]; });

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                fprintf(fsp, "\n");

                for (int ik = 0; ik < N; ik++) {
                    int k = order[ik];
                    // Full sample mean
                    double fI = 0;
                    for (int s = 0; s < nsamples; s++)
                        fI += spec_s[s][ti][k];
                    fI /= nsamples;

                    // Jackknife error over samples
                    double jI = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lI = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lI += spec_s[s][ti][k];
                        }
                        lI /= (nsamples - 1);
                        jI += (lI - fI) * (lI - fI);
                    }
                    jI = sqrt((nsamples - 1.0) / nsamples * jI);

                    fprintf(fsp, "%.12f\t%.8e\t%.8e\t%.8f\n",
                            omega[k], fI, jI, T);
                }
            }
            fclose(fsp);
            printf("  Written %s\n", specfile);
        }
    }

    // ================================================================
    // Parisi Overlap Distribution (only if nrep > 1)
    // ================================================================
    if (nrep > 1) {
        printf("\n── Parisi overlap (nrep=%d, nbins=%d, threads=%d) ──\n",
               nrep, nbins, nthreads);

        // hist_s[s][ti][bin] = histogram count for sample s, temperature ti
        // We accumulate counts, then normalize at the end.
        double qmin = -1.0, qmax = 1.0;
        double dq = (qmax - qmin) / nbins;

        // Per-sample, per-temperature histograms
        std::vector<std::vector<std::vector<double>>> hist_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[512];
            snprintf(sdir, sizeof(sdir), "data/PT_N%d_NT%d_NR%d_S%d",
                     N, NT, nrep, labels[s]);
            char confdir_base[512];
            snprintf(confdir_base, sizeof(confdir_base), "%s/configs", sdir);

            hist_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));

            // Parallel over temperatures
            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                // Each thread has its own local histogram per temperature
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;
                    std::string confdir(confdir_base);

                    // Find iteration numbers common to all replicas
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = find_config_iters(confdir.c_str(), tidx, r);

                    std::vector<int> common_iters = rep_iters[0];
                    for (int r = 1; r < nrep; r++) {
                        std::vector<int> tmp;
                        std::set_intersection(common_iters.begin(), common_iters.end(),
                                              rep_iters[r].begin(), rep_iters[r].end(),
                                              std::back_inserter(tmp));
                        common_iters = tmp;
                    }

                    if (common_iters.empty()) continue;

                    std::vector<double> local_hist(nbins, 0.0);
                    long long n_overlaps = 0;

                    for (int ci = 0; ci < (int)common_iters.size(); ci++) {
                        int iter = common_iters[ci];

                        std::vector<std::vector<double>> sre(nrep), sim(nrep);
                        bool all_ok = true;
                        for (int r = 0; r < nrep; r++) {
                            char fn[768];
                            snprintf(fn, sizeof(fn), "%s/conf_T%d_r%d_iter%d.bin",
                                     confdir.c_str(), tidx, r, iter);
                            if (!read_config_spins(fn, N, sre[r], sim[r])) {
                                all_ok = false;
                                break;
                            }
                        }
                        if (!all_ok) continue;

                        for (int a = 0; a < nrep; a++) {
                            for (int b = a + 1; b < nrep; b++) {
                                double re_sum = 0.0;
                                for (int k = 0; k < N; k++)
                                    re_sum += sre[a][k] * sre[b][k] + sim[a][k] * sim[b][k];
                                double q = re_sum / (2.0 * N);

                                int bin = (int)((q - qmin) / dq);
                                if (bin < 0) bin = 0;
                                if (bin >= nbins) bin = nbins - 1;
                                local_hist[bin] += 1.0;
                                n_overlaps++;
                            }
                        }
                    }

                    // Normalize and store
                    if (n_overlaps > 0) {
                        for (int b = 0; b < nbins; b++)
                            local_hist[b] /= (n_overlaps * dq);
                    }
                    hist_s[s][ti] = std::move(local_hist);
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++)
                threads.emplace_back(worker, t);
            for (auto& th : threads)
                th.join();

            printf("  Sample S%d: overlap computed\n", labels[s]);
        }

        // Write parisi_overlap.dat: NT blocks, jackknife over samples
        char olapfile[512];
        snprintf(olapfile, sizeof(olapfile), "%s/parisi_overlap.dat", outdir);
        FILE* fol = fopen(olapfile, "w");
        if (fol) {
            fprintf(fol, "# Parisi overlap distribution P(q)\n");
            fprintf(fol, "# N=%d NT=%d nrep=%d nbins=%d nsamples=%d\n",
                    N, NT, nrep, nbins, nsamples);
            fprintf(fol, "# q = Re[ (1/(2N)) sum_i a_i^alpha * conj(a_i^beta) ]\n");
            fprintf(fol, "# Columns: q_center  P(q)  Error_jk  Temperature\n");
            fprintf(fol, "# NT blocks separated by blank lines\n");

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                fprintf(fol, "\n");

                for (int b = 0; b < nbins; b++) {
                    double qc = qmin + (b + 0.5) * dq;

                    // Full sample mean
                    double fP = 0;
                    for (int s = 0; s < nsamples; s++)
                        fP += hist_s[s][ti][b];
                    fP /= nsamples;

                    // Jackknife error
                    double jP = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lP = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lP += hist_s[s][ti][b];
                        }
                        lP /= (nsamples - 1);
                        jP += (lP - fP) * (lP - fP);
                    }
                    jP = sqrt((nsamples - 1.0) / nsamples * jP);

                    fprintf(fol, "%.12f\t%.8e\t%.8e\t%.8f\n", qc, fP, jP, T);
                }
            }
            fclose(fol);
            printf("  Written %s\n", olapfile);
        }
    }

    // ================================================================
    // Intensity Fluctuation Overlap (IFO) Distribution (only if nrep > 1)
    // ================================================================
    if (nrep > 1) {
        printf("\n── IFO overlap (nrep=%d, nbins=%d, threads=%d) ──\n",
               nrep, nbins, nthreads);

        double cmin = -1.0, cmax = 1.0;
        double dc = (cmax - cmin) / nbins;

        std::vector<std::vector<std::vector<double>>> ifo_hist_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[512];
            snprintf(sdir, sizeof(sdir), "data/PT_N%d_NT%d_NR%d_S%d",
                     N, NT, nrep, labels[s]);
            char confdir_base[512];
            snprintf(confdir_base, sizeof(confdir_base), "%s/configs", sdir);

            ifo_hist_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;
                    std::string confdir(confdir_base);

                    // Find iteration numbers common to all replicas
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = find_config_iters(confdir.c_str(), tidx, r);

                    std::vector<int> common_iters = rep_iters[0];
                    for (int r = 1; r < nrep; r++) {
                        std::vector<int> tmp;
                        std::set_intersection(common_iters.begin(), common_iters.end(),
                                              rep_iters[r].begin(), rep_iters[r].end(),
                                              std::back_inserter(tmp));
                        common_iters = tmp;
                    }

                    int nsweeps_eq = (int)common_iters.size();
                    if (nsweeps_eq < 1) continue;

                    // Read all configs: Ik[r][sweep][k]
                    std::vector<std::vector<std::vector<double>>> Ik(nrep);
                    bool all_ok = true;
                    for (int r = 0; r < nrep && all_ok; r++) {
                        Ik[r].resize(nsweeps_eq);
                        for (int ci = 0; ci < nsweeps_eq; ci++) {
                            int iter = common_iters[ci];
                            char fn[768];
                            snprintf(fn, sizeof(fn), "%s/conf_T%d_r%d_iter%d.bin",
                                     confdir.c_str(), tidx, r, iter);
                            std::vector<double> re, im;
                            if (!read_config_spins(fn, N, re, im)) {
                                all_ok = false;
                                break;
                            }
                            Ik[r][ci].resize(N);
                            for (int k = 0; k < N; k++)
                                Ik[r][ci][k] = re[k] * re[k] + im[k] * im[k];
                        }
                    }
                    if (!all_ok) continue;

                    // Compute <I_k>(r) = mean over sweeps for each replica
                    // meanI[r][k]
                    std::vector<std::vector<double>> meanI(nrep, std::vector<double>(N, 0.0));
                    for (int r = 0; r < nrep; r++) {
                        for (int ci = 0; ci < nsweeps_eq; ci++)
                            for (int k = 0; k < N; k++)
                                meanI[r][k] += Ik[r][ci][k];
                        for (int k = 0; k < N; k++)
                            meanI[r][k] /= nsweeps_eq;
                    }

                    // Compute IFO for all pairs and sweeps
                    std::vector<double> local_hist(nbins, 0.0);
                    long long n_overlaps = 0;

                    for (int ci = 0; ci < nsweeps_eq; ci++) {
                        // delta[r][k] = I_k(r,s) - <I_k>(r)
                        std::vector<std::vector<double>> delta(nrep, std::vector<double>(N));
                        std::vector<double> norm2(nrep, 0.0); // sum_k delta_k^2
                        for (int r = 0; r < nrep; r++) {
                            for (int k = 0; k < N; k++) {
                                delta[r][k] = Ik[r][ci][k] - meanI[r][k];
                                norm2[r] += delta[r][k] * delta[r][k];
                            }
                        }

                        for (int a = 0; a < nrep; a++) {
                            for (int b = a + 1; b < nrep; b++) {
                                double den = sqrt(norm2[a] * norm2[b]);
                                if (den <= 0) continue;
                                double num = 0.0;
                                for (int k = 0; k < N; k++)
                                    num += delta[a][k] * delta[b][k];
                                double C = num / den;

                                int bin = (int)((C - cmin) / dc);
                                if (bin < 0) bin = 0;
                                if (bin >= nbins) bin = nbins - 1;
                                local_hist[bin] += 1.0;
                                n_overlaps++;
                            }
                        }
                    }

                    if (n_overlaps > 0) {
                        for (int b = 0; b < nbins; b++)
                            local_hist[b] /= (n_overlaps * dc);
                    }
                    ifo_hist_s[s][ti] = std::move(local_hist);
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++)
                threads.emplace_back(worker, t);
            for (auto& th : threads)
                th.join();

            printf("  Sample S%d: IFO computed\n", labels[s]);
        }

        // Write ifo_overlap.dat: NT blocks, jackknife over samples
        char ifofile[512];
        snprintf(ifofile, sizeof(ifofile), "%s/ifo_overlap.dat", outdir);
        FILE* fif = fopen(ifofile, "w");
        if (fif) {
            fprintf(fif, "# Intensity Fluctuation Overlap (IFO) distribution P(C)\n");
            fprintf(fif, "# N=%d NT=%d nrep=%d nbins=%d nsamples=%d\n",
                    N, NT, nrep, nbins, nsamples);
            fprintf(fif, "# C^{ab} = sum_k delta_k^a delta_k^b / sqrt(sum_k (delta_k^a)^2 * sum_k (delta_k^b)^2)\n");
            fprintf(fif, "# delta_k^r = I_k^r - <I_k^r>_sweeps,  I_k = |a_k|^2\n");
            fprintf(fif, "# Columns: C_center  P(C)  Error_jk  Temperature\n");
            fprintf(fif, "# NT blocks separated by blank lines\n");

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                fprintf(fif, "\n");

                for (int b = 0; b < nbins; b++) {
                    double cc = cmin + (b + 0.5) * dc;

                    double fP = 0;
                    for (int s = 0; s < nsamples; s++)
                        fP += ifo_hist_s[s][ti][b];
                    fP /= nsamples;

                    double jP = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lP = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lP += ifo_hist_s[s][ti][b];
                        }
                        lP /= (nsamples - 1);
                        jP += (lP - fP) * (lP - fP);
                    }
                    jP = sqrt((nsamples - 1.0) / nsamples * jP);

                    fprintf(fif, "%.12f\t%.8e\t%.8e\t%.8f\n", cc, fP, jP, T);
                }
            }
            fclose(fif);
            printf("  Written %s\n", ifofile);
        }
    }

    // ================================================================
    // Plotting (if --plot)
    // ================================================================
    if (do_plot) {
        using namespace sciplot;
        printf("\n── Generating plots ────────────────────────────────\n");

        char plotdir[512];
        snprintf(plotdir, sizeof(plotdir), "%s/plots", outdir);
        mkdir_p(plotdir);

        // --- 1) Intensity Spectrum ---
        {
            // Re-read the spectrum file we just wrote
            char specfile[512];
            snprintf(specfile, sizeof(specfile), "%s/intensity_spectrum.dat", outdir);
            // Parse: blocks separated by blank lines
            // Each line: frequency  Intensity  Error_jk  Temperature
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
                    // Colorbar: same blue->teal->gold->red gradient
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
                    char pngfile[512];
                    snprintf(pngfile, sizeof(pngfile), "%s/intensity_spectrum.png", plotdir);
                    canvas.save(pngfile);
                    printf("  Written %s\n", pngfile);
                }
            }
        }

        // --- 2) Equilibrium data (from mean file) ---
        {
            char meanfile[512];
            snprintf(meanfile, sizeof(meanfile), "%s/equilibrium_data_mean.dat", outdir);
            FILE* f = fopen(meanfile, "r");
            if (f) {
                struct EqLine { double T, E, Eerr, A, Aerr, Cv, Cverr; };
                std::vector<EqLine> data;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    EqLine el;
                    if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf",
                               &el.T, &el.E, &el.Eerr, &el.A, &el.Aerr, &el.Cv, &el.Cverr) == 7)
                        data.push_back(el);
                }
                fclose(f);

                int nd = (int)data.size();
                if (nd > 0) {
                    std::vector<double> vT(nd), vE(nd), vElo(nd), vEhi(nd);
                    std::vector<double> vA(nd), vAlo(nd), vAhi(nd);
                    std::vector<double> vCv(nd), vCvlo(nd), vCvhi(nd);
                    for (int i = 0; i < nd; i++) {
                        vT[i]   = data[i].T;
                        vE[i]   = data[i].E;   vElo[i]  = data[i].E - data[i].Eerr;  vEhi[i]  = data[i].E + data[i].Eerr;
                        vA[i]   = data[i].A;   vAlo[i]  = data[i].A - data[i].Aerr;  vAhi[i]  = data[i].A + data[i].Aerr;
                        vCv[i]  = data[i].Cv;  vCvlo[i] = data[i].Cv - data[i].Cverr; vCvhi[i] = data[i].Cv + data[i].Cverr;
                    }

                    // Energy plot
                    {
                        Plot2D plot;
                        plot.xlabel("{/Times-Italic T}");
                        plot.ylabel("{/Times-Italic E} / {/Times-Italic N}");
                        plot.fontName("Times");
                        plot.fontSize(18);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                        plot.drawCurvesFilled(vT, vElo, vEhi)
                            .fillColor("#4393c3").fillIntensity(0.35).fillTransparent()
                            .lineColor("#4393c3").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vE)
                            .lineColor("#2166ac").lineWidth(2.5).label("E/N");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/energy.png", plotdir);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
                    // MC Acceptance plot
                    {
                        Plot2D plot;
                        plot.xlabel("{/Times-Italic T}");
                        plot.ylabel("MC acceptance");
                        plot.fontName("Times");
                        plot.fontSize(18);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                        plot.drawCurvesFilled(vT, vAlo, vAhi)
                            .fillColor("#66c2a5").fillIntensity(0.35).fillTransparent()
                            .lineColor("#66c2a5").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vA)
                            .lineColor("#1b7837").lineWidth(2.5).label("MC acceptance");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/acceptance.png", plotdir);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
                    // Specific heat plot
                    {
                        Plot2D plot;
                        plot.xlabel("{/Times-Italic T}");
                        plot.ylabel("{/Times-Italic C}_{/Times-Italic v}");
                        plot.fontName("Times");
                        plot.fontSize(18);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                        plot.drawCurvesFilled(vT, vCvlo, vCvhi)
                            .fillColor("#f4a582").fillIntensity(0.35).fillTransparent()
                            .lineColor("#f4a582").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vCv)
                            .lineColor("#b2182b").lineWidth(2.5).label("C_v");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/specific_heat.png", plotdir);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
                }
            }
        }

        // --- 3) Exchange rates ---
        {
            char exfile[512];
            snprintf(exfile, sizeof(exfile), "%s/exchange_rates.dat", outdir);
            FILE* f = fopen(exfile, "r");
            if (f) {
                struct ExLine { int tidx; double Th, Tl, rate, err; };
                std::vector<ExLine> data;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    ExLine el;
                    if (sscanf(line, "%d %lf %lf %lf %lf",
                               &el.tidx, &el.Th, &el.Tl, &el.rate, &el.err) == 5)
                        data.push_back(el);
                }
                fclose(f);

                int nd = (int)data.size();
                if (nd > 0) {
                    std::vector<double> vT(nd), vR(nd), vRlo(nd), vRhi(nd);
                    for (int i = 0; i < nd; i++) {
                        vT[i] = 0.5 * (data[i].Th + data[i].Tl);
                        vR[i] = data[i].rate;
                        vRlo[i] = data[i].rate - data[i].err;
                        vRhi[i] = data[i].rate + data[i].err;
                    }
                    Plot2D plot;
                    plot.xlabel("{/Times-Italic T}");
                    plot.ylabel("PT exchange rate");
                    plot.fontName("Times");
                    plot.fontSize(18);
                    plot.legend().atTopRight();
                    plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                    plot.drawCurvesFilled(vT, vRlo, vRhi)
                        .fillColor("#8da0cb").fillIntensity(0.35).fillTransparent()
                        .lineColor("#8da0cb").lineWidth(0).labelNone();
                    plot.drawCurve(vT, vR)
                        .lineColor("#542788").lineWidth(2.5).label("PT exchange");
                    Figure fig = {{plot}};
                    Canvas canvas = {{fig}};
                    canvas.size(1600, 1000);
                    char pf[512]; snprintf(pf, sizeof(pf), "%s/exchange_rates.png", plotdir);
                    canvas.save(pf);
                    printf("  Written %s\n", pf);
                }
            }
        }

        // --- 4) History plots (from history_mean.dat) ---
        {
            char histfile[512];
            snprintf(histfile, sizeof(histfile), "%s/history_mean.dat", outdir);
            FILE* f = fopen(histfile, "r");
            if (f) {
                struct HistLine { int sweep; double E, Eerr, A, Aerr, PT, PTerr, T; };
                std::vector<std::vector<HistLine>> blocks;
                std::vector<HistLine> cur;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#') continue;
                    if (line[0] == '\n' || line[0] == '\r') {
                        if (!cur.empty()) { blocks.push_back(cur); cur.clear(); }
                        continue;
                    }
                    HistLine hl;
                    if (sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf",
                               &hl.sweep, &hl.E, &hl.Eerr, &hl.A, &hl.Aerr,
                               &hl.PT, &hl.PTerr, &hl.T) == 8)
                        cur.push_back(hl);
                }
                if (!cur.empty()) blocks.push_back(cur);
                fclose(f);

                if (!blocks.empty()) {
                    int nb = (int)blocks.size();
                    double Tmin = 1e30, Tmax = -1e30;
                    for (auto& bl : blocks) {
                        if (bl[0].T < Tmin) Tmin = bl[0].T;
                        if (bl[0].T > Tmax) Tmax = bl[0].T;
                    }

                    // Helper lambda: create a history plot
                    auto make_history_plot = [&](const char* ylabel_str,
                                                 int val_col, // 0=E, 1=A, 2=PT
                                                 const char* outname) {
                        Plot2D plot;
                        plot.xlabel("sweep");
                        plot.ylabel(ylabel_str);
                        plot.fontName("Times");
                        plot.fontSize(18);
                        plot.legend().hide();
                        plot.gnuplot("set logscale x 2");
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
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
                            for (int i = 0; i < n; i++) {
                                vx[i] = bl[i].sweep;
                                if (val_col == 0) vy[i] = bl[i].E;
                                else if (val_col == 1) vy[i] = bl[i].A;
                                else vy[i] = bl[i].PT;
                            }
                            double frac = (Tmax > Tmin) ? (bl[0].T - Tmin) / (Tmax - Tmin) : 0.5;
                            plot.drawCurve(vx, vy)
                                .lineColor(temp_color(frac))
                                .lineWidth(2)
                                .label("");
                        }
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1800, 1200);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/%s", plotdir, outname);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    };

                    make_history_plot("{/Times-Italic E} / {/Times-Italic N}", 0, "energy_history.png");
                    make_history_plot("MC acceptance", 1, "acceptance_history.png");
                    make_history_plot("PT exchange rate", 2, "exchange_history.png");
                }
            }
        }

        // --- Parisi overlap P(q) plot ---
        if (nrep > 1) {
            char olapfile[512];
            snprintf(olapfile, sizeof(olapfile), "%s/parisi_overlap.dat", outdir);

            // Read the overlap file we wrote earlier
            struct OlapRow { double q, pq, err, T; };
            std::vector<std::vector<OlapRow>> oblocks;  // blocks by temperature
            {
                FILE* f = fopen(olapfile, "r");
                if (f) {
                    char line[512];
                    std::vector<OlapRow> cur;
                    while (fgets(line, sizeof(line), f)) {
                        if (line[0] == '#') continue;
                        if (line[0] == '\n') {
                            if (!cur.empty()) oblocks.push_back(cur);
                            cur.clear();
                            continue;
                        }
                        OlapRow r;
                        if (sscanf(line, "%lf %lf %lf %lf", &r.q, &r.pq, &r.err, &r.T) == 4)
                            cur.push_back(r);
                    }
                    if (!cur.empty()) oblocks.push_back(cur);
                    fclose(f);
                }
            }

            if (!oblocks.empty()) {
                Plot2D plot;
                plot.xlabel("{/Times-Italic q}");
                plot.ylabel("{/Times-Italic P}({/Times-Italic q})");
                plot.fontName("Times");
                plot.fontSize(18);
                plot.legend().hide();
                plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                plot.gnuplot("set logscale y 10");
                plot.gnuplot("set format y '10^{%L}'");

                int nb = (int)oblocks.size();
                double Tmin_ov = oblocks.back()[0].T;
                double Tmax_ov = oblocks.front()[0].T;
                plot.gnuplot("set palette defined (0 '#1A33CC', 0.33 '#1AB580', 0.66 '#CC9919', 1.0 '#FF1A0D')");
                char cbr[128];
                snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin_ov, Tmax_ov);
                plot.gnuplot(cbr);
                plot.gnuplot("set cblabel '{/Times-Italic T}' font 'Times,16'");
                plot.gnuplot("set colorbox");

                for (int bi = 0; bi < nb; bi++) {
                    auto& bl = oblocks[bi];
                    std::vector<double> vq, vpq;
                    for (int i = 0; i < (int)bl.size(); i++) {
                        if (bl[i].pq > 0) { vq.push_back(bl[i].q); vpq.push_back(bl[i].pq); }
                    }
                    if (vq.empty()) continue;
                    double frac = (nb > 1) ? 1.0 - (double)bi / (nb - 1) : 0.5;
                    plot.drawCurve(vq, vpq)
                        .lineColor(temp_color(frac))
                        .lineWidth(2)
                        .label("");
                }

                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(1800, 1200);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/parisi_overlap.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- IFO overlap P(C) plot ---
        if (nrep > 1) {
            char ifofile[512];
            snprintf(ifofile, sizeof(ifofile), "%s/ifo_overlap.dat", outdir);

            struct OlapRow { double q, pq, err, T; };
            std::vector<std::vector<OlapRow>> iblocks;
            {
                FILE* f = fopen(ifofile, "r");
                if (f) {
                    char line[512];
                    std::vector<OlapRow> cur;
                    while (fgets(line, sizeof(line), f)) {
                        if (line[0] == '#') continue;
                        if (line[0] == '\n') {
                            if (!cur.empty()) iblocks.push_back(cur);
                            cur.clear();
                            continue;
                        }
                        OlapRow r;
                        if (sscanf(line, "%lf %lf %lf %lf", &r.q, &r.pq, &r.err, &r.T) == 4)
                            cur.push_back(r);
                    }
                    if (!cur.empty()) iblocks.push_back(cur);
                    fclose(f);
                }
            }

            if (!iblocks.empty()) {
                Plot2D plot;
                plot.xlabel("{/Times-Italic C}");
                plot.ylabel("{/Times-Italic P}({/Times-Italic C})");
                plot.fontName("Times");
                plot.fontSize(18);
                plot.legend().hide();
                plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 2.5 dt 2");
                plot.gnuplot("set logscale y 10");
                plot.gnuplot("set format y '10^{%L}'");

                int nb = (int)iblocks.size();
                double Tmin_if = iblocks.back()[0].T;
                double Tmax_if = iblocks.front()[0].T;
                plot.gnuplot("set palette defined (0 '#1A33CC', 0.33 '#1AB580', 0.66 '#CC9919', 1.0 '#FF1A0D')");
                char cbr[128];
                snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin_if, Tmax_if);
                plot.gnuplot(cbr);
                plot.gnuplot("set cblabel '{/Times-Italic T}' font 'Times,16'");
                plot.gnuplot("set colorbox");

                for (int bi = 0; bi < nb; bi++) {
                    auto& bl = iblocks[bi];
                    std::vector<double> vc, vpc;
                    for (int i = 0; i < (int)bl.size(); i++) {
                        if (bl[i].pq > 0) { vc.push_back(bl[i].q); vpc.push_back(bl[i].pq); }
                    }
                    if (vc.empty()) continue;
                    double frac = (nb > 1) ? 1.0 - (double)bi / (nb - 1) : 0.5;
                    plot.drawCurve(vc, vpc)
                        .lineColor(temp_color(frac))
                        .lineWidth(2)
                        .label("");
                }

                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(1800, 1200);
                char ipf[512]; snprintf(ipf, sizeof(ipf), "%s/ifo_overlap.png", plotdir);
                canvas.save(ipf);
                printf("  Written %s\n", ipf);
            }
        }
    }

    printf("\n── Done ────────────────────────────────────────────\n\n");
    return 0;
}
