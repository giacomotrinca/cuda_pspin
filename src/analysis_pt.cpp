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
#include <sys/stat.h>
#include <dirent.h>

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

// Per-temperature-pair exchange data for one sample
struct ExBlock {
    int tidx;
    double T_high, T_low;
    long long total_acc, total_prop;
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
    fprintf(stderr, "Usage: %s -N <N> -NT <NT> [-nrep <nrep>]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0, NT = 0, nrep = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-NT") == 0 && i + 1 < argc)
            NT = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else usage(argv[0]);
    }
    if (N < 4 || NT < 2) usage(argv[0]);

    auto labels = find_samples(N, NT, nrep);
    if (labels.empty()) {
        fprintf(stderr, "No sample directories found for data/PT_N%d_NT%d_NR%d_S*\n", N, NT, nrep);
        return 1;
    }
    int nsamples = (int)labels.size();

    char outdir[256];
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
    for (int s = 0; s < nsamples; s++) {
        char sdir[256];
        snprintf(sdir, sizeof(sdir), "data/PT_N%d_NT%d_NR%d_S%d", N, NT, nrep, labels[s]);
        all_data[s] = read_pt_data(sdir, nrep);
        all_ex[s] = read_exchange_data(sdir);
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
    // Intensity Spectrum
    // ================================================================
    {
        // Read frequencies from reference sample
        char refdir[256];
        snprintf(refdir, sizeof(refdir), "data/PT_N%d_NT%d_NR%d_S%d",
                 N, NT, nrep, labels[ref]);
        auto omega = read_frequencies(refdir, N);

        // Per-sample, per-temperature mean spectrum
        // spec_s[s][ti][k] = <I_k> averaged over (replica, iteration) configs
        std::vector<std::vector<std::vector<double>>> spec_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
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

    printf("\n── Done ────────────────────────────────────────────\n\n");
    return 0;
}
