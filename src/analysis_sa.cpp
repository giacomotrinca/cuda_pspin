// Analysis of simulated annealing data.
//
// Reads energy_accept.txt from data/SA_N{N}_NR{nrep}_S{0,1,...} directories.
// For each temperature, uses the second half of the time series to compute
// mean energy, acceptance, and specific heat (jackknife errors over samples).
//
// Output directory: analysis/SA_N{N}_NR{nrep}/
//   equilibrium_data_nr{r}.dat  — per-replica results
//   equilibrium_data_mean.dat   — replica-averaged results
//
// Columns:
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

// One measurement row: T, sweep, and per-replica E/N and acceptance
struct Row {
    double T;
    int sweep;
    std::vector<double> energy; // E/N per replica
    std::vector<double> acc;    // acceptance per replica
};

// Per-temperature data: all rows at a given T for one sample
struct TempBlock {
    double T;
    std::vector<int> sweeps;
    std::vector<std::vector<double>> energy; // [measurement][replica]
    std::vector<std::vector<double>> acc;
};

// Read energy_accept.txt, return rows grouped by temperature (in order of appearance)
static std::vector<TempBlock> read_sa_data(const char* datadir, int nrep) {
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/energy_accept.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) return {};

    // Parse all rows
    std::vector<Row> rows;
    char line[4096];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        Row row;
        row.energy.resize(nrep);
        row.acc.resize(nrep);

        char* tok = strtok(line, " \t\n");
        if (!tok) continue;
        row.T = atof(tok);

        tok = strtok(nullptr, " \t\n");
        if (!tok) continue;
        row.sweep = atoi(tok);

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

    // Group by temperature in order of appearance
    std::vector<TempBlock> blocks;
    std::map<double, int> temp_idx; // T -> index in blocks

    for (auto& row : rows) {
        auto it = temp_idx.find(row.T);
        int idx;
        if (it == temp_idx.end()) {
            idx = (int)blocks.size();
            temp_idx[row.T] = idx;
            TempBlock tb;
            tb.T = row.T;
            blocks.push_back(tb);
        } else {
            idx = it->second;
        }
        blocks[idx].sweeps.push_back(row.sweep);
        blocks[idx].energy.push_back(row.energy);
        blocks[idx].acc.push_back(row.acc);
    }
    return blocks;
}

// Find all sample directories matching data/SA_N{N}_NR{nrep}_S*
static std::vector<int> find_samples(int N, int nrep) {
    std::vector<int> labels;
    DIR* dir = opendir("data");
    if (!dir) return labels;

    char pat[128];
    snprintf(pat, sizeof(pat), "SA_N%d_NR%d_S", N, nrep);
    int plen = (int)strlen(pat);

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, pat, plen) == 0) {
            int label = atoi(ent->d_name + plen);
            char check[128];
            snprintf(check, sizeof(check), "SA_N%d_NR%d_S%d", N, nrep, label);
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
    double e_mean;  // <e>  (e = E/N)
    double e2_mean; // <e^2>
    double a_mean;  // <acceptance>
};

// Compute observables from the second half of a TempBlock for one replica
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
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0;
    int nrep = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else usage(argv[0]);
    }
    if (N < 4) usage(argv[0]);

    // Find samples
    auto labels = find_samples(N, nrep);
    if (labels.empty()) {
        fprintf(stderr, "No sample directories found for data/SA_N%d_NR%d_S*\n", N, nrep);
        return 1;
    }
    int nsamples = (int)labels.size();

    // Output directory
    char outdir[256];
    snprintf(outdir, sizeof(outdir), "analysis/SA_N%d_NR%d", N, nrep);
    mkdir_p(outdir);

    printf("=== SA Analysis ===\n");
    printf("N=%d  nrep=%d  samples=%d\n", N, nrep, nsamples);
    printf("Output: %s/\n", outdir);

    // Read all sample data
    // all_data[sample_idx] = vector of TempBlocks (one per temperature)
    std::vector<std::vector<TempBlock>> all_data(nsamples);
    for (int s = 0; s < nsamples; s++) {
        char sdir[256];
        snprintf(sdir, sizeof(sdir), "data/SA_N%d_NR%d_S%d", N, nrep, labels[s]);
        all_data[s] = read_sa_data(sdir, nrep);
        if (all_data[s].empty()) {
            fprintf(stderr, "Warning: no data in %s\n", sdir);
        }
    }

    // Use temperature list from first non-empty sample
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
        fprintf(fout, "# N=%d nrep=%d replica=%d nsamples=%d\n", N, nrep, r, nsamples);
        fprintf(fout, "# Energy = E/N (as stored). Cv = N*(<e^2>-<e>^2)/T^2. Jackknife over %d samples.\n", nsamples);

        for (int ti = 0; ti < ntemps; ti++) {
            double T = all_data[ref][ti].T;

            // Per-sample observables
            std::vector<double> sE(nsamples), sC(nsamples), sA(nsamples);
            for (int s = 0; s < nsamples; s++) {
                if (ti >= (int)all_data[s].size()) continue;
                SampleObs obs = compute_obs(all_data[s][ti], r);
                sE[s] = obs.e_mean;
                sA[s] = obs.a_mean;
                sC[s] = N * (obs.e2_mean - obs.e_mean * obs.e_mean) / (T * T);
            }

            // Full means
            double fE = 0, fA = 0, fC = 0;
            for (int s = 0; s < nsamples; s++) { fE += sE[s]; fA += sA[s]; fC += sC[s]; }
            fE /= nsamples; fA /= nsamples; fC /= nsamples;

            // Jackknife errors
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
        fprintf(fout, "# N=%d nrep=%d nsamples=%d\n", N, nrep, nsamples);
        fprintf(fout, "# Replica-averaged, then jackknife over %d samples. Cv = N*(<e^2>-<e>^2)/T^2.\n", nsamples);

        for (int ti = 0; ti < ntemps; ti++) {
            double T = all_data[ref][ti].T;

            // Per-sample: average over replicas
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

            // Full means
            double fE = 0, fA = 0, fC = 0;
            for (int s = 0; s < nsamples; s++) { fE += smE[s]; fA += smA[s]; fC += smC[s]; }
            fE /= nsamples; fA /= nsamples; fC /= nsamples;

            // Jackknife errors
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

    printf("=== SA Analysis done ===\n");
    return 0;
}
