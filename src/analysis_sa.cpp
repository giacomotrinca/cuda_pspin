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

// ================================================================

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
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>] [--plot]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0;
    int nrep = 1;
    bool do_plot = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "--plot") == 0)
            do_plot = true;
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

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║           p-Spin 2+4 :: Analysis (SA)            ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  %-22s %d\n", "N", N);
    printf("  %-22s %d\n", "nrep", nrep);
    printf("  %-22s %d\n", "samples", nsamples);
    printf("  %-22s %s/\n\n", "output", outdir);

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

    // ================================================================
    // Intensity Spectrum
    // ================================================================
    {
        // Read frequencies from reference sample
        char refdir[256];
        snprintf(refdir, sizeof(refdir), "data/SA_N%d_NR%d_S%d", N, nrep, labels[ref]);
        auto omega = read_frequencies(refdir, N);

        // Per-sample, per-temperature mean spectrum
        // spec_s[s][ti][k] = <I_k> averaged over replicas for sample s at temp ti
        std::vector<std::vector<std::vector<double>>> spec_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/SA_N%d_NR%d_S%d", N, nrep, labels[s]);
            char confdir[512];
            snprintf(confdir, sizeof(confdir), "%s/configs", sdir);

            spec_s[s].resize(ntemps, std::vector<double>(N, 0.0));

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                int nconfigs = 0;

                for (int r = 0; r < nrep; r++) {
                    // SA config naming: conf_r{rep}_T{T:.6f}.bin
                    char conffile[768];
                    snprintf(conffile, sizeof(conffile),
                             "%s/conf_r%d_T%.6f.bin", confdir, r, T);
                    std::vector<double> Ik;
                    if (read_config_intensities(conffile, N, Ik)) {
                        for (int k = 0; k < N; k++)
                            spec_s[s][ti][k] += Ik[k];
                        nconfigs++;
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
            fprintf(fsp, "# N=%d nrep=%d nsamples=%d\n", N, nrep, nsamples);
            fprintf(fsp, "# Columns: frequency  Intensity  Error_jk  Temperature\n");
            fprintf(fsp, "# %d temperature blocks separated by blank lines\n", ntemps);

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
                    double fI = 0;
                    for (int s = 0; s < nsamples; s++)
                        fI += spec_s[s][ti][k];
                    fI /= nsamples;

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

        // --- 2) Equilibrium data ---
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

                    // Energy
                    {
                        Plot2D plot;
                        plot.xlabel("Temperature T"); plot.ylabel("Energy  E/N");
                        plot.fontName("Helvetica"); plot.fontSize(16);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 0.8 dt 3");
                        plot.gnuplot("set border lw 1.5");
                        plot.gnuplot("set tics font 'Helvetica,13'");
                        plot.drawCurvesFilled(vT, vElo, vEhi)
                            .fillColor("#4393c3").fillIntensity(0.35).fillTransparent()
                            .lineColor("#4393c3").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vE)
                            .lineColor("#2166ac").lineWidth(2.5).label("E/N");
                        Figure fig = {{plot}}; Canvas canvas = {{fig}}; canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/energy.png", plotdir);
                        canvas.save(pf); printf("  Written %s\n", pf);
                    }
                    // MC Acceptance
                    {
                        Plot2D plot;
                        plot.xlabel("Temperature T"); plot.ylabel("MC acceptance rate");
                        plot.fontName("Helvetica"); plot.fontSize(16);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 0.8 dt 3");
                        plot.gnuplot("set border lw 1.5");
                        plot.gnuplot("set tics font 'Helvetica,13'");
                        plot.drawCurvesFilled(vT, vAlo, vAhi)
                            .fillColor("#66c2a5").fillIntensity(0.35).fillTransparent()
                            .lineColor("#66c2a5").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vA)
                            .lineColor("#1b7837").lineWidth(2.5).label("MC acceptance");
                        Figure fig = {{plot}}; Canvas canvas = {{fig}}; canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/acceptance.png", plotdir);
                        canvas.save(pf); printf("  Written %s\n", pf);
                    }
                    // Specific heat
                    {
                        Plot2D plot;
                        plot.xlabel("Temperature T"); plot.ylabel("Specific heat  C_v");
                        plot.fontName("Helvetica"); plot.fontSize(16);
                        plot.legend().atTopRight();
                        plot.gnuplot("set grid ls 0 lc rgb '#CCCCCC' lw 0.8 dt 3");
                        plot.gnuplot("set border lw 1.5");
                        plot.gnuplot("set tics font 'Helvetica,13'");
                        plot.drawCurvesFilled(vT, vCvlo, vCvhi)
                            .fillColor("#f4a582").fillIntensity(0.35).fillTransparent()
                            .lineColor("#f4a582").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vCv)
                            .lineColor("#b2182b").lineWidth(2.5).label("C_v");
                        Figure fig = {{plot}}; Canvas canvas = {{fig}}; canvas.size(1600, 1000);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/specific_heat.png", plotdir);
                        canvas.save(pf); printf("  Written %s\n", pf);
                    }
                }
            }
        }
    }

    printf("\n── Done ────────────────────────────────────────────\n\n");
    return 0;
}
