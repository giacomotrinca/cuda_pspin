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
#include <set>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <sys/stat.h>
#include <dirent.h>
#include <sciplot/sciplot.hpp>

// Color interpolation for temperature gradient (cool blue → warm red)
// Uses a curated 5-stop gradient inspired by RdYlBu (reversed)
static std::string temp_color(double frac) {
    // frac in [0,1]:  0 = coldest (blue), 1 = hottest (red)
    struct RGB { double r, g, b; };
    // 5 control points: deep blue → sky → pale yellow → orange → crimson
    static const RGB stops[] = {
        {0.129, 0.400, 0.675},  // #2166AC  deep blue
        {0.400, 0.663, 0.812},  // #67A9CF  sky blue
        {0.545, 0.741, 0.341},  // #8BBD57  yellow-green
        {0.906, 0.541, 0.384},  // #E78A62  warm orange
        {0.698, 0.094, 0.169},  // #B2182B  crimson
    };
    const int ns = 5;
    double t = frac * (ns - 1);
    int i = (int)t;
    if (i >= ns - 1) i = ns - 2;
    double s = t - i;
    double r = stops[i].r + s * (stops[i+1].r - stops[i].r);
    double g = stops[i].g + s * (stops[i+1].g - stops[i].g);
    double b = stops[i].b + s * (stops[i+1].b - stops[i].b);
    char hex[16];
    snprintf(hex, sizeof(hex), "#%02X%02X%02X",
             (int)(r*255), (int)(g*255), (int)(b*255));
    return hex;
}

// gnuplot palette matching temp_color stops
static const char* TEMP_PALETTE =
    "set palette defined ("
    "0 '#2166AC', 0.25 '#67A9CF', 0.5 '#8BBD57', 0.75 '#E78A62', 1.0 '#B2182B'"
    ")";

// Compute mean, variance, skewness, excess kurtosis from a normalised histogram
static void hist_moments(const std::vector<double>& hist, int nbins,
                         double xmin, double dx,
                         double& mu, double& var, double& skew, double& kurt)
{
    mu = 0; var = 0; skew = 0; kurt = 0;
    for (int b = 0; b < nbins; b++) {
        double xc = xmin + (b + 0.5) * dx;
        mu += xc * hist[b] * dx;
    }
    for (int b = 0; b < nbins; b++) {
        double xc = xmin + (b + 0.5) * dx;
        double d = xc - mu;
        double d2 = d * d;
        var  += d2 * hist[b] * dx;
        skew += d2 * d * hist[b] * dx;
        kurt += d2 * d2 * hist[b] * dx;
    }
    if (var > 0) {
        skew /= (var * sqrt(var));
        kurt = kurt / (var * var) - 3.0;  // excess kurtosis
    }
}

// Write a _moments.dat file and return the data for plotting
struct MomRow { double T, mu, mu_e, var, var_e, skew, skew_e, kurt, kurt_e; };
static std::vector<MomRow> write_moments(
    const char* path, const char* header,
    const std::vector<std::vector<std::vector<double>>>& hist_s,
    int nsamples, int ntemps, int nbins,
    double xmin, double dx,
    const std::vector<double>& temps)
{
    std::vector<MomRow> rows(ntemps);
    for (int ti = 0; ti < ntemps; ti++) {
        double T = temps[ti];
        // full-sample mean histogram
        std::vector<double> mh(nbins, 0.0);
        for (int b = 0; b < nbins; b++) {
            for (int s = 0; s < nsamples; s++) mh[b] += hist_s[s][ti][b];
            mh[b] /= nsamples;
        }
        double fmu, fvar, fskew, fkurt;
        hist_moments(mh, nbins, xmin, dx, fmu, fvar, fskew, fkurt);
        // jackknife
        double jmu=0, jvar=0, jskew=0, jkurt=0;
        for (int j = 0; j < nsamples; j++) {
            std::vector<double> lh(nbins, 0.0);
            for (int b = 0; b < nbins; b++) {
                for (int s = 0; s < nsamples; s++) {
                    if (s == j) continue;
                    lh[b] += hist_s[s][ti][b];
                }
                lh[b] /= (nsamples - 1);
            }
            double lm, lv, ls, lk;
            hist_moments(lh, nbins, xmin, dx, lm, lv, ls, lk);
            jmu  += (lm - fmu)*(lm - fmu);
            jvar += (lv - fvar)*(lv - fvar);
            jskew+= (ls - fskew)*(ls - fskew);
            jkurt+= (lk - fkurt)*(lk - fkurt);
        }
        double f = sqrt((nsamples - 1.0) / nsamples);
        rows[ti] = {T, fmu, f*sqrt(jmu), fvar, f*sqrt(jvar),
                       fskew, f*sqrt(jskew), fkurt, f*sqrt(jkurt)};
    }
    FILE* fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "# %s\n", header);
        fprintf(fp, "# Columns: T  mean  mean_err  var  var_err  skew  skew_err  kurt  kurt_err\n");
        for (int ti = 0; ti < ntemps; ti++) {
            auto& r = rows[ti];
            fprintf(fp, "%.8f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n",
                    r.T, r.mu, r.mu_e, r.var, r.var_e, r.skew, r.skew_e, r.kurt, r.kurt_e);
        }
        fclose(fp);
        printf("  Written %s\n", path);
    }
    return rows;
}

// Common plot setup for analysis figures
static void setup_analysis_plot(sciplot::Plot2D& plot) {
    plot.fontName("Helvetica");
    plot.fontSize(15);
    plot.gnuplot("set border 3 lw 1.4 lc rgb '#2D2D2D'");
    plot.gnuplot("set style line 100 lt 1 lc rgb '#E8E8E8' lw 0.6");
    plot.gnuplot("set grid back ls 100");
    plot.gnuplot("set tics nomirror out scale 0.6");
    plot.gnuplot("set tics font 'Helvetica,12'");
    plot.gnuplot("set lmargin 12");
    plot.gnuplot("set rmargin 4");
    plot.gnuplot("set tmargin 2");
    plot.gnuplot("set bmargin 4.5");
    plot.gnuplot("set key opaque box lc rgb '#CCCCCC' lw 0.5");
    plot.gnuplot("set key spacing 1.3");
}

// Setup for plots with a temperature colorbar on the right
static void setup_colorbar_plot(sciplot::Plot2D& plot,
                                double Tmin, double Tmax,
                                bool log_cb = false) {
    setup_analysis_plot(plot);
    plot.gnuplot("set rmargin 14");  // room for colorbar
    plot.gnuplot(TEMP_PALETTE);
    char cbr[128];
    snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin, Tmax);
    plot.gnuplot(cbr);
    if (log_cb) plot.gnuplot("set log cb");
    plot.gnuplot("set cblabel '{/Helvetica-Oblique T}' font 'Helvetica,13' offset 1,0");
    plot.gnuplot("set cbtics font 'Helvetica,11'");
    plot.gnuplot("set colorbox vertical user origin 0.88, 0.15 size 0.025, 0.7");
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
            omega[k] = (double)k;
    }
    return omega;
}

// ================================================================
// Glass observables: chi, Binder g4, non-self-averaging A
// Computed from per-sample overlap histograms (P(q), P(C), etc.)
// ================================================================
struct GlassObs { double T, chi, chi_e, g4, g4_e, A, A_e; };

static std::vector<GlassObs> compute_glass_observables(
    const char* tag,
    const std::vector<std::vector<std::vector<double>>>& hist_s,
    int nsamples, int ntemps, int nbins, int N,
    double xmin, double dx,
    const std::vector<double>& temps,
    const char* outdir)
{
    std::vector<GlassObs> rows(ntemps);

    // Per-sample raw moments: x2_s[s][ti], x4_s[s][ti]
    std::vector<std::vector<double>> x2_s(nsamples, std::vector<double>(ntemps));
    std::vector<std::vector<double>> x4_s(nsamples, std::vector<double>(ntemps));

    for (int s = 0; s < nsamples; s++)
        for (int ti = 0; ti < ntemps; ti++) {
            double m2 = 0, m4 = 0;
            for (int b = 0; b < nbins; b++) {
                double xc = xmin + (b + 0.5) * dx;
                double xc2 = xc * xc;
                m2 += xc2 * hist_s[s][ti][b] * dx;
                m4 += xc2 * xc2 * hist_s[s][ti][b] * dx;
            }
            x2_s[s][ti] = m2;
            x4_s[s][ti] = m4;
        }

    for (int ti = 0; ti < ntemps; ti++) {
        double x2f = 0, x4f = 0;
        for (int s = 0; s < nsamples; s++) { x2f += x2_s[s][ti]; x4f += x4_s[s][ti]; }
        x2f /= nsamples;  x4f /= nsamples;

        double chi = N * x2f;
        double g4  = (x2f > 0) ? 0.5 * (3.0 - x4f / (x2f * x2f)) : 0;
        double x2v = 0;
        for (int s = 0; s < nsamples; s++)
            x2v += (x2_s[s][ti] - x2f) * (x2_s[s][ti] - x2f);
        x2v /= nsamples;
        double A = (x2f > 0) ? x2v / (x2f * x2f) : 0;

        double jc = 0, jg = 0, jA = 0;
        for (int j = 0; j < nsamples; j++) {
            double lx2 = 0, lx4 = 0;
            for (int s = 0; s < nsamples; s++) {
                if (s == j) continue;
                lx2 += x2_s[s][ti]; lx4 += x4_s[s][ti];
            }
            lx2 /= (nsamples - 1); lx4 /= (nsamples - 1);
            double lchi = N * lx2;
            double lg4  = (lx2 > 0) ? 0.5 * (3.0 - lx4 / (lx2 * lx2)) : 0;
            jc += (lchi - chi) * (lchi - chi);
            jg += (lg4  - g4)  * (lg4  - g4);
            double lv = 0;
            for (int s = 0; s < nsamples; s++) {
                if (s == j) continue;
                lv += (x2_s[s][ti] - lx2) * (x2_s[s][ti] - lx2);
            }
            lv /= (nsamples - 1);
            double lA = (lx2 > 0) ? lv / (lx2 * lx2) : 0;
            jA += (lA - A) * (lA - A);
        }
        double f = sqrt((nsamples - 1.0) / nsamples);
        rows[ti] = {temps[ti], chi, f*sqrt(jc), g4, f*sqrt(jg), A, f*sqrt(jA)};
    }

    // Write glass_observables file
    char gf[512];
    snprintf(gf, sizeof(gf), "%s/%s_glass_observables.dat", outdir, tag);
    FILE* fg = fopen(gf, "w");
    if (fg) {
        fprintf(fg, "# Glass observables from %s overlap\n", tag);
        fprintf(fg, "# chi = N*<x^2>,  g4 = 0.5*(3-<x^4>/<x^2>^2),  A = Var_J[<x^2>]/<x^2>^2\n");
        fprintf(fg, "# Columns: T  chi  chi_err  g4  g4_err  A  A_err\n");
        for (auto& r : rows)
            fprintf(fg, "%.8f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n",
                    r.T, r.chi, r.chi_e, r.g4, r.g4_e, r.A, r.A_e);
        fclose(fg);
        printf("  Written %s\n", gf);
    }

    // Write per-sample <x^2>
    char sf[512];
    snprintf(sf, sizeof(sf), "%s/%s_sample_x2.dat", outdir, tag);
    FILE* fs = fopen(sf, "w");
    if (fs) {
        fprintf(fs, "# Per-sample <x^2> for %s\n", tag);
        fprintf(fs, "# Columns: T  sample_0  sample_1  ...  sample_{n-1}\n");
        for (int ti = 0; ti < ntemps; ti++) {
            fprintf(fs, "%.8f", temps[ti]);
            for (int s = 0; s < nsamples; s++)
                fprintf(fs, "\t%.8e", x2_s[s][ti]);
            fprintf(fs, "\n");
        }
        fclose(fs);
        printf("  Written %s\n", sf);
    }

    return rows;
}

// ================================================================
// Quartet helpers for link overlap
// ================================================================
struct AnalysisQuartet { int i, j, k, l, ch; };

static std::vector<AnalysisQuartet> read_quartets(const char* datadir) {
    std::vector<AnalysisQuartet> quarts;
    char qfile[512];
    snprintf(qfile, sizeof(qfile), "%s/quartets.txt", datadir);
    FILE* f = fopen(qfile, "r");
    if (!f) return quarts;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int idx, qi, qj, qk, ql, qch;
        double g;
        if (sscanf(line, "%d %d %d %d %d %d %lf", &idx, &qi, &qj, &qk, &ql, &qch, &g) == 7)
            quarts.push_back({qi, qj, qk, ql, qch});
    }
    fclose(f);
    return quarts;
}

// Re[product] for a sparse quartet with given conjugation channel
// data layout: re0 im0 re1 im1 ...
static double quartet_term(const double* data, int si, int sj, int sk, int sl, int ch) {
    double ri = data[2*si], xi = data[2*si+1];
    double rj = data[2*sj], xj = data[2*sj+1];
    double rk = data[2*sk], xk = data[2*sk+1];
    double rl = data[2*sl], xl = data[2*sl+1];
    // ch=0: a_i * a_j * conj(a_k) * conj(a_l)
    // ch=1: a_i * conj(a_j) * conj(a_k) * a_l
    // ch=2: a_i * conj(a_j) * a_k * conj(a_l)
    if (ch == 0)      { xk = -xk; xl = -xl; }
    else if (ch == 1) { xj = -xj; xk = -xk; }
    else              { xj = -xj; xl = -xl; }
    double p01r = ri*rj - xi*xj, p01i = ri*xj + xi*rj;
    double p23r = rk*rl - xk*xl, p23i = rk*xl + xk*rl;
    return p01r * p23r - p01i * p23i;
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



// ----------------------------------------------------------------
// ConfigStore: loads all spin configurations from configs.bin
//              (new single-file format) or individual conf_T*_r*_iter*.bin
//              files (old format).  Data is indexed by [tidx][rep].
// ----------------------------------------------------------------
struct ConfigStore {
    struct Entry { int iter; std::vector<double> data; }; // 2*N doubles
    std::vector<std::vector<std::vector<Entry>>> entries; // [tidx][rep]
    int N, NT, nrep;

    bool load(const char* datadir, int N_, int NT_, int nrep_) {
        N = N_; NT = NT_; nrep = nrep_;
        entries.assign(NT, std::vector<std::vector<Entry>>(nrep));

        // Try configs.bin first
        char binpath[512];
        snprintf(binpath, sizeof(binpath), "%s/configs.bin", datadir);
        FILE* f = fopen(binpath, "rb");
        if (f) {
            int hdr[3];
            if (fread(hdr, sizeof(int), 3, f) == 3
                && hdr[0] == N && hdr[1] == NT && hdr[2] == nrep) {
                int rec = 2 * N;
                while (true) {
                    int sweep;
                    if (fread(&sweep, sizeof(int), 1, f) != 1) break;
                    bool ok = true;
                    for (int t = 0; t < NT && ok; t++)
                        for (int r = 0; r < nrep && ok; r++) {
                            Entry e;
                            e.iter = sweep;
                            e.data.resize(rec);
                            if ((int)fread(e.data.data(), sizeof(double), rec, f) != rec)
                                ok = false;
                            else
                                entries[t][r].push_back(std::move(e));
                        }
                    if (!ok) break;
                }
                fclose(f);
                return true;
            }
            fclose(f);
        }

        // Fallback: individual files in datadir/configs/
        char confdir[512];
        snprintf(confdir, sizeof(confdir), "%s/configs", datadir);
        for (int t = 0; t < NT; t++)
            for (int r = 0; r < nrep; r++) {
                auto iters = find_config_iters(confdir, t, r);
                for (int it : iters) {
                    char fn[768];
                    snprintf(fn, sizeof(fn), "%s/conf_T%d_r%d_iter%d.bin",
                             confdir, t, r, it);
                    FILE* fc = fopen(fn, "rb");
                    if (!fc) continue;
                    Entry e;
                    e.iter = it;
                    e.data.resize(2 * N);
                    if ((int)fread(e.data.data(), sizeof(double), 2*N, fc) == 2*N)
                        entries[t][r].push_back(std::move(e));
                    fclose(fc);
                }
            }
        return true;
    }

    std::vector<int> get_iters(int tidx, int rep) const {
        std::vector<int> v;
        if (tidx >= 0 && tidx < NT && rep >= 0 && rep < nrep)
            for (auto& e : entries[tidx][rep]) v.push_back(e.iter);
        return v;
    }

    const Entry* find_entry(int tidx, int rep, int iter) const {
        if (tidx < 0 || tidx >= NT || rep < 0 || rep >= nrep) return nullptr;
        for (auto& e : entries[tidx][rep])
            if (e.iter == iter) return &e;
        return nullptr;
    }
};

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

// Find all sample directories matching data/{prefix}_N{N}_NT{NT}_NR{nrep}_S*
static std::vector<int> find_samples(int N, int NT, int nrep, const char* prefix) {
    std::vector<int> labels;
    DIR* dir = opendir("data");
    if (!dir) return labels;

    char pat[128];
    snprintf(pat, sizeof(pat), "%s_N%d_NT%d_NR%d_S", prefix, N, NT, nrep);
    int plen = (int)strlen(pat);

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (strncmp(ent->d_name, pat, plen) == 0) {
            int label = atoi(ent->d_name + plen);
            char check[128];
            snprintf(check, sizeof(check), "%s_N%d_NT%d_NR%d_S%d", prefix, N, NT, nrep, label);
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
    fprintf(stderr, "Usage: %s -N <N> -NT <NT> [-nrep <nrep>] [-nbins <nbins>] [-nbins_spec <B>] [-nthreads <t>] [--sparse] [--plot] [--log-temp]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0, NT = 0, nrep = 1, nbins = 100, nbins_spec = 0, nthreads = 0;
    bool do_plot = false;
    bool sparse = false;
    bool log_temp = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-NT") == 0 && i + 1 < argc)
            NT = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nbins") == 0 && i + 1 < argc)
            nbins = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nbins_spec") == 0 && i + 1 < argc)
            nbins_spec = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nthreads") == 0 && i + 1 < argc)
            nthreads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sparse") == 0)
            sparse = true;
        else if (strcmp(argv[i], "--plot") == 0)
            do_plot = true;
        else if (strcmp(argv[i], "--log-temp") == 0)
            log_temp = true;
        else usage(argv[0]);
    }
    const char* prefix = sparse ? "PTS" : "PT";
    if (nthreads <= 0)
        nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (N < 4 || NT < 2) usage(argv[0]);
    if (nbins_spec <= 0) nbins_spec = nbins;

    auto labels = find_samples(N, NT, nrep, prefix);
    if (labels.empty()) {
        fprintf(stderr, "No sample directories found for data/%s_N%d_NT%d_NR%d_S*\n", prefix, N, NT, nrep);
        return 1;
    }
    int nsamples = (int)labels.size();

    char outdir[256];
    snprintf(outdir, sizeof(outdir), "analysis/%s_N%d_NT%d_NR%d", prefix, N, NT, nrep);
    mkdir_p(outdir);

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║       p-Spin 2+4 :: Analysis (%s)            ║\n", prefix);
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  %-22s %d\n", "N", N);
    printf("  %-22s %d\n", "NT", NT);
    printf("  %-22s %d\n", "nrep", nrep);
    printf("  %-22s %s\n", "mode", sparse ? "sparse (PTS)" : "standard (PT)");
    printf("  %-22s %d\n", "samples", nsamples);
    printf("  %-22s %s/\n\n", "output", outdir);

    // Read all sample data
    std::vector<std::vector<TempBlock>> all_data(nsamples);
    std::vector<std::vector<ExBlock>> all_ex(nsamples);
    std::vector<std::vector<ExSweepBlock>> all_ex_sweep(nsamples);
    for (int s = 0; s < nsamples; s++) {
        char sdir[256];
        snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d", prefix, N, NT, nrep, labels[s]);
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
    std::vector<double> temps(ntemps);
    for (int ti = 0; ti < ntemps; ti++) temps[ti] = all_data[ref][ti].T;

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
                    // Skip truncated remainder blocks (e.g. 1 leftover measurement)
                    if (bend - pos < bsize) { pos = bend; bsize *= 2; continue; }
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
                        // Skip truncated remainder blocks (e.g. 1 leftover measurement)
                        if (bend - pos < bsize) { pos = bend; bsize *= 2; b++; continue; }
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
    // Intensity Spectrum (rebinned)
    // ================================================================
    {
        printf("\n── Intensity spectrum (nbins_spec=%d) ──────────────\n", nbins_spec);

        // Read per-sample frequencies
        std::vector<std::vector<double>> omega_s(nsamples);
        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            omega_s[s] = read_frequencies(sdir, N);
        }

        // Rebinned spectrum: spec_bin_s[s][ti][b]
        // Uniform bins on [0,1], bin width = 1.0 / nbins_spec
        double dw = 1.0 / nbins_spec;
        std::vector<std::vector<std::vector<double>>> spec_bin_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            spec_bin_s[s].resize(ntemps, std::vector<double>(nbins_spec, 0.0));

            for (int ti = 0; ti < ntemps; ti++) {
                int tidx = all_data[ref][ti].tidx;
                int nconfigs = 0;

                // Accumulate rebinned spectrum over configs
                std::vector<double> bin_acc(nbins_spec, 0.0);

                for (int r = 0; r < nrep; r++) {
                    for (auto& entry : cstore.entries[tidx][r]) {
                        double total = 0;
                        std::vector<double> Ik(N);
                        for (int k = 0; k < N; k++) {
                            double re = entry.data[2*k], im = entry.data[2*k+1];
                            Ik[k] = re * re + im * im;
                            total += Ik[k];
                        }
                        if (total > 0) {
                            for (int k = 0; k < N; k++) Ik[k] /= total;
                            // Rebin: assign I_k to bin based on omega_s[s][k]
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
                        spec_bin_s[s][ti][b] = bin_acc[b] / nconfigs;
                }
            }
            printf("  Sample S%d: spectrum computed\n", labels[s]);
        }

        // Write intensity_spectrum.dat: NT blocks, jackknife over samples
        char specfile[512];
        snprintf(specfile, sizeof(specfile), "%s/intensity_spectrum.dat", outdir);
        FILE* fsp = fopen(specfile, "w");
        if (fsp) {
            fprintf(fsp, "# Rebinned intensity spectrum: I(omega) on uniform grid [0,1]\n");
            fprintf(fsp, "# N=%d NT=%d nrep=%d nsamples=%d nbins_spec=%d\n",
                    N, NT, nrep, nsamples, nbins_spec);
            fprintf(fsp, "# Columns: omega_center  Intensity  Error_jk  Temperature\n");
            fprintf(fsp, "# NT blocks separated by blank lines\n");

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                fprintf(fsp, "\n");

                for (int b = 0; b < nbins_spec; b++) {
                    double wc = (b + 0.5) * dw;

                    // Full sample mean
                    double fI = 0;
                    for (int s = 0; s < nsamples; s++)
                        fI += spec_bin_s[s][ti][b];
                    fI /= nsamples;

                    // Jackknife error over samples
                    double jI = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lI = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lI += spec_bin_s[s][ti][b];
                        }
                        lI /= (nsamples - 1);
                        jI += (lI - fI) * (lI - fI);
                    }
                    jI = sqrt((nsamples - 1.0) / nsamples * jI);

                    fprintf(fsp, "%.12f\t%.8e\t%.8e\t%.8f\n", wc, fI, jI, T);
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
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            hist_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));

            // Parallel over temperatures
            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                // Each thread has its own local histogram per temperature
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;

                    // Find iteration numbers common to all replicas
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = cstore.get_iters(tidx, r);

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

                    // Pre-allocate spin buffers outside sweep loop
                    std::vector<std::vector<double>> sre(nrep, std::vector<double>(N));
                    std::vector<std::vector<double>> sim(nrep, std::vector<double>(N));

                    for (int ci = 0; ci < (int)common_iters.size(); ci++) {
                        int iter = common_iters[ci];

                        bool all_ok = true;
                        for (int r = 0; r < nrep; r++) {
                            auto* ep = cstore.find_entry(tidx, r, iter);
                            if (!ep) { all_ok = false; break; }
                            for (int k = 0; k < N; k++) {
                                sre[r][k] = ep->data[2*k];
                                sim[r][k] = ep->data[2*k+1];
                            }
                        }
                        if (!all_ok) continue;

                        for (int a = 0; a < nrep; a++) {
                            for (int b = a + 1; b < nrep; b++) {
                                double re_sum = 0.0;
                                for (int k = 0; k < N; k++)
                                    re_sum += sre[a][k] * sre[b][k] + sim[a][k] * sim[b][k];
                                double q = re_sum / N;

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
            fprintf(fol, "# q = Re[ (1/N) sum_i a_i^alpha * conj(a_i^beta) ]\n");
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

        // Moments of P(q)
        {
            char mf[512]; snprintf(mf, sizeof(mf), "%s/parisi_moments.dat", outdir);
            write_moments(mf, "Moments of P(q)",
                          hist_s, nsamples, ntemps, nbins, -1.0, 2.0/nbins,
                          temps);
        }

        // Glass observables from P(q): chi_SG, g4 Binder, A non-self-averaging
        compute_glass_observables("parisi", hist_s, nsamples, ntemps, nbins, N,
                                  -1.0, 2.0/nbins, temps, outdir);
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
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            ifo_hist_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;

                    // Find iteration numbers common to all replicas
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = cstore.get_iters(tidx, r);

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
                            auto* ep = cstore.find_entry(tidx, r, iter);
                            if (!ep) { all_ok = false; break; }
                            Ik[r][ci].resize(N);
                            for (int k = 0; k < N; k++)
                                Ik[r][ci][k] = ep->data[2*k]*ep->data[2*k]
                                             + ep->data[2*k+1]*ep->data[2*k+1];
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

        // Moments of P(C) [IFO]
        {
            char mf[512]; snprintf(mf, sizeof(mf), "%s/ifo_moments.dat", outdir);
            write_moments(mf, "Moments of P(C) [IFO]",
                          ifo_hist_s, nsamples, ntemps, nbins, -1.0, 2.0/nbins,
                          temps);
        }

        // Glass observables from P(C) [IFO]: chi_IFO, g4, A
        compute_glass_observables("ifo", ifo_hist_s, nsamples, ntemps, nbins, N,
                                  -1.0, 2.0/nbins, temps, outdir);
    }

    // ================================================================
    // Experimental IFO (exp_ifo): overlap of time-averaged spectra
    // delta_k^r = <I_k>_r - <<I_k>>,  <<I_k>> = (1/R) sum_r <I_k>_r
    // C^{ab} = sum_k d_k^a d_k^b / sqrt(sum_k (d_k^a)^2 * sum_k (d_k^b)^2)
    // One C per pair of replicas (no sweep index).
    // ================================================================
    if (nrep > 1) {
        printf("\n── Exp-IFO overlap (nrep=%d, nbins=%d, threads=%d) ──\n",
               nrep, nbins, nthreads);

        double cmin = -1.0, cmax = 1.0;
        double dc = (cmax - cmin) / nbins;

        std::vector<std::vector<std::vector<double>>> exp_ifo_hist_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            exp_ifo_hist_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;

                    // Find common iterations
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = cstore.get_iters(tidx, r);

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

                    // Read configs and compute I_k
                    std::vector<std::vector<std::vector<double>>> Ik(nrep);
                    bool all_ok = true;
                    for (int r = 0; r < nrep && all_ok; r++) {
                        Ik[r].resize(nsweeps_eq);
                        for (int ci = 0; ci < nsweeps_eq; ci++) {
                            int iter = common_iters[ci];
                            auto* ep = cstore.find_entry(tidx, r, iter);
                            if (!ep) { all_ok = false; break; }
                            Ik[r][ci].resize(N);
                            for (int k = 0; k < N; k++)
                                Ik[r][ci][k] = ep->data[2*k]*ep->data[2*k]
                                             + ep->data[2*k+1]*ep->data[2*k+1];
                        }
                    }
                    if (!all_ok) continue;

                    // <I_k>_r = time average for each replica
                    std::vector<std::vector<double>> meanI(nrep, std::vector<double>(N, 0.0));
                    for (int r = 0; r < nrep; r++) {
                        for (int ci = 0; ci < nsweeps_eq; ci++)
                            for (int k = 0; k < N; k++)
                                meanI[r][k] += Ik[r][ci][k];
                        for (int k = 0; k < N; k++)
                            meanI[r][k] /= nsweeps_eq;
                    }

                    // <<I_k>> = replica average of <I_k>_r
                    std::vector<double> grandMeanI(N, 0.0);
                    for (int r = 0; r < nrep; r++)
                        for (int k = 0; k < N; k++)
                            grandMeanI[k] += meanI[r][k];
                    for (int k = 0; k < N; k++)
                        grandMeanI[k] /= nrep;

                    // delta_k^r = <I_k>_r - <<I_k>>
                    std::vector<std::vector<double>> delta(nrep, std::vector<double>(N));
                    std::vector<double> norm2(nrep, 0.0);
                    for (int r = 0; r < nrep; r++) {
                        for (int k = 0; k < N; k++) {
                            delta[r][k] = meanI[r][k] - grandMeanI[k];
                            norm2[r] += delta[r][k] * delta[r][k];
                        }
                    }

                    // C^{ab} for all pairs
                    std::vector<double> local_hist(nbins, 0.0);
                    long long n_overlaps = 0;
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

                    if (n_overlaps > 0) {
                        for (int b = 0; b < nbins; b++)
                            local_hist[b] /= (n_overlaps * dc);
                    }
                    exp_ifo_hist_s[s][ti] = std::move(local_hist);
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++)
                threads.emplace_back(worker, t);
            for (auto& th : threads)
                th.join();

            printf("  Sample S%d: exp-IFO computed\n", labels[s]);
        }

        // Write exp_ifo_overlap.dat
        char expfile[512];
        snprintf(expfile, sizeof(expfile), "%s/exp_ifo_overlap.dat", outdir);
        FILE* fex = fopen(expfile, "w");
        if (fex) {
            fprintf(fex, "# Experimental IFO distribution P(C)\n");
            fprintf(fex, "# N=%d NT=%d nrep=%d nbins=%d nsamples=%d\n",
                    N, NT, nrep, nbins, nsamples);
            fprintf(fex, "# delta_k^r = <I_k>_r - <<I_k>>,  <<I_k>> = (1/R) sum_r <I_k>_r\n");
            fprintf(fex, "# C^{ab} = sum_k d_k^a d_k^b / sqrt(sum_k (d_k^a)^2 * sum_k (d_k^b)^2)\n");
            fprintf(fex, "# Columns: C_center  P(C)  Error_jk  Temperature\n");
            fprintf(fex, "# NT blocks separated by blank lines\n");

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                fprintf(fex, "\n");

                for (int b = 0; b < nbins; b++) {
                    double cc = cmin + (b + 0.5) * dc;

                    double fP = 0;
                    for (int s = 0; s < nsamples; s++)
                        fP += exp_ifo_hist_s[s][ti][b];
                    fP /= nsamples;

                    double jP = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lP = 0;
                        for (int s = 0; s < nsamples; s++) {
                            if (s == j) continue;
                            lP += exp_ifo_hist_s[s][ti][b];
                        }
                        lP /= (nsamples - 1);
                        jP += (lP - fP) * (lP - fP);
                    }
                    jP = sqrt((nsamples - 1.0) / nsamples * jP);

                    fprintf(fex, "%.12f\t%.8e\t%.8e\t%.8f\n", cc, fP, jP, T);
                }
            }
            fclose(fex);
            printf("  Written %s\n", expfile);
        }

        // Moments of P(C) [exp-IFO]
        {
            char mf[512]; snprintf(mf, sizeof(mf), "%s/exp_ifo_moments.dat", outdir);
            write_moments(mf, "Moments of P(C) [exp-IFO]",
                          exp_ifo_hist_s, nsamples, ntemps, nbins, -1.0, 2.0/nbins,
                          temps);
        }

        // Glass observables from P(C) [exp-IFO]: chi, g4, A
        compute_glass_observables("exp_ifo", exp_ifo_hist_s, nsamples, ntemps, nbins, N,
                                  -1.0, 2.0/nbins, temps, outdir);
    }

    // ================================================================
    // P(|a|^2) marginal, IPR Y2/Y4, Phase overlap, Link overlap, Scatter
    // ================================================================
    if (nrep > 1) {
        printf("\n── P(|a|²), IPR, phase/link overlaps ───────────────\n");

        // Histogram settings
        int    nbins_a2 = 100;
        double a2_max   = 10.0;
        double da2      = a2_max / nbins_a2;
        double dqphi    = 2.0 / nbins;   // phase overlap bins on [-1,1]
        double dqlink   = 2.0 / nbins;   // link  overlap bins on [-1,1]

        // Per-sample storage
        std::vector<std::vector<std::vector<double>>> marginal_s(nsamples);
        std::vector<std::vector<double>> ipr2_s(nsamples, std::vector<double>(ntemps, 0.0));
        std::vector<std::vector<double>> ipr4_s(nsamples, std::vector<double>(ntemps, 0.0));
        std::vector<std::vector<std::vector<double>>> phase_s(nsamples);
        std::vector<std::vector<std::vector<double>>> link_s(nsamples);

        struct ScatterPt { double q, ql; };
        std::vector<std::vector<std::vector<ScatterPt>>> scat_s(nsamples);
        bool has_quartets_global = false;

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            auto quarts = read_quartets(sdir);
            int M = (int)quarts.size();
            bool hq = (M > 0);
            if (hq) has_quartets_global = true;

            marginal_s[s].resize(ntemps, std::vector<double>(nbins_a2, 0.0));
            phase_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));
            link_s[s].resize(ntemps, std::vector<double>(nbins, 0.0));
            scat_s[s].resize(ntemps);

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;

                    int tidx = all_data[ref][ti].tidx;

                    // Common iterations across replicas
                    std::vector<std::vector<int>> rep_iters(nrep);
                    for (int r = 0; r < nrep; r++)
                        rep_iters[r] = cstore.get_iters(tidx, r);
                    std::vector<int> common_iters = rep_iters[0];
                    for (int r = 1; r < nrep; r++) {
                        std::vector<int> tmp;
                        std::set_intersection(common_iters.begin(), common_iters.end(),
                                              rep_iters[r].begin(), rep_iters[r].end(),
                                              std::back_inserter(tmp));
                        common_iters = tmp;
                    }
                    if (common_iters.empty()) continue;

                    // Local accumulators
                    std::vector<double> loc_marg(nbins_a2, 0.0);
                    double loc_y2 = 0, loc_y4 = 0;
                    long long n_ipr = 0;
                    std::vector<double> loc_phase(nbins, 0.0);
                    long long n_ph = 0;
                    std::vector<double> loc_link(nbins, 0.0);
                    long long n_lk = 0;
                    std::vector<ScatterPt> loc_scat;

                    // Spin buffers & raw pointers
                    std::vector<std::vector<double>> sre(nrep, std::vector<double>(N));
                    std::vector<std::vector<double>> sim(nrep, std::vector<double>(N));
                    std::vector<const double*> raw(nrep, nullptr);

                    for (int ci = 0; ci < (int)common_iters.size(); ci++) {
                        int iter = common_iters[ci];
                        bool ok = true;
                        for (int r = 0; r < nrep; r++) {
                            auto* ep = cstore.find_entry(tidx, r, iter);
                            if (!ep) { ok = false; break; }
                            raw[r] = ep->data.data();
                            for (int k = 0; k < N; k++) {
                                sre[r][k] = ep->data[2*k];
                                sim[r][k] = ep->data[2*k+1];
                            }
                        }
                        if (!ok) continue;

                        // --- P(|a|^2) and IPR for each replica config ---
                        for (int r = 0; r < nrep; r++) {
                            double sI = 0, sI2 = 0, sI4 = 0;
                            for (int k = 0; k < N; k++) {
                                double Ik = sre[r][k]*sre[r][k] + sim[r][k]*sim[r][k];
                                sI += Ik; sI2 += Ik*Ik; sI4 += Ik*Ik*Ik*Ik;
                                int bin = (int)(Ik / da2);
                                if (bin >= 0 && bin < nbins_a2)
                                    loc_marg[bin]++;
                            }
                            double d2 = sI * sI;
                            if (d2 > 0) {
                                loc_y2 += sI2 / d2;
                                loc_y4 += sI4 / (d2 * d2);
                                n_ipr++;
                            }
                        }

                        // --- Overlap pairs: phase, link, scatter ---
                        for (int a = 0; a < nrep; a++) {
                            for (int b = a + 1; b < nrep; b++) {
                                // Phase overlap: q_phi = (1/N_eff) sum cos(theta_a - theta_b)
                                double ph = 0; int npm = 0;
                                for (int k = 0; k < N; k++) {
                                    double ra2 = sre[a][k]*sre[a][k]+sim[a][k]*sim[a][k];
                                    double rb2 = sre[b][k]*sre[b][k]+sim[b][k]*sim[b][k];
                                    if (ra2 > 1e-24 && rb2 > 1e-24) {
                                        ph += (sre[a][k]*sre[b][k]+sim[a][k]*sim[b][k])
                                              / sqrt(ra2 * rb2);
                                        npm++;
                                    }
                                }
                                double qph = (npm > 0) ? ph / npm : 0;
                                {
                                    int bin = (int)((qph + 1.0) / dqphi);
                                    if (bin < 0) bin = 0;
                                    if (bin >= nbins) bin = nbins - 1;
                                    loc_phase[bin]++;
                                    n_ph++;
                                }

                                // Parisi overlap (needed for scatter)
                                double rs = 0;
                                for (int k = 0; k < N; k++)
                                    rs += sre[a][k]*sre[b][k] + sim[a][k]*sim[b][k];
                                double qp = rs / N;

                                // Link overlap
                                if (hq) {
                                    double ls = 0;
                                    for (int qi = 0; qi < M; qi++) {
                                        auto& qq = quarts[qi];
                                        double ta = quartet_term(raw[a], qq.i, qq.j, qq.k, qq.l, qq.ch);
                                        double tb = quartet_term(raw[b], qq.i, qq.j, qq.k, qq.l, qq.ch);
                                        ls += ta * tb;
                                    }
                                    double ql = ls / M;
                                    {
                                        int bin = (int)((ql + 1.0) / dqlink);
                                        if (bin < 0) bin = 0;
                                        if (bin >= nbins) bin = nbins - 1;
                                        loc_link[bin]++;
                                        n_lk++;
                                    }
                                    loc_scat.push_back({qp, ql});
                                }
                            }
                        }
                    }

                    // Normalize histograms to probability densities
                    long long nmodes = n_ipr * N;
                    if (nmodes > 0)
                        for (auto& v : loc_marg) v /= (nmodes * da2);
                    if (n_ipr > 0) { loc_y2 /= n_ipr; loc_y4 /= n_ipr; }
                    if (n_ph > 0)
                        for (auto& v : loc_phase) v /= (n_ph * dqphi);
                    if (n_lk > 0)
                        for (auto& v : loc_link) v /= (n_lk * dqlink);

                    marginal_s[s][ti] = std::move(loc_marg);
                    ipr2_s[s][ti] = loc_y2;
                    ipr4_s[s][ti] = loc_y4;
                    phase_s[s][ti] = std::move(loc_phase);
                    link_s[s][ti] = std::move(loc_link);
                    scat_s[s][ti] = std::move(loc_scat);
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();

            printf("  Sample S%d done\n", labels[s]);
        }

        // ---- Write P(|a|^2) marginal distribution ----
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/marginal_a2.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Marginal distribution P(|a_k|^2 = I_k)\n");
                fprintf(f, "# Columns: I_center  P(I)  Error_jk  Temperature\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    double T = temps[ti];
                    fprintf(f, "\n");
                    for (int b = 0; b < nbins_a2; b++) {
                        double Ic = (b + 0.5) * da2;
                        double fP = 0;
                        for (int ss = 0; ss < nsamples; ss++) fP += marginal_s[ss][ti][b];
                        fP /= nsamples;
                        double jP = 0;
                        for (int j = 0; j < nsamples; j++) {
                            double lP = 0;
                            for (int ss = 0; ss < nsamples; ss++) {
                                if (ss == j) continue;
                                lP += marginal_s[ss][ti][b];
                            }
                            lP /= (nsamples - 1);
                            jP += (lP - fP) * (lP - fP);
                        }
                        jP = sqrt((nsamples - 1.0) / nsamples * jP);
                        fprintf(f, "%.12f\t%.8e\t%.8e\t%.8f\n", Ic, fP, jP, T);
                    }
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }

        // ---- Write IPR Y2, Y4 ----
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/ipr.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Inverse Participation Ratios\n");
                fprintf(f, "# Y2 = sum I_k^2 / (sum I_k)^2,  Y4 = sum I_k^4 / (sum I_k)^4\n");
                fprintf(f, "# 1/Y2 = effective number of participating modes\n");
                fprintf(f, "# Columns: T  Y2  Y2_err  Y4  Y4_err\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    double T = temps[ti];
                    double y2f = 0, y4f = 0;
                    for (int ss = 0; ss < nsamples; ss++) {
                        y2f += ipr2_s[ss][ti]; y4f += ipr4_s[ss][ti];
                    }
                    y2f /= nsamples; y4f /= nsamples;
                    double jy2 = 0, jy4 = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double ly2 = 0, ly4 = 0;
                        for (int ss = 0; ss < nsamples; ss++) {
                            if (ss == j) continue;
                            ly2 += ipr2_s[ss][ti]; ly4 += ipr4_s[ss][ti];
                        }
                        ly2 /= (nsamples - 1); ly4 /= (nsamples - 1);
                        jy2 += (ly2 - y2f) * (ly2 - y2f);
                        jy4 += (ly4 - y4f) * (ly4 - y4f);
                    }
                    double ff = sqrt((nsamples - 1.0) / nsamples);
                    fprintf(f, "%.8f\t%.8e\t%.8e\t%.8e\t%.8e\n",
                            T, y2f, ff*sqrt(jy2), y4f, ff*sqrt(jy4));
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }

        // ---- Write per-sample IPR for sample-by-sample analysis ----
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/ipr_sample.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Per-sample Y2 values\n");
                fprintf(f, "# Columns: T  sample_0  sample_1  ... sample_{n-1}\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    fprintf(f, "%.8f", temps[ti]);
                    for (int ss = 0; ss < nsamples; ss++)
                        fprintf(f, "\t%.8e", ipr2_s[ss][ti]);
                    fprintf(f, "\n");
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }

        // ---- Write phase overlap P(q_phi) ----
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/phase_overlap.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Phase overlap distribution P(q_phi)\n");
                fprintf(f, "# q_phi = (1/N_eff) sum_k cos(theta_k^a - theta_k^b)\n");
                fprintf(f, "# Columns: q_phi_center  P(q_phi)  Error_jk  Temperature\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    double T = temps[ti];
                    fprintf(f, "\n");
                    for (int b = 0; b < nbins; b++) {
                        double qc = -1.0 + (b + 0.5) * dqphi;
                        double fP = 0;
                        for (int ss = 0; ss < nsamples; ss++) fP += phase_s[ss][ti][b];
                        fP /= nsamples;
                        double jP = 0;
                        for (int j = 0; j < nsamples; j++) {
                            double lP = 0;
                            for (int ss = 0; ss < nsamples; ss++) {
                                if (ss == j) continue;
                                lP += phase_s[ss][ti][b];
                            }
                            lP /= (nsamples - 1);
                            jP += (lP - fP) * (lP - fP);
                        }
                        jP = sqrt((nsamples - 1.0) / nsamples * jP);
                        fprintf(f, "%.12f\t%.8e\t%.8e\t%.8f\n", qc, fP, jP, T);
                    }
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }

        // Phase overlap moments & glass observables
        {
            char mf[512];
            snprintf(mf, sizeof(mf), "%s/phase_moments.dat", outdir);
            write_moments(mf, "Moments of P(q_phi)",
                          phase_s, nsamples, ntemps, nbins, -1.0, dqphi, temps);
        }
        compute_glass_observables("phase", phase_s, nsamples, ntemps, nbins, N,
                                  -1.0, dqphi, temps, outdir);

        // ---- Link overlap (only if quartets available) ----
        if (has_quartets_global) {
            // Write link overlap P(q_link)
            {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/link_overlap.dat", outdir);
                FILE* f = fopen(fname, "w");
                if (f) {
                    fprintf(f, "# Link overlap distribution P(q_link)\n");
                    fprintf(f, "# q_link = (1/M) sum_q t_q(a)*t_q(b),  t_q = Re[prod with channel]\n");
                    fprintf(f, "# Columns: q_link_center  P(q_link)  Error_jk  Temperature\n");
                    for (int ti = 0; ti < ntemps; ti++) {
                        double T = temps[ti];
                        fprintf(f, "\n");
                        for (int b = 0; b < nbins; b++) {
                            double qc = -1.0 + (b + 0.5) * dqlink;
                            double fP = 0;
                            for (int ss = 0; ss < nsamples; ss++) fP += link_s[ss][ti][b];
                            fP /= nsamples;
                            double jP = 0;
                            for (int j = 0; j < nsamples; j++) {
                                double lP = 0;
                                for (int ss = 0; ss < nsamples; ss++) {
                                    if (ss == j) continue;
                                    lP += link_s[ss][ti][b];
                                }
                                lP /= (nsamples - 1);
                                jP += (lP - fP) * (lP - fP);
                            }
                            jP = sqrt((nsamples - 1.0) / nsamples * jP);
                            fprintf(f, "%.12f\t%.8e\t%.8e\t%.8f\n", qc, fP, jP, T);
                        }
                    }
                    fclose(f);
                    printf("  Written %s\n", fname);
                }
            }

            // Link moments & glass observables
            {
                char mf[512];
                snprintf(mf, sizeof(mf), "%s/link_moments.dat", outdir);
                write_moments(mf, "Moments of P(q_link)",
                              link_s, nsamples, ntemps, nbins, -1.0, dqlink, temps);
            }
            compute_glass_observables("link", link_s, nsamples, ntemps, nbins, N,
                                      -1.0, dqlink, temps, outdir);

            // Write scatter data: (q_parisi, q_link) pairs  [downsampled]
            {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/scatter_q_qlink.dat", outdir);
                FILE* f = fopen(fname, "w");
                if (f) {
                    const int max_pts_per_T = 2000;
                    std::mt19937 rng(42);
                    fprintf(f, "# Scatter: Parisi overlap q vs Link overlap q_link\n");
                    fprintf(f, "# Columns: q_parisi  q_link  Temperature\n");
                    fprintf(f, "# (downsampled to max %d points per temperature)\n", max_pts_per_T);
                    for (int ti = 0; ti < ntemps; ti++) {
                        double T = temps[ti];
                        fprintf(f, "\n");
                        // collect all points for this T
                        std::vector<ScatterPt> pool;
                        for (int ss = 0; ss < nsamples; ss++)
                            pool.insert(pool.end(), scat_s[ss][ti].begin(), scat_s[ss][ti].end());
                        // Fisher-Yates shuffle + truncate
                        if ((int)pool.size() > max_pts_per_T) {
                            for (int i = 0; i < max_pts_per_T; i++) {
                                std::uniform_int_distribution<int> dist(i, (int)pool.size() - 1);
                                std::swap(pool[i], pool[dist(rng)]);
                            }
                            pool.resize(max_pts_per_T);
                        }
                        for (auto& pt : pool)
                            fprintf(f, "%.8e\t%.8e\t%.8f\n", pt.q, pt.ql, T);
                    }
                    fclose(f);
                    printf("  Written %s\n", fname);
                }
            }
        }
    }

    // ================================================================
    // Equipartition Parameter EP(T) + Shannon Entropy + Gini
    // Merged: single ConfigStore pass, threaded over temperatures.
    // EP = N * Var_k[<I_k>_t] / [mean_k <I_k>_t]^2
    // S  = -sum_k p_k ln(p_k)    p_k = I_k / sum I_k
    // Gini from sorted intensities
    // ================================================================
    if (nrep >= 1) {
        printf("\n── Equipartition + Shannon/Gini (threads=%d) ───────\n", nthreads);

        std::vector<std::vector<double>> ep_s(nsamples), shan_s(nsamples), gini_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);
            ep_s[s].resize(ntemps, 0.0);
            shan_s[s].resize(ntemps, 0.0);
            gini_s[s].resize(ntemps, 0.0);

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;
                    int tidx = all_data[ref][ti].tidx;

                    // --- Compute <I_k> for EP ---
                    std::vector<double> mean_Ik(N, 0.0);
                    double sum_S = 0, sum_G = 0;
                    int count = 0;

                    for (int r = 0; r < nrep; r++) {
                        for (auto& entry : cstore.entries[tidx][r]) {
                            std::vector<double> Ik(N);
                            double Itot = 0;
                            for (int k = 0; k < N; k++) {
                                double re = entry.data[2*k], im = entry.data[2*k+1];
                                Ik[k] = re*re + im*im;
                                mean_Ik[k] += Ik[k];
                                Itot += Ik[k];
                            }
                            // Shannon entropy
                            if (Itot > 0) {
                                double S = 0;
                                for (int k = 0; k < N; k++) {
                                    double pk = Ik[k] / Itot;
                                    if (pk > 0) S -= pk * log(pk);
                                }
                                sum_S += S;
                                // Gini coefficient
                                std::sort(Ik.begin(), Ik.end());
                                double G_num = 0;
                                for (int k = 0; k < N; k++)
                                    G_num += (2.0 * (k + 1) - N - 1.0) * Ik[k];
                                sum_G += G_num / (N * Itot);
                            }
                            count++;
                        }
                    }
                    if (count == 0) continue;

                    // EP
                    for (int k = 0; k < N; k++) mean_Ik[k] /= count;
                    double mean_of_Ik = 0, var_of_Ik = 0;
                    for (int k = 0; k < N; k++) mean_of_Ik += mean_Ik[k];
                    mean_of_Ik /= N;
                    for (int k = 0; k < N; k++)
                        var_of_Ik += (mean_Ik[k] - mean_of_Ik) * (mean_Ik[k] - mean_of_Ik);
                    var_of_Ik /= N;
                    ep_s[s][ti] = (mean_of_Ik > 0)
                        ? N * var_of_Ik / (mean_of_Ik * mean_of_Ik) : 0.0;

                    // Shannon & Gini averages
                    shan_s[s][ti] = sum_S / count;
                    gini_s[s][ti] = sum_G / count;
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
            printf("  Sample S%d done\n", labels[s]);
        }

        // Write EP
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/ep.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Equipartition parameter EP(T) = N*Var_k[<I_k>]/<I_k>^2\n");
                fprintf(f, "# T\tEP\tEP_err\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    std::vector<double> vals(nsamples);
                    for (int s = 0; s < nsamples; s++) vals[s] = ep_s[s][ti];
                    double fm = 0;
                    for (auto v : vals) fm += v;
                    fm /= nsamples;
                    double jk = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lv = 0;
                        for (int s = 0; s < nsamples; s++) if (s != j) lv += vals[s];
                        lv /= (nsamples - 1);
                        jk += (lv - fm) * (lv - fm);
                    }
                    jk = sqrt((nsamples - 1.0) / nsamples * jk);
                    fprintf(f, "%.8f\t%.8e\t%.8e\n", temps[ti], fm, jk);
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }
        // Write Shannon/Gini
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/entropy_gini.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Shannon entropy S and Gini coefficient G vs T\n");
                fprintf(f, "# T\tS\tS_err\tGini\tGini_err\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    std::vector<double> vS(nsamples), vG(nsamples);
                    for (int s = 0; s < nsamples; s++) {
                        vS[s] = shan_s[s][ti]; vG[s] = gini_s[s][ti];
                    }
                    double fS = 0, fG = 0;
                    for (int s = 0; s < nsamples; s++) { fS += vS[s]; fG += vG[s]; }
                    fS /= nsamples; fG /= nsamples;
                    double jS = 0, jG = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lS = 0, lG = 0;
                        for (int s = 0; s < nsamples; s++) if (s != j) { lS += vS[s]; lG += vG[s]; }
                        lS /= (nsamples - 1); lG /= (nsamples - 1);
                        jS += (lS - fS) * (lS - fS);
                        jG += (lG - fG) * (lG - fG);
                    }
                    jS = sqrt((nsamples - 1.0) / nsamples * jS);
                    jG = sqrt((nsamples - 1.0) / nsamples * jG);
                    fprintf(f, "%.8f\t%.8e\t%.8e\t%.8e\t%.8e\n", temps[ti], fS, jS, fG, jG);
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }
    }

    // ================================================================
    // Non-Self-Averaging parameter for energy: A_E = N*Var_J[<E/N>] / <E/N>^2
    // ================================================================
    {
        printf("\n── Non-self-averaging A_E ──────────────────────────\n");

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/nsa_energy.dat", outdir);
        FILE* f = fopen(fname, "w");
        if (f) {
            fprintf(f, "# Non-self-averaging parameter A_E = N*Var_J[<E/N>]/<E/N>^2\n");
            fprintf(f, "# T\tA_E\tA_E_err\n");

            for (int ti = 0; ti < ntemps; ti++) {
                double T = all_data[ref][ti].T;
                // Per-sample <E/N> (averaged over replicas and second-half measurements)
                std::vector<double> eJ(nsamples, 0.0);
                for (int s = 0; s < nsamples; s++) {
                    if (ti >= (int)all_data[s].size()) continue;
                    double sum_e = 0;
                    int cnt = 0;
                    for (int r = 0; r < nrep; r++) {
                        SampleObs obs = compute_obs(all_data[s][ti], r);
                        sum_e += obs.e_mean;
                        cnt++;
                    }
                    eJ[s] = (cnt > 0) ? sum_e / cnt : 0.0;
                }
                double mean_eJ = 0;
                for (auto v : eJ) mean_eJ += v;
                mean_eJ /= nsamples;
                double var_eJ = 0;
                for (auto v : eJ) var_eJ += (v - mean_eJ) * (v - mean_eJ);
                var_eJ /= nsamples;

                double A_E = (mean_eJ != 0)
                    ? N * var_eJ / (mean_eJ * mean_eJ) : 0.0;

                // Jackknife
                double jk = 0;
                for (int j = 0; j < nsamples; j++) {
                    double lm = 0, lv = 0;
                    for (int s = 0; s < nsamples; s++) if (s != j) lm += eJ[s];
                    lm /= (nsamples - 1);
                    for (int s = 0; s < nsamples; s++) if (s != j) lv += (eJ[s] - lm) * (eJ[s] - lm);
                    lv /= (nsamples - 1);
                    double lA = (lm != 0) ? N * lv / (lm * lm) : 0.0;
                    jk += (lA - A_E) * (lA - A_E);
                }
                jk = sqrt((nsamples - 1.0) / nsamples * jk);
                fprintf(f, "%.8f\t%.8e\t%.8e\n", T, A_E, jk);
            }
            fclose(f);
            printf("  Written %s\n", fname);
        }
    }

    // ================================================================
    // Mode-frequency correlation: <I_k> vs omega_k
    // ================================================================
    if (nrep >= 1) {
        printf("\n── Mode-frequency correlation ──────────────────────\n");

        // Check if frequencies exist
        std::vector<double> omega_ref;
        {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[0]);
            omega_ref = read_frequencies(sdir, N);
        }
        if (!omega_ref.empty()) {
            // Compute <I_k> per temperature per mode (averaged over samples)
            // Save for coldest and hottest temperatures
            int t_cold = ntemps - 1, t_hot = 0;
            int tsel[] = { t_hot, t_cold };
            const char* tnames[] = { "hot", "cold" };

            for (int tt = 0; tt < 2; tt++) {
                int ti = tsel[tt];
                int tidx = all_data[ref][ti].tidx;
                std::vector<double> meanIk(N, 0.0);
                int total_count = 0;

                for (int s = 0; s < nsamples; s++) {
                    char sdir[256];
                    snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                             prefix, N, NT, nrep, labels[s]);
                    ConfigStore cstore;
                    cstore.load(sdir, N, NT, nrep);
                    for (int r = 0; r < nrep; r++) {
                        for (auto& entry : cstore.entries[tidx][r]) {
                            for (int k = 0; k < N; k++) {
                                double re = entry.data[2*k], im = entry.data[2*k+1];
                                meanIk[k] += re*re + im*im;
                            }
                            total_count++;
                        }
                    }
                }
                if (total_count > 0)
                    for (int k = 0; k < N; k++) meanIk[k] /= total_count;

                char fname[512];
                snprintf(fname, sizeof(fname), "%s/mode_freq_corr_%s.dat", outdir, tnames[tt]);
                FILE* f = fopen(fname, "w");
                if (f) {
                    fprintf(f, "# Mode-frequency correlation at T=%s (T=%.6f)\n", tnames[tt], temps[ti]);
                    fprintf(f, "# omega_k\t<I_k>\n");
                    for (int k = 0; k < N; k++)
                        fprintf(f, "%.8e\t%.8e\n", omega_ref[k], meanIk[k]);
                    fclose(f);
                    printf("  Written %s\n", fname);
                }
            }
        } else {
            printf("  No frequencies.txt found — skipping\n");
        }
    }

    // ================================================================
    // Franz-Parisi potential V(q) = -T * ln P(q)
    // ================================================================
    if (nrep > 1) {
        printf("\n── Franz-Parisi potential V(q) ─────────────────────\n");

        // Read P(q) data from pq_*.dat files (already written above)
        for (int ti = 0; ti < ntemps; ti++) {
            char pqfile[512];
            snprintf(pqfile, sizeof(pqfile), "%s/pq_T%.4f.dat", outdir, temps[ti]);
            FILE* fin = fopen(pqfile, "r");
            if (!fin) continue;

            char vqfile[512];
            snprintf(vqfile, sizeof(vqfile), "%s/vq_T%.4f.dat", outdir, temps[ti]);
            FILE* fout = fopen(vqfile, "w");
            if (!fout) { fclose(fin); continue; }

            fprintf(fout, "# Franz-Parisi potential V(q) = -T*ln(P(q)) at T=%.6f\n", temps[ti]);
            fprintf(fout, "# q\tV(q)\n");

            char line[1024];
            while (fgets(line, sizeof(line), fin)) {
                if (line[0] == '#') continue;
                double q, pq, pq_err;
                if (sscanf(line, "%lf\t%lf\t%lf", &q, &pq, &pq_err) >= 2 && pq > 0) {
                    double Vq = -temps[ti] * log(pq);
                    fprintf(fout, "%.8e\t%.8e\n", q, Vq);
                }
            }
            fclose(fin);
            fclose(fout);
        }
        printf("  Written V(q) files\n");
    }

    // ================================================================
    // Autocorrelation C(tau) from configs
    // C(tau) = (1/N) sum_k Re(a*_k(t) a_k(t+tau)) averaged over t
    // ================================================================
    if (nrep >= 1) {
        printf("\n── Autocorrelation C(tau) (threads=%d) ─────────────\n", nthreads);

        // autocorr_s[s][ti][lag]
        int max_lag = 50;
        std::vector<std::vector<std::vector<double>>> autocorr_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);
            autocorr_s[s].resize(ntemps);

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;
                    int tidx = all_data[ref][ti].tidx;
                    int nsnap = 0;
                    for (int r = 0; r < nrep; r++)
                        nsnap = std::max(nsnap, (int)cstore.entries[tidx][r].size());
                    int nlags = std::min(max_lag, nsnap / 2);
                    if (nlags < 1) nlags = 1;
                    autocorr_s[s][ti].assign(nlags, 0.0);

                    for (int r = 0; r < nrep; r++) {
                        int ns = (int)cstore.entries[tidx][r].size();
                        if (ns < 2) continue;
                        for (int lag = 0; lag < nlags; lag++) {
                            double c_sum = 0;
                            int c_count = 0;
                            for (int t0 = 0; t0 + lag < ns; t0++) {
                                double dot = 0, n0 = 0, n1 = 0;
                                auto& d0 = cstore.entries[tidx][r][t0].data;
                                auto& d1 = cstore.entries[tidx][r][t0 + lag].data;
                                for (int k = 0; k < N; k++) {
                                    double re0 = d0[2*k], im0 = d0[2*k+1];
                                    double re1 = d1[2*k], im1 = d1[2*k+1];
                                    dot += re0*re1 + im0*im1;
                                    n0 += re0*re0 + im0*im0;
                                    n1 += re1*re1 + im1*im1;
                                }
                                double norm = sqrt(n0 * n1);
                                c_sum += (norm > 0) ? dot / norm : 0.0;
                                c_count++;
                            }
                            if (c_count > 0)
                                autocorr_s[s][ti][lag] += c_sum / c_count;
                        }
                    }
                    for (int lag = 0; lag < nlags; lag++)
                        autocorr_s[s][ti][lag] /= nrep;
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
            printf("  Sample S%d done\n", labels[s]);
        }

        // Write for a few temperatures
        int tsel[] = { 0, ntemps / 4, ntemps / 2, 3 * ntemps / 4, ntemps - 1 };
        int ntsel = 5;
        for (int tt = 0; tt < ntsel; tt++) {
            int ti = tsel[tt];
            if (ti < 0 || ti >= ntemps) continue;
            int nlags = 0;
            for (int s = 0; s < nsamples; s++)
                nlags = std::max(nlags, (int)autocorr_s[s][ti].size());
            if (nlags < 1) continue;

            char fname[512];
            snprintf(fname, sizeof(fname), "%s/autocorr_T%.4f.dat", outdir, temps[ti]);
            FILE* f = fopen(fname, "w");
            if (!f) continue;
            fprintf(f, "# Autocorrelation C(tau) at T=%.6f\n", temps[ti]);
            fprintf(f, "# lag\tC\tC_err\n");

            for (int lag = 0; lag < nlags; lag++) {
                std::vector<double> vals(nsamples, 0.0);
                for (int s = 0; s < nsamples; s++)
                    if (lag < (int)autocorr_s[s][ti].size())
                        vals[s] = autocorr_s[s][ti][lag];
                double fm = 0;
                for (auto v : vals) fm += v;
                fm /= nsamples;
                double jk = 0;
                for (int j = 0; j < nsamples; j++) {
                    double lv = 0;
                    for (int s = 0; s < nsamples; s++) if (s != j) lv += vals[s];
                    lv /= (nsamples - 1);
                    jk += (lv - fm) * (lv - fm);
                }
                jk = sqrt((nsamples - 1.0) / nsamples * jk);
                fprintf(f, "%d\t%.8e\t%.8e\n", lag, fm, jk);
            }
            fclose(f);
            printf("  Written %s\n", fname);
        }
    }

    // ================================================================
    // Decorrelation time from energy autocorrelation
    // tau_E defined by C_E(tau_E) = 1/e
    // ================================================================
    {
        printf("\n── Decorrelation time (energy) ─────────────────────\n");

        // tau_s[s][ti] = decorrelation time for sample s, temperature ti
        std::vector<std::vector<double>> tau_s(nsamples);

        for (int s = 0; s < nsamples; s++) {
            tau_s[s].resize(ntemps, 0.0);
            for (int ti = 0; ti < ntemps; ti++) {
                if (ti >= (int)all_data[s].size()) continue;
                auto& tb = all_data[s][ti];
                int M = (int)tb.energy.size();
                int start = M / 2;
                if (start >= M) continue;
                int n = M - start;
                if (n < 4) continue;

                // Compute energy time series (replica-averaged)
                std::vector<double> e_ts(n);
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int r = 0; r < nrep; r++)
                        sum += tb.energy[start + i][r];
                    e_ts[i] = sum / nrep;
                }
                double e_mean = 0, e2_mean = 0;
                for (double v : e_ts) { e_mean += v; e2_mean += v * v; }
                e_mean /= n; e2_mean /= n;
                double var_e = e2_mean - e_mean * e_mean;
                if (var_e <= 0) continue;

                // Compute C_E(lag) and find where it drops below 1/e
                double tau = 0;
                for (int lag = 1; lag < n / 2; lag++) {
                    double c = 0;
                    int cnt = 0;
                    for (int i = 0; i + lag < n; i++) {
                        c += (e_ts[i] - e_mean) * (e_ts[i + lag] - e_mean);
                        cnt++;
                    }
                    c /= cnt;
                    double rho = c / var_e;
                    if (rho < 1.0 / M_E) { // 1/e
                        // Linear interpolation
                        double c_prev = var_e;
                        if (lag > 1) {
                            double cp = 0;
                            int cnt2 = 0;
                            for (int i = 0; i + lag - 1 < n; i++) {
                                cp += (e_ts[i] - e_mean) * (e_ts[i + lag - 1] - e_mean);
                                cnt2++;
                            }
                            c_prev = cp / cnt2;
                        }
                        double rho_prev = c_prev / var_e;
                        if (rho_prev > 1.0 / M_E)
                            tau = (lag - 1) + (rho_prev - 1.0/M_E) / (rho_prev - rho);
                        else
                            tau = lag;
                        break;
                    }
                    if (lag == n / 2 - 1) tau = lag; // never crossed 1/e
                }
                tau_s[s][ti] = tau;
            }
        }

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/decorrelation_time.dat", outdir);
        FILE* f = fopen(fname, "w");
        if (f) {
            fprintf(f, "# Energy decorrelation time tau_E (in save_freq units)\n");
            fprintf(f, "# T\ttau_E\ttau_E_err\n");
            for (int ti = 0; ti < ntemps; ti++) {
                std::vector<double> vals(nsamples);
                for (int s = 0; s < nsamples; s++) vals[s] = tau_s[s][ti];
                double fm = 0;
                for (auto v : vals) fm += v;
                fm /= nsamples;
                double jk = 0;
                for (int j = 0; j < nsamples; j++) {
                    double lv = 0;
                    for (int s = 0; s < nsamples; s++) if (s != j) lv += vals[s];
                    lv /= (nsamples - 1);
                    jk += (lv - fm) * (lv - fm);
                }
                jk = sqrt((nsamples - 1.0) / nsamples * jk);
                fprintf(f, "%.8f\t%.8e\t%.8e\n", temps[ti], fm, jk);
            }
            fclose(f);
            printf("  Written %s\n", fname);
        }
    }

    // ================================================================
    // Mode-mode correlation matrix C_ij eigenvalue spectrum
    // C_ij = <delta_I_i * delta_I_j> / sqrt(<delta_I_i^2> * <delta_I_j^2>)
    // ================================================================
    if (nrep >= 1 && N <= 200) {
        printf("\n── Mode-mode correlation eigenvalues (threads=%d) ──\n", nthreads);

        int tsel_mm[] = { 0, ntemps / 2, ntemps - 1 };
        int ntsel_mm = 3;
        const char* tnames_mm[] = { "hot", "mid", "cold" };

        // Per-temperature accumulators
        std::vector<std::vector<double>> all_meanI(ntsel_mm, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> all_Cij(ntsel_mm, std::vector<double>(N * N, 0.0));
        std::vector<int> all_count(ntsel_mm, 0);
        std::vector<std::mutex> mtx_mm(ntsel_mm);

        // Load ConfigStore once per sample, accumulate for all 3 temperatures in parallel
        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            std::atomic<int> tt_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int tt = tt_next.fetch_add(1);
                    if (tt >= ntsel_mm) break;
                    int ti = tsel_mm[tt];
                    if (ti < 0 || ti >= ntemps) continue;
                    int tidx = all_data[ref][ti].tidx;

                    std::vector<double> local_meanI(N, 0.0);
                    std::vector<double> local_Cij(N * N, 0.0);
                    int local_count = 0;

                    for (int r = 0; r < nrep; r++) {
                        for (auto& entry : cstore.entries[tidx][r]) {
                            std::vector<double> Ik(N);
                            for (int k = 0; k < N; k++) {
                                double re = entry.data[2*k], im = entry.data[2*k+1];
                                Ik[k] = re*re + im*im;
                            }
                            for (int i = 0; i < N; i++) {
                                local_meanI[i] += Ik[i];
                                for (int j = i; j < N; j++)
                                    local_Cij[i * N + j] += Ik[i] * Ik[j];
                            }
                            local_count++;
                        }
                    }

                    std::lock_guard<std::mutex> lk(mtx_mm[tt]);
                    for (int i = 0; i < N; i++) all_meanI[tt][i] += local_meanI[i];
                    for (int i = 0; i < N * N; i++) all_Cij[tt][i] += local_Cij[i];
                    all_count[tt] += local_count;
                }
            };

            int nt = std::min(nthreads, ntsel_mm);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }

        // Finalize and write for each temperature
        for (int tt = 0; tt < ntsel_mm; tt++) {
            int ti = tsel_mm[tt];
            if (ti < 0 || ti >= ntemps) continue;
            if (all_count[tt] == 0) continue;

            auto& meanI = all_meanI[tt];
            auto& Cij = all_Cij[tt];
            int tc = all_count[tt];

            for (int i = 0; i < N; i++) meanI[i] /= tc;
            for (int i = 0; i < N; i++)
                for (int j = i; j < N; j++) {
                    Cij[i * N + j] = Cij[i * N + j] / tc - meanI[i] * meanI[j];
                    Cij[j * N + i] = Cij[i * N + j];
                }
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    double di = Cij[i * N + i], dj = Cij[j * N + j];
                    if (di > 0 && dj > 0 && i != j)
                        Cij[i * N + j] /= sqrt(di * dj);
                }
            for (int i = 0; i < N; i++) Cij[i * N + i] = 1.0;

            // Jacobi eigenvalue algorithm
            std::vector<double> A(Cij);
            int n = N;
            for (int iter = 0; iter < 100 * n; iter++) {
                int p = 0, q = 1;
                double max_val = 0;
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                        if (fabs(A[i*n+j]) > max_val) { max_val = fabs(A[i*n+j]); p = i; q = j; }
                if (max_val < 1e-12) break;

                double app = A[p*n+p], aqq = A[q*n+q], apq = A[p*n+q];
                double theta = 0.5 * atan2(2*apq, app - aqq);
                double c = cos(theta), s = sin(theta);

                for (int i = 0; i < n; i++) {
                    if (i == p || i == q) continue;
                    double aip = A[i*n+p], aiq = A[i*n+q];
                    A[i*n+p] = A[p*n+i] = c*aip + s*aiq;
                    A[i*n+q] = A[q*n+i] = -s*aip + c*aiq;
                }
                A[p*n+p] = c*c*app + 2*s*c*apq + s*s*aqq;
                A[q*n+q] = s*s*app - 2*s*c*apq + c*c*aqq;
                A[p*n+q] = A[q*n+p] = 0;
            }

            std::vector<double> evals(n);
            for (int i = 0; i < n; i++) evals[i] = A[i*n+i];
            std::sort(evals.begin(), evals.end(), std::greater<double>());

            char fname[512];
            snprintf(fname, sizeof(fname), "%s/eigvals_%s.dat", outdir, tnames_mm[tt]);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Eigenvalues of mode-mode correlation matrix at T=%s (%.6f)\n",
                        tnames_mm[tt], temps[ti]);
                fprintf(f, "# index\tlambda\tlambda/N\n");
                for (int i = 0; i < n; i++)
                    fprintf(f, "%d\t%.8e\t%.8e\n", i, evals[i], evals[i] / n);
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }
    }

    // ================================================================
    // Ultrametricity test (requires nrep >= 3)
    // For triplets (a,b,c), compute overlaps q_ab, q_bc, q_ac
    // Sort: q1 <= q2 <= q3. Ultrametric => q1 ~ q2.
    //
    // STREAMING: 2D histogram + running mean instead of storing all
    // raw points. Memory: O(nbins² × ntemps) instead of O(ntriplets).
    // Snapshot subsampling to cap O(nrep³ × nsnap) computation.
    // ================================================================
    if (nrep >= 3) {
        printf("\n── Ultrametricity test (nrep=%d, threads=%d) ───────\n", nrep, nthreads);

        const int nbins_um = 100;
        const double d12_max = 2.0;           // |q1-q2| ∈ [0, 2]
        const double q3_min = -1.0, q3_max = 1.0;
        const double d12_dbin = d12_max / nbins_um;
        const double q3_dbin  = (q3_max - q3_min) / nbins_um;
        const int max_iters_um = 256;         // cap snapshots per (ti,s)

        // 2D histogram accumulated over all samples — ~3 MB total
        std::vector<std::vector<double>> hist2d(ntemps,
            std::vector<double>(nbins_um * nbins_um, 0.0));
        // Per-sample mean |q1-q2| for jackknife
        std::vector<std::vector<double>> mean_d12_s(nsamples,
            std::vector<double>(ntemps, 0.0));

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;
                    int tidx = all_data[ref][ti].tidx;

                    // Find common iterations across all replicas
                    auto iters0 = cstore.get_iters(tidx, 0);
                    std::set<int> common(iters0.begin(), iters0.end());
                    for (int r = 1; r < nrep; r++) {
                        auto it_r = cstore.get_iters(tidx, r);
                        std::set<int> s_r(it_r.begin(), it_r.end());
                        std::set<int> tmp;
                        std::set_intersection(common.begin(), common.end(),
                                              s_r.begin(), s_r.end(),
                                              std::inserter(tmp, tmp.begin()));
                        common = tmp;
                    }
                    std::vector<int> c_iters(common.begin(), common.end());
                    if (c_iters.empty()) continue;

                    // Subsample snapshots if too many
                    if ((int)c_iters.size() > max_iters_um) {
                        std::mt19937 rng(42 + s * ntemps + ti);
                        std::shuffle(c_iters.begin(), c_iters.end(), rng);
                        c_iters.resize(max_iters_um);
                    }

                    // Thread-local histogram + streaming mean
                    std::vector<double> local_hist(nbins_um * nbins_um, 0.0);
                    double sum_d12 = 0;
                    int count = 0;

                    for (int it : c_iters) {
                        for (int ra = 0; ra < nrep; ra++)
                            for (int rb = ra + 1; rb < nrep; rb++)
                                for (int rc = rb + 1; rc < nrep; rc++) {
                                    auto* ea = cstore.find_entry(tidx, ra, it);
                                    auto* eb = cstore.find_entry(tidx, rb, it);
                                    auto* ec = cstore.find_entry(tidx, rc, it);
                                    if (!ea || !eb || !ec) continue;

                                    auto overlap = [&](const std::vector<double>& d1,
                                                       const std::vector<double>& d2) {
                                        double num = 0, n1 = 0, n2 = 0;
                                        for (int k = 0; k < N; k++) {
                                            double r1 = d1[2*k], i1 = d1[2*k+1];
                                            double r2 = d2[2*k], i2 = d2[2*k+1];
                                            num += r1*r2 + i1*i2;
                                            n1 += r1*r1 + i1*i1;
                                            n2 += r2*r2 + i2*i2;
                                        }
                                        double norm = sqrt(n1 * n2);
                                        return (norm > 0) ? num / norm : 0.0;
                                    };

                                    double q[3] = {
                                        overlap(ea->data, eb->data),
                                        overlap(eb->data, ec->data),
                                        overlap(ea->data, ec->data)
                                    };
                                    std::sort(q, q + 3);
                                    double d12 = fabs(q[0] - q[1]);
                                    double q3  = q[2];

                                    int b_d12 = (int)(d12 / d12_dbin);
                                    int b_q3  = (int)((q3 - q3_min) / q3_dbin);
                                    if (b_d12 >= nbins_um) b_d12 = nbins_um - 1;
                                    if (b_q3 < 0) b_q3 = 0;
                                    if (b_q3 >= nbins_um) b_q3 = nbins_um - 1;
                                    local_hist[b_d12 * nbins_um + b_q3]++;
                                    sum_d12 += d12;
                                    count++;
                                }
                    }

                    // Merge — no lock: unique ti per thread, join between samples
                    for (int i = 0; i < nbins_um * nbins_um; i++)
                        hist2d[ti][i] += local_hist[i];
                    mean_d12_s[s][ti] = (count > 0) ? sum_d12 / count : 0.0;
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
            printf("  Sample S%d done\n", labels[s]);
        }

        // Write 2D histogram per temperature (heatmap-ready)
        for (int ti = 0; ti < ntemps; ti++) {
            double total = 0;
            for (auto h : hist2d[ti]) total += h;
            if (total == 0) continue;

            char fname[512];
            snprintf(fname, sizeof(fname), "%s/ultrametric_T%.4f.dat", outdir, temps[ti]);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Ultrametricity 2D histogram at T=%.6f\n", temps[ti]);
                fprintf(f, "# |q1-q2|\tq3\tdensity\n");
                for (int bd = 0; bd < nbins_um; bd++)
                    for (int bq = 0; bq < nbins_um; bq++) {
                        double val = hist2d[ti][bd * nbins_um + bq];
                        if (val > 0)
                            fprintf(f, "%.8e\t%.8e\t%.8e\n",
                                    (bd + 0.5) * d12_dbin,
                                    q3_min + (bq + 0.5) * q3_dbin,
                                    val / (total * d12_dbin * q3_dbin));
                    }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }

        // Write mean |q1-q2| vs T with jackknife errors
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/ultrametric_mean.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Mean |q1-q2| vs T  (ultrametricity: 0 = perfectly ultrametric)\n");
                fprintf(f, "# T\t<|q1-q2|>\terr\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    std::vector<double> vals(nsamples);
                    for (int ss = 0; ss < nsamples; ss++) vals[ss] = mean_d12_s[ss][ti];
                    double fm = 0;
                    for (auto v : vals) fm += v;
                    fm /= nsamples;
                    double jk = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double lv = 0;
                        for (int ss = 0; ss < nsamples; ss++)
                            if (ss != j) lv += vals[ss];
                        lv /= (nsamples - 1);
                        jk += (lv - fm) * (lv - fm);
                    }
                    jk = sqrt((nsamples - 1.0) / nsamples * jk);
                    fprintf(f, "%.8f\t%.8e\t%.8e\n", temps[ti], fm, jk);
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        }
    }

    // ================================================================
    // Inter-sample overlap P_inter(q)
    // Overlap between configs from DIFFERENT disorder samples
    // ================================================================
    if (nsamples >= 2 && nrep >= 1) {
        printf("\n── Inter-sample overlap (threads=%d) ───────────────\n", nthreads);

        double qmin = -1.0, qmax = 1.0;
        double dq = (qmax - qmin) / nbins;

        // Load configs for all samples
        std::vector<ConfigStore> all_configs(nsamples);
        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            all_configs[s].load(sdir, N, NT, nrep);
        }

        // Per-temperature histograms (each thread writes its own ti)
        std::vector<std::vector<double>> hists(ntemps);
        std::vector<int> total_pairs_per_ti(ntemps, 0);
        for (int ti = 0; ti < ntemps; ti++) hists[ti].assign(nbins, 0.0);

        std::atomic<int> ti_next(0);
        auto worker = [&](int /*id*/) {
            while (true) {
                int ti = ti_next.fetch_add(1);
                if (ti >= ntemps) break;
                int tidx = all_data[ref][ti].tidx;
                auto& hist = hists[ti];
                int& tp = total_pairs_per_ti[ti];

                for (int sa = 0; sa < nsamples; sa++)
                    for (int sb = sa + 1; sb < nsamples; sb++) {
                        for (int ra = 0; ra < nrep; ra++) {
                            auto& ea = all_configs[sa].entries[tidx][ra];
                            if (ea.empty()) continue;
                            auto& da = ea.back().data;
                            for (int rb = 0; rb < nrep; rb++) {
                                auto& eb = all_configs[sb].entries[tidx][rb];
                                if (eb.empty()) continue;
                                auto& db = eb.back().data;

                                double num = 0, n1 = 0, n2 = 0;
                                for (int k = 0; k < N; k++) {
                                    double r1 = da[2*k], i1 = da[2*k+1];
                                    double r2 = db[2*k], i2 = db[2*k+1];
                                    num += r1*r2 + i1*i2;
                                    n1 += r1*r1 + i1*i1;
                                    n2 += r2*r2 + i2*i2;
                                }
                                double norm = sqrt(n1 * n2);
                                double q = (norm > 0) ? num / norm : 0.0;
                                int bin = (int)((q - qmin) / dq);
                                if (bin < 0) bin = 0;
                                if (bin >= nbins) bin = nbins - 1;
                                hist[bin]++;
                                tp++;
                            }
                        }
                    }
            }
        };

        int nt = std::min(nthreads, ntemps);
        std::vector<std::thread> threads;
        for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
        for (auto& th : threads) th.join();

        // Write files
        for (int ti = 0; ti < ntemps; ti++) {
            if (total_pairs_per_ti[ti] == 0) continue;
            for (auto& h : hists[ti]) h /= (total_pairs_per_ti[ti] * dq);

            char fname[512];
            snprintf(fname, sizeof(fname), "%s/pq_inter_T%.4f.dat", outdir, temps[ti]);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Inter-sample overlap P_inter(q) at T=%.6f\n", temps[ti]);
                fprintf(f, "# q\tP_inter(q)\n");
                for (int b = 0; b < nbins; b++)
                    fprintf(f, "%.8e\t%.8e\n", qmin + (b + 0.5) * dq, hists[ti][b]);
                fclose(f);
            }
        }
        printf("  Written inter-sample overlap files\n");
    }

    // ================================================================
    // E2/E4 split vs T (from energy_split.txt)
    // ================================================================
    {
        printf("\n── E2/E4 energy split ──────────────────────────────\n");

        // Read energy_split.txt for each sample
        // Format: sweep Tidx T  E2_0/N E4_0/N  E2_1/N E4_1/N ...
        // e2_s[s][ti], e4_s[s][ti] = time-averaged split energies
        std::vector<std::vector<double>> e2_s(nsamples), e4_s(nsamples);
        bool have_split = false;

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            char esfile[512];
            snprintf(esfile, sizeof(esfile), "%s/energy_split.txt", sdir);
            FILE* f = fopen(esfile, "r");
            if (!f) continue;
            have_split = true;

            // Accumulate per-temperature averages
            std::vector<double> sum_e2(ntemps, 0.0), sum_e4(ntemps, 0.0);
            std::vector<int> count(ntemps, 0);

            char line[4096];
            while (fgets(line, sizeof(line), f)) {
                if (line[0] == '#') continue;
                int sweep, tidx_val;
                double T;
                int n_parsed = sscanf(line, "%d\t%d\t%lf", &sweep, &tidx_val, &T);
                if (n_parsed < 3) continue;
                // Only use second half
                if (sweep < all_data[ref][0].sweeps.back() / 2) continue;

                // Parse replica E2/E4 pairs
                char* ptr = line;
                for (int skip = 0; skip < 3; skip++) { // skip sweep, Tidx, T
                    while (*ptr && *ptr != '\t') ptr++;
                    if (*ptr) ptr++;
                }
                double rep_e2_sum = 0, rep_e4_sum = 0;
                int nrep_read = 0;
                for (int r = 0; r < nrep; r++) {
                    double e2v, e4v;
                    if (sscanf(ptr, "%lf\t%lf", &e2v, &e4v) == 2) {
                        rep_e2_sum += e2v;
                        rep_e4_sum += e4v;
                        nrep_read++;
                    }
                    // Advance past two tab-separated values
                    for (int skip = 0; skip < 2; skip++) {
                        while (*ptr && *ptr != '\t') ptr++;
                        if (*ptr) ptr++;
                    }
                }

                // Find temperature index
                int ti_match = -1;
                for (int ti = 0; ti < ntemps; ti++) {
                    if (fabs(temps[ti] - T) < 1e-6) { ti_match = ti; break; }
                }
                if (ti_match >= 0 && nrep_read > 0) {
                    sum_e2[ti_match] += rep_e2_sum / nrep_read;
                    sum_e4[ti_match] += rep_e4_sum / nrep_read;
                    count[ti_match]++;
                }
            }
            fclose(f);

            e2_s[s].resize(ntemps, 0.0);
            e4_s[s].resize(ntemps, 0.0);
            for (int ti = 0; ti < ntemps; ti++) {
                if (count[ti] > 0) {
                    e2_s[s][ti] = sum_e2[ti] / count[ti];
                    e4_s[s][ti] = sum_e4[ti] / count[ti];
                }
            }
        }

        if (have_split) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/energy_split.dat", outdir);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# E2/N and E4/N vs T (split energy contributions)\n");
                fprintf(f, "# T\tE2/N\tE2_err\tE4/N\tE4_err\n");
                for (int ti = 0; ti < ntemps; ti++) {
                    std::vector<double> v2(nsamples), v4(nsamples);
                    for (int s = 0; s < nsamples; s++) {
                        v2[s] = (s < (int)e2_s.size() && ti < (int)e2_s[s].size()) ? e2_s[s][ti] : 0;
                        v4[s] = (s < (int)e4_s.size() && ti < (int)e4_s[s].size()) ? e4_s[s][ti] : 0;
                    }
                    double f2 = 0, f4 = 0;
                    for (int s = 0; s < nsamples; s++) { f2 += v2[s]; f4 += v4[s]; }
                    f2 /= nsamples; f4 /= nsamples;
                    double j2 = 0, j4 = 0;
                    for (int j = 0; j < nsamples; j++) {
                        double l2 = 0, l4 = 0;
                        for (int s = 0; s < nsamples; s++) if (s != j) { l2 += v2[s]; l4 += v4[s]; }
                        l2 /= (nsamples - 1); l4 /= (nsamples - 1);
                        j2 += (l2 - f2) * (l2 - f2);
                        j4 += (l4 - f4) * (l4 - f4);
                    }
                    j2 = sqrt((nsamples - 1.0) / nsamples * j2);
                    j4 = sqrt((nsamples - 1.0) / nsamples * j4);
                    fprintf(f, "%.8f\t%.8e\t%.8e\t%.8e\t%.8e\n", temps[ti], f2, j2, f4, j4);
                }
                fclose(f);
                printf("  Written %s\n", fname);
            }
        } else {
            printf("  No energy_split.txt found — run simulation first\n");
        }
    }

    // ================================================================
    // FSS: Cv peak extraction (Tc estimate)
    // ================================================================
    {
        printf("\n── FSS: Cv peak ────────────────────────────────────\n");

        // Compute Cv per temperature (replica-averaged, sample-averaged)
        std::vector<double> cv_mean(ntemps, 0.0);
        for (int ti = 0; ti < ntemps; ti++) {
            double T = temps[ti];
            double sum_cv = 0;
            for (int s = 0; s < nsamples; s++) {
                if (ti >= (int)all_data[s].size()) continue;
                double cv_s = 0;
                for (int r = 0; r < nrep; r++) {
                    SampleObs obs = compute_obs(all_data[s][ti], r);
                    cv_s += N * (obs.e2_mean - obs.e_mean * obs.e_mean) / (T * T);
                }
                sum_cv += cv_s / nrep;
            }
            cv_mean[ti] = sum_cv / nsamples;
        }

        // Find peak
        int i_peak = 0;
        for (int ti = 1; ti < ntemps; ti++)
            if (cv_mean[ti] > cv_mean[i_peak]) i_peak = ti;

        // Parabolic interpolation around peak
        double Tc = temps[i_peak], Cv_max = cv_mean[i_peak];
        if (i_peak > 0 && i_peak < ntemps - 1) {
            double T0 = temps[i_peak-1], T1 = temps[i_peak], T2 = temps[i_peak+1];
            double C0 = cv_mean[i_peak-1], C1 = cv_mean[i_peak], C2 = cv_mean[i_peak+1];
            double denom = (T0 - T1) * (T0 - T2) * (T1 - T2);
            if (fabs(denom) > 1e-30) {
                double a = (T2 * (C1 - C0) + T1 * (C0 - C2) + T0 * (C2 - C1)) / denom;
                double b = (T2*T2*(C0 - C1) + T1*T1*(C2 - C0) + T0*T0*(C1 - C2)) / denom;
                if (fabs(a) > 1e-30) {
                    Tc = -b / (2 * a);
                    Cv_max = (a * Tc + b) * Tc +
                             C0 - a * T0 * T0 - b * T0; // evaluate parabola
                }
            }
        }

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/fss_cv_peak.dat", outdir);
        FILE* f = fopen(fname, "w");
        if (f) {
            fprintf(f, "# FSS: Cv peak analysis\n");
            fprintf(f, "# N=%d\n", N);
            fprintf(f, "# Tc_estimate\tCv_max\tpeak_Tidx\n");
            fprintf(f, "%.8f\t%.8e\t%d\n", Tc, Cv_max, i_peak);
            fclose(f);
            printf("  Tc(N=%d) ~ %.6f (Cv_max=%.4f, peak at T[%d]=%.6f)\n",
                   N, Tc, Cv_max, i_peak, temps[i_peak]);
            printf("  Written %s\n", fname);
        }
    }

    // ================================================================
    // Kurtosis (Binder g4) crossing point
    // ================================================================
    if (nrep > 1) {
        printf("\n── Binder g4 crossing analysis ─────────────────────\n");

        // Read glass_obs.dat to get g4 vs T
        char gofile[512];
        snprintf(gofile, sizeof(gofile), "%s/glass_obs.dat", outdir);
        FILE* fin = fopen(gofile, "r");
        if (fin) {
            std::vector<double> Tg, g4v;
            char line[1024];
            while (fgets(line, sizeof(line), fin)) {
                if (line[0] == '#') continue;
                double T, chi, chi_err, g4_val, g4_err, A, A_err;
                if (sscanf(line, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
                           &T, &chi, &chi_err, &g4_val, &g4_err, &A, &A_err) >= 4) {
                    Tg.push_back(T);
                    g4v.push_back(g4_val);
                }
            }
            fclose(fin);

            // g4 crossing: find where g4 crosses specific reference value
            // For a p-spin glass, the crossing point at Tc is where
            // g4(N1, T) = g4(N2, T). For single-N, report g4 inflection.
            // Here we just find where g4 departs from its infinite-T value (1).
            // Look for maximum slope (steepest descent)
            int n_g = (int)Tg.size();
            if (n_g >= 3) {
                double max_slope = 0;
                int i_inflect = 0;
                for (int i = 1; i < n_g - 1; i++) {
                    double slope = fabs((g4v[i+1] - g4v[i-1]) / (Tg[i+1] - Tg[i-1]));
                    if (slope > max_slope) { max_slope = slope; i_inflect = i; }
                }

                char fname[512];
                snprintf(fname, sizeof(fname), "%s/g4_inflection.dat", outdir);
                FILE* f = fopen(fname, "w");
                if (f) {
                    fprintf(f, "# Binder g4 inflection analysis\n");
                    fprintf(f, "# N=%d\n", N);
                    fprintf(f, "# T_inflection\tg4_at_inflection\tmax_slope\n");
                    fprintf(f, "%.8f\t%.8e\t%.8e\n", Tg[i_inflect], g4v[i_inflect], max_slope);
                    fclose(f);
                    printf("  g4 inflection at T ~ %.6f (g4=%.4f, slope=%.4f)\n",
                           Tg[i_inflect], g4v[i_inflect], max_slope);
                    printf("  Written %s\n", fname);
                }
            }
        } else {
            printf("  glass_obs.dat not found — run overlap analysis first\n");
        }
    }

    // ================================================================
    // Multi-overlap scatter plots (q vs q_phi, IFO vs q, q_phi vs q_link)
    // ================================================================
    if (nrep >= 2) {
        printf("\n── Multi-overlap scatter (threads=%d) ──────────────\n", nthreads);

        struct MultiOvlPt { double q, q_phi, ifo; };
        std::vector<std::vector<MultiOvlPt>> pts_per_ti(ntemps);
        std::vector<std::mutex> mtx_pts(ntemps);

        for (int s = 0; s < nsamples; s++) {
            char sdir[256];
            snprintf(sdir, sizeof(sdir), "data/%s_N%d_NT%d_NR%d_S%d",
                     prefix, N, NT, nrep, labels[s]);
            ConfigStore cstore;
            cstore.load(sdir, N, NT, nrep);

            std::atomic<int> ti_next(0);
            auto worker = [&](int /*id*/) {
                while (true) {
                    int ti = ti_next.fetch_add(1);
                    if (ti >= ntemps) break;
                    int tidx = all_data[ref][ti].tidx;

                    auto iters0 = cstore.get_iters(tidx, 0);
                    std::set<int> common(iters0.begin(), iters0.end());
                    for (int r = 1; r < nrep; r++) {
                        auto it_r = cstore.get_iters(tidx, r);
                        std::set<int> s_r(it_r.begin(), it_r.end());
                        std::set<int> tmp;
                        std::set_intersection(common.begin(), common.end(),
                                              s_r.begin(), s_r.end(),
                                              std::inserter(tmp, tmp.begin()));
                        common = tmp;
                    }
                    std::vector<int> c_iters(common.begin(), common.end());
                    if (c_iters.empty()) continue;

                    std::vector<MultiOvlPt> local_pts;

                    for (int ra = 0; ra < nrep; ra++)
                        for (int rb = ra + 1; rb < nrep; rb++) {
                            std::vector<double> meanIa(N,0), meanIb(N,0);
                            int nit = (int)c_iters.size();
                            for (int it : c_iters) {
                                auto* ea = cstore.find_entry(tidx, ra, it);
                                auto* eb = cstore.find_entry(tidx, rb, it);
                                if (!ea || !eb) continue;
                                for (int k = 0; k < N; k++) {
                                    double ra_ = ea->data[2*k], ia_ = ea->data[2*k+1];
                                    double rb_ = eb->data[2*k], ib_ = eb->data[2*k+1];
                                    meanIa[k] += ra_*ra_ + ia_*ia_;
                                    meanIb[k] += rb_*rb_ + ib_*ib_;
                                }
                            }
                            for (int k = 0; k < N; k++) { meanIa[k] /= nit; meanIb[k] /= nit; }

                            for (int it : c_iters) {
                                auto* ea = cstore.find_entry(tidx, ra, it);
                                auto* eb = cstore.find_entry(tidx, rb, it);
                                if (!ea || !eb) continue;

                                double num_q = 0, n1q = 0, n2q = 0;
                                double num_qp = 0;
                                int cnt_qp = 0;
                                double num_ifo = 0, d1_ifo = 0, d2_ifo = 0;

                                for (int k = 0; k < N; k++) {
                                    double ra_ = ea->data[2*k], ia_ = ea->data[2*k+1];
                                    double rb_ = eb->data[2*k], ib_ = eb->data[2*k+1];
                                    double Ia = ra_*ra_ + ia_*ia_;
                                    double Ib = rb_*rb_ + ib_*ib_;

                                    num_q += ra_*rb_ + ia_*ib_;
                                    n1q += Ia; n2q += Ib;

                                    if (Ia > 1e-20 && Ib > 1e-20) {
                                        double ma = sqrt(Ia), mb = sqrt(Ib);
                                        num_qp += (ra_*rb_ + ia_*ib_) / (ma * mb);
                                        cnt_qp++;
                                    }

                                    double dIa = Ia - meanIa[k];
                                    double dIb = Ib - meanIb[k];
                                    num_ifo += dIa * dIb;
                                    d1_ifo += dIa * dIa;
                                    d2_ifo += dIb * dIb;
                                }

                                double nrm = sqrt(n1q * n2q);
                                double q_val = (nrm > 0) ? num_q / nrm : 0.0;
                                double qp_val = (cnt_qp > 0) ? num_qp / cnt_qp : 0.0;
                                double ifo_denom = sqrt(d1_ifo * d2_ifo);
                                double ifo_val = (ifo_denom > 0) ? num_ifo / ifo_denom : 0.0;

                                local_pts.push_back({q_val, qp_val, ifo_val});
                            }
                        }

                    if (!local_pts.empty()) {
                        std::lock_guard<std::mutex> lk(mtx_pts[ti]);
                        pts_per_ti[ti].insert(pts_per_ti[ti].end(),
                                              local_pts.begin(), local_pts.end());
                    }
                }
            };

            int nt = std::min(nthreads, ntemps);
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
            printf("  Sample S%d done\n", labels[s]);
        }

        // Write per-temperature files
        for (int ti = 0; ti < ntemps; ti++) {
            auto& pts = pts_per_ti[ti];
            if (pts.empty()) continue;

            // Downsample
            int max_pts = 5000;
            if ((int)pts.size() > max_pts) {
                std::mt19937 rng(42 + ti);
                for (int i = 0; i < max_pts; i++) {
                    std::uniform_int_distribution<int> dist(i, (int)pts.size() - 1);
                    std::swap(pts[i], pts[dist(rng)]);
                }
                pts.resize(max_pts);
            }

            char fname[512];
            snprintf(fname, sizeof(fname), "%s/multi_overlap_T%.4f.dat", outdir, temps[ti]);
            FILE* f = fopen(fname, "w");
            if (f) {
                fprintf(f, "# Multi-overlap scatter at T=%.6f\n", temps[ti]);
                fprintf(f, "# q_parisi\tq_phase\tIFO\n");
                for (auto& p : pts)
                    fprintf(f, "%.8e\t%.8e\t%.8e\n", p.q, p.q_phi, p.ifo);
                fclose(f);
            }
        }
        printf("  Written multi-overlap scatter files\n");
    }

    // ================================================================
    // Plotting (if --plot)
    // ================================================================
    if (do_plot) {
        using namespace sciplot;
        printf("\n── Generating plots ────────────────────────────────\n");

        char plotdir[384];
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
                    int nb = (int)blocks.size();
                    double Tmin = blocks.back()[0].T;
                    double Tmax = blocks.front()[0].T;
                    setup_colorbar_plot(plot, Tmin, Tmax, log_temp);
                    plot.xlabel("{/Helvetica-Oblique {/Symbol w}}");
                    plot.ylabel("{/Helvetica-Oblique I}_{/Helvetica-Oblique k}");
                    plot.legend().hide();
                    for (int bi = 0; bi < nb; bi++) {
                        auto& bl = blocks[bi];
                        int n = (int)bl.size();
                        std::vector<double> vx(n), vy(n);
                        for (int i = 0; i < n; i++) { vx[i] = bl[i].omega; vy[i] = bl[i].I; }
                        char pcol[64];
                        snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                        plot.drawCurve(vx, vy)
                            .lineColor(pcol)
                            .lineWidth(2)
                            .label("");
                    }
                    Figure fig = {{plot}};
                    Canvas canvas = {{fig}};
                    canvas.size(1000, 700);
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

                    // Energy plot (linear + log10)
                    for (int lm = 0; lm < 2; lm++) {
                        Plot2D plot;
                        setup_analysis_plot(plot);
                        plot.xlabel("{/Helvetica-Oblique T}");
                        plot.ylabel("{/Helvetica-Oblique E} / {/Helvetica-Oblique N}");
                        if (lm) plot.gnuplot("set logscale x 10");
                        plot.legend().atTopRight().fontSize(12);
                        plot.drawCurvesFilled(vT, vElo, vEhi)
                            .fillColor("#4393c3").fillIntensity(0.35).fillTransparent()
                            .lineColor("#4393c3").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vE)
                            .lineColor("#2166ac").lineWidth(2).label("E/N");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(900, 600);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/energy%s.png", plotdir, lm ? "_log" : "");
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
                    // MC Acceptance plot (linear + log10)
                    for (int lm = 0; lm < 2; lm++) {
                        Plot2D plot;
                        setup_analysis_plot(plot);
                        plot.xlabel("{/Helvetica-Oblique T}");
                        plot.ylabel("MC acceptance");
                        if (lm) plot.gnuplot("set logscale x 10");
                        plot.legend().atTopRight().fontSize(12);
                        plot.drawCurvesFilled(vT, vAlo, vAhi)
                            .fillColor("#66c2a5").fillIntensity(0.35).fillTransparent()
                            .lineColor("#66c2a5").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vA)
                            .lineColor("#1b7837").lineWidth(2).label("MC acceptance");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(900, 600);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/acceptance%s.png", plotdir, lm ? "_log" : "");
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
                    // Specific heat plot (linear + log10)
                    for (int lm = 0; lm < 2; lm++) {
                        Plot2D plot;
                        setup_analysis_plot(plot);
                        plot.xlabel("{/Helvetica-Oblique T}");
                        plot.ylabel("{/Helvetica-Oblique C}_{/Helvetica-Oblique v}");
                        if (lm) plot.gnuplot("set logscale x 10");
                        plot.legend().atTopRight().fontSize(12);
                        plot.drawCurvesFilled(vT, vCvlo, vCvhi)
                            .fillColor("#f4a582").fillIntensity(0.35).fillTransparent()
                            .lineColor("#f4a582").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vCv)
                            .lineColor("#b2182b").lineWidth(2).label("C_v");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(900, 600);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/specific_heat%s.png", plotdir, lm ? "_log" : "");
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
                    for (int lm = 0; lm < 2; lm++) {
                        Plot2D plot;
                        setup_analysis_plot(plot);
                        plot.xlabel("{/Helvetica-Oblique T}");
                        plot.ylabel("PT exchange rate");
                        if (lm) plot.gnuplot("set logscale x 10");
                        plot.legend().atTopRight().fontSize(12);
                        plot.drawCurvesFilled(vT, vRlo, vRhi)
                            .fillColor("#8da0cb").fillIntensity(0.35).fillTransparent()
                            .lineColor("#8da0cb").lineWidth(0).labelNone();
                        plot.drawCurve(vT, vR)
                            .lineColor("#542788").lineWidth(2).label("PT exchange");
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(900, 600);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/exchange_rates%s.png", plotdir, lm ? "_log" : "");
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    }
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
                        setup_analysis_plot(plot);
                        plot.xlabel("sweep");
                        plot.ylabel(ylabel_str);
                        plot.legend().hide();
                        plot.gnuplot("set logscale x 2");
                        plot.gnuplot(TEMP_PALETTE);
                        char cbr[128];
                        snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin, Tmax);
                        plot.gnuplot(cbr);
                        if (log_temp) plot.gnuplot("set log cb");
                        plot.gnuplot("set cblabel '{/Helvetica-Oblique T}' font 'Helvetica,12'");
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
                            double frac = (Tmax > Tmin)
                                ? (log_temp ? (log(bl[0].T) - log(Tmin)) / (log(Tmax) - log(Tmin))
                                            : (bl[0].T - Tmin) / (Tmax - Tmin))
                                : 0.5;
                            plot.drawCurve(vx, vy)
                                .lineColor(temp_color(frac))
                                .lineWidth(2)
                                .label("");
                        }
                        Figure fig = {{plot}};
                        Canvas canvas = {{fig}};
                        canvas.size(1000, 700);
                        char pf[512]; snprintf(pf, sizeof(pf), "%s/%s", plotdir, outname);
                        canvas.save(pf);
                        printf("  Written %s\n", pf);
                    };

                    make_history_plot("{/Helvetica-Oblique E} / {/Helvetica-Oblique N}", 0, "energy_history.png");
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
                plot.xlabel("{/Helvetica-Oblique q}");
                plot.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique q})");
                plot.legend().hide();
                plot.gnuplot("set logscale y 10");
                plot.gnuplot("set format y '10^{%L}'");
                plot.gnuplot("set yrange [1e-5:*]");

                int nb = (int)oblocks.size();
                double Tmin_ov = oblocks.back()[0].T;
                double Tmax_ov = oblocks.front()[0].T;
                setup_colorbar_plot(plot, Tmin_ov, Tmax_ov, log_temp);

                for (int bi = 0; bi < nb; bi++) {
                    auto& bl = oblocks[bi];
                    std::vector<double> vq, vpq;
                    for (int i = 0; i < (int)bl.size(); i++) {
                        if (bl[i].pq > 0) { vq.push_back(bl[i].q); vpq.push_back(bl[i].pq); }
                    }
                    if (vq.empty()) continue;
                    char pcol[64];
                    snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                    plot.drawCurve(vq, vpq)
                        .lineColor(pcol)
                        .lineWidth(2)
                        .label("");
                }

                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(1000, 700);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/parisi_overlap.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);

                // --- Linear-scale P(q) ---
                {
                    Plot2D plot2;
                    plot2.xlabel("{/Helvetica-Oblique q}");
                    plot2.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique q})");
                    plot2.legend().hide();
                    setup_colorbar_plot(plot2, Tmin_ov, Tmax_ov, log_temp);

                    for (int bi = 0; bi < nb; bi++) {
                        auto& bl = oblocks[bi];
                        std::vector<double> vq, vpq;
                        for (int i = 0; i < (int)bl.size(); i++) {
                            vq.push_back(bl[i].q); vpq.push_back(bl[i].pq);
                        }
                        if (vq.empty()) continue;
                        char pcol[64];
                        snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                        plot2.drawCurve(vq, vpq)
                            .lineColor(pcol)
                            .lineWidth(2)
                            .label("");
                    }
                    Figure fig2 = {{plot2}};
                    Canvas canvas2 = {{fig2}};
                    canvas2.size(1000, 700);
                    char pf2[512]; snprintf(pf2, sizeof(pf2), "%s/parisi_overlap_linear.png", plotdir);
                    canvas2.save(pf2);
                    printf("  Written %s\n", pf2);
                }
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
                plot.xlabel("{/Helvetica-Oblique C}");
                plot.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique C})");
                plot.legend().hide();
                plot.gnuplot("set logscale y 10");
                plot.gnuplot("set format y '10^{%L}'");
                plot.gnuplot("set yrange [1e-5:*]");

                int nb = (int)iblocks.size();
                double Tmin_if = iblocks.back()[0].T;
                double Tmax_if = iblocks.front()[0].T;
                setup_colorbar_plot(plot, Tmin_if, Tmax_if, log_temp);

                for (int bi = 0; bi < nb; bi++) {
                    auto& bl = iblocks[bi];
                    std::vector<double> vc, vpc;
                    for (int i = 0; i < (int)bl.size(); i++) {
                        if (bl[i].pq > 0) { vc.push_back(bl[i].q); vpc.push_back(bl[i].pq); }
                    }
                    if (vc.empty()) continue;
                    char pcol[64];
                    snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                    plot.drawCurve(vc, vpc)
                        .lineColor(pcol)
                        .lineWidth(2)
                        .label("");
                }

                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(1000, 700);
                char ipf[512]; snprintf(ipf, sizeof(ipf), "%s/ifo_overlap.png", plotdir);
                canvas.save(ipf);
                printf("  Written %s\n", ipf);

                // --- Linear-scale IFO ---
                {
                    Plot2D plot2;
                    plot2.xlabel("{/Helvetica-Oblique C}");
                    plot2.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique C})");
                    plot2.legend().hide();
                    setup_colorbar_plot(plot2, Tmin_if, Tmax_if, log_temp);

                    for (int bi = 0; bi < nb; bi++) {
                        auto& bl = iblocks[bi];
                        std::vector<double> vc, vpc;
                        for (int i = 0; i < (int)bl.size(); i++) {
                            vc.push_back(bl[i].q); vpc.push_back(bl[i].pq);
                        }
                        if (vc.empty()) continue;
                        char pcol[64];
                        snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                        plot2.drawCurve(vc, vpc)
                            .lineColor(pcol)
                            .lineWidth(2)
                            .label("");
                    }
                    Figure fig2 = {{plot2}};
                    Canvas canvas2 = {{fig2}};
                    canvas2.size(1000, 700);
                    char ipf2[512]; snprintf(ipf2, sizeof(ipf2), "%s/ifo_overlap_linear.png", plotdir);
                    canvas2.save(ipf2);
                    printf("  Written %s\n", ipf2);
                }
            }
        }

        // --- Exp-IFO overlap P(C) plots (log + linear) ---
        if (nrep > 1) {
            char expfile[512];
            snprintf(expfile, sizeof(expfile), "%s/exp_ifo_overlap.dat", outdir);

            struct OlapRow { double q, pq, err, T; };
            std::vector<std::vector<OlapRow>> eblocks;
            {
                FILE* f = fopen(expfile, "r");
                if (f) {
                    char line[512];
                    std::vector<OlapRow> cur;
                    while (fgets(line, sizeof(line), f)) {
                        if (line[0] == '#') continue;
                        if (line[0] == '\n') {
                            if (!cur.empty()) eblocks.push_back(cur);
                            cur.clear();
                            continue;
                        }
                        OlapRow r;
                        if (sscanf(line, "%lf %lf %lf %lf", &r.q, &r.pq, &r.err, &r.T) == 4)
                            cur.push_back(r);
                    }
                    if (!cur.empty()) eblocks.push_back(cur);
                    fclose(f);
                }
            }

            if (!eblocks.empty()) {
                int nb = (int)eblocks.size();
                double Tmin_ei = eblocks.back()[0].T;
                double Tmax_ei = eblocks.front()[0].T;

                // Log-scale
                {
                    Plot2D plot;
                    plot.xlabel("{/Helvetica-Oblique C}");
                    plot.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique C})");
                    plot.legend().hide();
                    plot.gnuplot("set logscale y 10");
                    plot.gnuplot("set format y '10^{%L}'");
                    plot.gnuplot("set yrange [1e-5:*]");
                    setup_colorbar_plot(plot, Tmin_ei, Tmax_ei, log_temp);

                    for (int bi = 0; bi < nb; bi++) {
                        auto& bl = eblocks[bi];
                        std::vector<double> vc, vpc;
                        for (int i = 0; i < (int)bl.size(); i++) {
                            if (bl[i].pq > 0) { vc.push_back(bl[i].q); vpc.push_back(bl[i].pq); }
                        }
                        if (vc.empty()) continue;
                        char pcol[64];
                        snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                        plot.drawCurve(vc, vpc)
                            .lineColor(pcol)
                            .lineWidth(2)
                            .label("");
                    }
                    Figure fig = {{plot}};
                    Canvas canvas = {{fig}};
                    canvas.size(1000, 700);
                    char pf[512]; snprintf(pf, sizeof(pf), "%s/exp_ifo_overlap.png", plotdir);
                    canvas.save(pf);
                    printf("  Written %s\n", pf);
                }

                // Linear-scale
                {
                    Plot2D plot;
                    plot.xlabel("{/Helvetica-Oblique C}");
                    plot.ylabel("{/Helvetica-Oblique P}({/Helvetica-Oblique C})");
                    plot.legend().hide();
                    setup_colorbar_plot(plot, Tmin_ei, Tmax_ei, log_temp);

                    for (int bi = 0; bi < nb; bi++) {
                        auto& bl = eblocks[bi];
                        std::vector<double> vc, vpc;
                        for (int i = 0; i < (int)bl.size(); i++) {
                            vc.push_back(bl[i].q); vpc.push_back(bl[i].pq);
                        }
                        if (vc.empty()) continue;
                        char pcol[64];
                        snprintf(pcol, sizeof(pcol), "palette cb %.8f", bl[0].T);
                        plot.drawCurve(vc, vpc)
                            .lineColor(pcol)
                            .lineWidth(2)
                            .label("");
                    }
                    Figure fig = {{plot}};
                    Canvas canvas = {{fig}};
                    canvas.size(1000, 700);
                    char pf[512]; snprintf(pf, sizeof(pf), "%s/exp_ifo_overlap_linear.png", plotdir);
                    canvas.save(pf);
                    printf("  Written %s\n", pf);
                }
            }
        }

        // --- Moments plots ---
        {
            const char* mom_files[3] = {
                "parisi_moments.dat", "ifo_moments.dat", "exp_ifo_moments.dat"
            };
            const char* mom_pngs[3] = {
                "parisi_moments.png", "ifo_moments.png", "exp_ifo_moments.png"
            };
            const char* mom_var[3] = { "q", "C", "C" };

            for (int mi = 0; mi < 3; mi++) {
                char mf[512]; snprintf(mf, sizeof(mf), "%s/%s", outdir, mom_files[mi]);
                FILE* f = fopen(mf, "r");
                if (!f) continue;

                std::vector<double> vT, vMu, eMu, vVar, eVar, vSkew, eSkew, vKurt, eKurt;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    double T_, m_, me_, v_, ve_, s_, se_, k_, ke_;
                    if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                               &T_, &m_, &me_, &v_, &ve_, &s_, &se_, &k_, &ke_) == 9) {
                        vT.push_back(T_);  vMu.push_back(m_);  eMu.push_back(me_);
                        vVar.push_back(v_); eVar.push_back(ve_);
                        vSkew.push_back(s_); eSkew.push_back(se_);
                        vKurt.push_back(k_); eKurt.push_back(ke_);
                    }
                }
                fclose(f);
                if (vT.empty()) continue;

                int nd = (int)vT.size();
                std::vector<double> muLo(nd), muHi(nd), varLo(nd), varHi(nd);
                std::vector<double> skLo(nd), skHi(nd), kuLo(nd), kuHi(nd);
                for (int i = 0; i < nd; i++) {
                    muLo[i]  = vMu[i]   - eMu[i];   muHi[i]  = vMu[i]   + eMu[i];
                    varLo[i] = vVar[i]  - eVar[i];   varHi[i] = vVar[i]  + eVar[i];
                    skLo[i]  = vSkew[i] - eSkew[i];  skHi[i]  = vSkew[i] + eSkew[i];
                    kuLo[i]  = vKurt[i] - eKurt[i];  kuHi[i]  = vKurt[i] + eKurt[i];
                }

                char vl[64];

                Plot2D p1; setup_analysis_plot(p1);
                p1.gnuplot("set logscale x 10");
                p1.xlabel("{/Helvetica-Oblique T}");
                snprintf(vl, sizeof(vl), "<{/Helvetica-Oblique %s}>", mom_var[mi]);
                p1.ylabel(vl);
                p1.legend().hide();
                p1.drawCurvesFilled(vT, muLo, muHi)
                    .fillColor("#4393c3").fillIntensity(0.35).fillTransparent()
                    .lineColor("#4393c3").lineWidth(0).labelNone();
                p1.drawCurve(vT, vMu).lineColor("#2166ac").lineWidth(2).label("");

                Plot2D p2; setup_analysis_plot(p2);
                p2.gnuplot("set logscale x 10");
                p2.xlabel("{/Helvetica-Oblique T}");
                snprintf(vl, sizeof(vl), "Var({/Helvetica-Oblique %s})", mom_var[mi]);
                p2.ylabel(vl);
                p2.legend().hide();
                p2.drawCurvesFilled(vT, varLo, varHi)
                    .fillColor("#66c2a5").fillIntensity(0.35).fillTransparent()
                    .lineColor("#66c2a5").lineWidth(0).labelNone();
                p2.drawCurve(vT, vVar).lineColor("#1b9e77").lineWidth(2).label("");

                Plot2D p3; setup_analysis_plot(p3);
                p3.gnuplot("set logscale x 10");
                p3.xlabel("{/Helvetica-Oblique T}");
                snprintf(vl, sizeof(vl), "Skew({/Helvetica-Oblique %s})", mom_var[mi]);
                p3.ylabel(vl);
                p3.legend().hide();
                p3.drawCurvesFilled(vT, skLo, skHi)
                    .fillColor("#fc8d62").fillIntensity(0.35).fillTransparent()
                    .lineColor("#fc8d62").lineWidth(0).labelNone();
                p3.drawCurve(vT, vSkew).lineColor("#d95f02").lineWidth(2).label("");

                Plot2D p4; setup_analysis_plot(p4);
                p4.gnuplot("set logscale x 10");
                p4.xlabel("{/Helvetica-Oblique T}");
                snprintf(vl, sizeof(vl), "Kurt({/Helvetica-Oblique %s})", mom_var[mi]);
                p4.ylabel(vl);
                p4.legend().hide();
                p4.drawCurvesFilled(vT, kuLo, kuHi)
                    .fillColor("#8da0cb").fillIntensity(0.35).fillTransparent()
                    .lineColor("#8da0cb").lineWidth(0).labelNone();
                p4.drawCurve(vT, vKurt).lineColor("#7570b3").lineWidth(2).label("");

                Figure fig = {{p1, p2}, {p3, p4}};
                Canvas canvas = {{fig}};
                canvas.size(1200, 900);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/%s", plotdir, mom_pngs[mi]);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // ============================================================
        // New Plots: distributions, glass observables, IPR
        // ============================================================

        // Helper lambda: generic colorbar distribution plot from file
        auto make_dist_plot = [&](const char* datafile, const char* xl,
                                  const char* yl, const char* pngname,
                                  bool logy = true) {
            struct Row { double x, p, e, T; };
            std::vector<std::vector<Row>> blocks;
            FILE* f = fopen(datafile, "r");
            if (f) {
                char line[512]; std::vector<Row> cur;
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#') continue;
                    if (line[0] == '\n') {
                        if (!cur.empty()) blocks.push_back(cur);
                        cur.clear(); continue;
                    }
                    Row r;
                    if (sscanf(line, "%lf %lf %lf %lf", &r.x, &r.p, &r.e, &r.T) == 4)
                        cur.push_back(r);
                }
                if (!cur.empty()) blocks.push_back(cur);
                fclose(f);
            }
            if (blocks.empty()) return;

            Plot2D plot; plot.xlabel(xl); plot.ylabel(yl); plot.legend().hide();
            if (logy) {
                plot.gnuplot("set logscale y 10");
                plot.gnuplot("set format y '10^{%L}'");
                plot.gnuplot("set yrange [1e-5:*]");
            }
            double Tlo = blocks.back()[0].T, Thi = blocks.front()[0].T;
            setup_colorbar_plot(plot, Tlo, Thi, log_temp);
            for (auto& bl : blocks) {
                std::vector<double> vx, vy;
                for (auto& r : bl) {
                    if (logy && r.p <= 0) continue;
                    vx.push_back(r.x); vy.push_back(r.p);
                }
                if (vx.empty()) continue;
                char pc[64]; snprintf(pc, sizeof(pc), "palette cb %.8f", bl[0].T);
                plot.drawCurve(vx, vy).lineColor(pc).lineWidth(2).label("");
            }
            Figure fig = {{plot}};
            Canvas canvas = {{fig}};
            canvas.size(1000, 700);
            char pf[512]; snprintf(pf, sizeof(pf), "%s/%s", plotdir, pngname);
            canvas.save(pf);
            printf("  Written %s\n", pf);
        };

        // Phase overlap P(q_phi)
        {
            char df[512]; snprintf(df, sizeof(df), "%s/phase_overlap.dat", outdir);
            make_dist_plot(df,
                "{/Helvetica-Oblique q}_{/Symbol f}",
                "{/Helvetica-Oblique P}({/Helvetica-Oblique q}_{/Symbol f})",
                "phase_overlap.png");
            make_dist_plot(df,
                "{/Helvetica-Oblique q}_{/Symbol f}",
                "{/Helvetica-Oblique P}({/Helvetica-Oblique q}_{/Symbol f})",
                "phase_overlap_linear.png", false);
        }

        // Link overlap P(q_link)
        {
            char df[512]; snprintf(df, sizeof(df), "%s/link_overlap.dat", outdir);
            make_dist_plot(df,
                "{/Helvetica-Oblique q}_{link}",
                "{/Helvetica-Oblique P}({/Helvetica-Oblique q}_{link})",
                "link_overlap.png");
            make_dist_plot(df,
                "{/Helvetica-Oblique q}_{link}",
                "{/Helvetica-Oblique P}({/Helvetica-Oblique q}_{link})",
                "link_overlap_linear.png", false);
        }

        // Marginal P(|a|^2) — both log and linear
        {
            char df[512]; snprintf(df, sizeof(df), "%s/marginal_a2.dat", outdir);
            make_dist_plot(df,
                "|{/Helvetica-Oblique a}_k|^2",
                "{/Helvetica-Oblique P}(|{/Helvetica-Oblique a}_k|^2)",
                "marginal_a2_log.png", true);
            make_dist_plot(df,
                "|{/Helvetica-Oblique a}_k|^2",
                "{/Helvetica-Oblique P}(|{/Helvetica-Oblique a}_k|^2)",
                "marginal_a2.png", false);
        }

        // --- Glass observables combined plot (chi, g4, A vs T) ---
        {
            struct GOLine { double T, chi, chi_e, g4, g4_e, A, A_e; };
            auto read_go = [](const char* path) {
                std::vector<GOLine> rows;
                FILE* f = fopen(path, "r");
                if (!f) return rows;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    GOLine r;
                    if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf",
                               &r.T, &r.chi, &r.chi_e, &r.g4, &r.g4_e, &r.A, &r.A_e) == 7)
                        rows.push_back(r);
                }
                fclose(f);
                return rows;
            };

            const char* tags[]   = {"parisi", "ifo", "exp_ifo", "phase", "link"};
            const char* gnames[] = {"Parisi", "IFO", "exp-IFO", "Phase", "Link"};
            const char* gcols[]  = {"#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"};
            const int ntags = 5;
            std::vector<std::vector<GOLine>> godata(ntags);
            for (int i = 0; i < ntags; i++) {
                char gf[512]; snprintf(gf, sizeof(gf), "%s/%s_glass_observables.dat", outdir, tags[i]);
                godata[i] = read_go(gf);
            }

            // Any data at all?
            bool any = false;
            for (int i = 0; i < ntags; i++) if (!godata[i].empty()) any = true;
            if (any) {
                // chi panel
                Plot2D pChi; setup_analysis_plot(pChi);
                pChi.gnuplot("set title '{/Helvetica-Bold Susceptibility {/Symbol c}}'");
                pChi.gnuplot("set logscale x 10");
                pChi.xlabel("{/Helvetica-Oblique T}");
                pChi.ylabel("{/Symbol c}");
                for (int i = 0; i < ntags; i++) {
                    if (godata[i].empty()) continue;
                    std::vector<double> vT, vC, vClo, vChi2;
                    for (auto& r : godata[i]) {
                        vT.push_back(r.T); vC.push_back(r.chi);
                        vClo.push_back(r.chi - r.chi_e);
                        vChi2.push_back(r.chi + r.chi_e);
                    }
                    pChi.drawCurvesFilled(vT, vClo, vChi2)
                        .fillColor(gcols[i]).fillIntensity(0.2).fillTransparent()
                        .lineColor(gcols[i]).lineWidth(0).labelNone();
                    pChi.drawCurve(vT, vC).lineColor(gcols[i]).lineWidth(2).label(gnames[i]);
                }

                // g4 panel
                Plot2D pG4; setup_analysis_plot(pG4);
                pG4.gnuplot("set title '{/Helvetica-Bold Binder parameter g_4}'");
                pG4.gnuplot("set logscale x 10");
                pG4.xlabel("{/Helvetica-Oblique T}");
                pG4.ylabel("{/Helvetica-Oblique g}_4");
                for (int i = 0; i < ntags; i++) {
                    if (godata[i].empty()) continue;
                    std::vector<double> vT, vG, vGlo, vGhi;
                    for (auto& r : godata[i]) {
                        vT.push_back(r.T); vG.push_back(r.g4);
                        vGlo.push_back(r.g4 - r.g4_e);
                        vGhi.push_back(r.g4 + r.g4_e);
                    }
                    pG4.drawCurvesFilled(vT, vGlo, vGhi)
                        .fillColor(gcols[i]).fillIntensity(0.2).fillTransparent()
                        .lineColor(gcols[i]).lineWidth(0).labelNone();
                    pG4.drawCurve(vT, vG).lineColor(gcols[i]).lineWidth(2).label(gnames[i]);
                }

                // A panel
                Plot2D pA; setup_analysis_plot(pA);
                pA.gnuplot("set title '{/Helvetica-Bold Non-self-averaging A}'");
                pA.gnuplot("set logscale x 10");
                pA.xlabel("{/Helvetica-Oblique T}");
                pA.ylabel("{/Helvetica-Oblique A}({/Helvetica-Oblique T})");
                for (int i = 0; i < ntags; i++) {
                    if (godata[i].empty()) continue;
                    std::vector<double> vT, vA, vAlo, vAhi;
                    for (auto& r : godata[i]) {
                        vT.push_back(r.T); vA.push_back(r.A);
                        vAlo.push_back(r.A - r.A_e);
                        vAhi.push_back(r.A + r.A_e);
                    }
                    pA.drawCurvesFilled(vT, vAlo, vAhi)
                        .fillColor(gcols[i]).fillIntensity(0.2).fillTransparent()
                        .lineColor(gcols[i]).lineWidth(0).labelNone();
                    pA.drawCurve(vT, vA).lineColor(gcols[i]).lineWidth(2).label(gnames[i]);
                }

                // Scatter panel: q_parisi vs q_link
                Plot2D pScat; setup_analysis_plot(pScat);
                pScat.gnuplot("set title '{/Helvetica-Bold Scatter q vs q_{link}}'");
                pScat.xlabel("{/Helvetica-Oblique q}_{Parisi}");
                pScat.ylabel("{/Helvetica-Oblique q}_{link}");
                bool has_scatter = false;
                {
                    char sf[512]; snprintf(sf, sizeof(sf), "%s/scatter_q_qlink.dat", outdir);
                    FILE* fscat = fopen(sf, "r");
                    if (fscat) {
                        // Group points by temperature
                        struct SPt { double q, ql, T; };
                        std::vector<SPt> allpts;
                        char line[512];
                        while (fgets(line, sizeof(line), fscat)) {
                            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
                            SPt p;
                            if (sscanf(line, "%lf %lf %lf", &p.q, &p.ql, &p.T) == 3)
                                allpts.push_back(p);
                        }
                        fclose(fscat);
                        if (!allpts.empty()) {
                            has_scatter = true;
                            // Find T range
                            double Tmin = allpts[0].T, Tmax = allpts[0].T;
                            for (auto& p : allpts) {
                                if (p.T < Tmin) Tmin = p.T;
                                if (p.T > Tmax) Tmax = p.T;
                            }
                            pScat.gnuplot("set rmargin 14");
                            pScat.gnuplot(TEMP_PALETTE);
                            char cbr[128];
                            snprintf(cbr, sizeof(cbr), "set cbrange [%g:%g]", Tmin, Tmax);
                            pScat.gnuplot(cbr);
                            pScat.gnuplot("set cblabel '{/Helvetica-Oblique T}' font 'Helvetica,13' offset 1,0");
                            pScat.gnuplot("set cbtics font 'Helvetica,11'");
                            pScat.gnuplot("set colorbox vertical user origin 0.88, 0.15 size 0.025, 0.7");
                            pScat.legend().hide();
                            // Group by unique T
                            std::map<double,std::vector<SPt>> byT;
                            for (auto& p : allpts) byT[p.T].push_back(p);
                            for (auto& [T, pts] : byT) {
                                std::vector<double> vq, vql;
                                for (auto& p : pts) { vq.push_back(p.q); vql.push_back(p.ql); }
                                char pcol[64];
                                snprintf(pcol, sizeof(pcol), "palette cb %.8f", T);
                                pScat.drawPoints(vq, vql)
                                    .lineColor(pcol)
                                    .pointType(7).pointSize(1)
                                    .label("");
                            }
                        }
                    }
                }
                if (!has_scatter) {
                    pScat.legend().hide();
                }

                Figure fig = {{pChi, pG4}, {pA, pScat}};
                Canvas canvas = {{fig}};
                canvas.size(1600, 1200);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/glass_observables.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- IPR plot (Y2 and 1/Y2 vs T) ---
        {
            struct IPRLine { double T, y2, y2e, y4, y4e; };
            std::vector<IPRLine> ipr_rows;
            char iprfile[512]; snprintf(iprfile, sizeof(iprfile), "%s/ipr.dat", outdir);
            FILE* f = fopen(iprfile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    IPRLine r;
                    if (sscanf(line, "%lf %lf %lf %lf %lf",
                               &r.T, &r.y2, &r.y2e, &r.y4, &r.y4e) == 5)
                        ipr_rows.push_back(r);
                }
                fclose(f);
            }
            if (!ipr_rows.empty()) {
                std::vector<double> vT, vY2, vY2lo, vY2hi, vInv, vInvLo, vInvHi;
                for (auto& r : ipr_rows) {
                    vT.push_back(r.T);
                    vY2.push_back(r.y2);
                    vY2lo.push_back(r.y2 - r.y2e);
                    vY2hi.push_back(r.y2 + r.y2e);
                    double inv = (r.y2 > 0) ? 1.0 / r.y2 : 0;
                    double inv_e = (r.y2 > 0) ? r.y2e / (r.y2 * r.y2) : 0;
                    vInv.push_back(inv);
                    vInvLo.push_back(inv - inv_e);
                    vInvHi.push_back(inv + inv_e);
                }

                Plot2D p1; setup_analysis_plot(p1);
                p1.gnuplot("set logscale x 10");
                p1.xlabel("{/Helvetica-Oblique T}");
                p1.ylabel("{/Helvetica-Oblique Y}_2");
                p1.legend().hide();
                p1.drawCurvesFilled(vT, vY2lo, vY2hi)
                    .fillColor("#e41a1c").fillIntensity(0.3).fillTransparent()
                    .lineColor("#e41a1c").lineWidth(0).labelNone();
                p1.drawCurve(vT, vY2).lineColor("#e41a1c").lineWidth(2).label("");

                Plot2D p2; setup_analysis_plot(p2);
                p2.gnuplot("set logscale x 10");
                p2.xlabel("{/Helvetica-Oblique T}");
                p2.ylabel("1/{/Helvetica-Oblique Y}_2  (participating modes)");
                p2.legend().hide();
                p2.drawCurvesFilled(vT, vInvLo, vInvHi)
                    .fillColor("#377eb8").fillIntensity(0.3).fillTransparent()
                    .lineColor("#377eb8").lineWidth(0).labelNone();
                p2.drawCurve(vT, vInv).lineColor("#377eb8").lineWidth(2).label("");

                Figure fig = {{p1, p2}};
                Canvas canvas = {{fig}};
                canvas.size(1400, 500);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/ipr.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Equipartition parameter EP(T) ---
        {
            struct EPLine { double T, ep, eperr; };
            std::vector<EPLine> ep_rows;
            char epfile[512]; snprintf(epfile, sizeof(epfile), "%s/ep.dat", outdir);
            FILE* f = fopen(epfile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    EPLine r;
                    if (sscanf(line, "%lf %lf %lf", &r.T, &r.ep, &r.eperr) == 3)
                        ep_rows.push_back(r);
                }
                fclose(f);
            }
            if (!ep_rows.empty()) {
                std::vector<double> vT, vEP, vLo, vHi;
                for (auto& r : ep_rows) {
                    vT.push_back(r.T); vEP.push_back(r.ep);
                    vLo.push_back(r.ep - r.eperr); vHi.push_back(r.ep + r.eperr);
                }
                Plot2D plot; setup_analysis_plot(plot);
                if (log_temp) plot.gnuplot("set logscale x 10");
                plot.xlabel("{/Helvetica-Oblique T}");
                plot.ylabel("{/Helvetica-Oblique EP}");
                plot.legend().hide();
                plot.drawCurvesFilled(vT, vLo, vHi)
                    .fillColor("#984ea3").fillIntensity(0.3).fillTransparent()
                    .lineColor("#984ea3").lineWidth(0).labelNone();
                plot.drawCurve(vT, vEP).lineColor("#984ea3").lineWidth(2).label("");
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/equipartition.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Shannon Entropy & Gini ---
        {
            struct SGLine { double T, S, Serr, G, Gerr; };
            std::vector<SGLine> sg_rows;
            char sgfile[512]; snprintf(sgfile, sizeof(sgfile), "%s/entropy_gini.dat", outdir);
            FILE* f = fopen(sgfile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    SGLine r;
                    if (sscanf(line, "%lf %lf %lf %lf %lf",
                               &r.T, &r.S, &r.Serr, &r.G, &r.Gerr) == 5)
                        sg_rows.push_back(r);
                }
                fclose(f);
            }
            if (!sg_rows.empty()) {
                std::vector<double> vT, vS, vSlo, vShi, vG, vGlo, vGhi;
                for (auto& r : sg_rows) {
                    vT.push_back(r.T);
                    vS.push_back(r.S); vSlo.push_back(r.S - r.Serr); vShi.push_back(r.S + r.Serr);
                    vG.push_back(r.G); vGlo.push_back(r.G - r.Gerr); vGhi.push_back(r.G + r.Gerr);
                }
                Plot2D pS; setup_analysis_plot(pS);
                if (log_temp) pS.gnuplot("set logscale x 10");
                pS.xlabel("{/Helvetica-Oblique T}");
                pS.ylabel("Shannon entropy  {/Helvetica-Oblique S}");
                pS.legend().hide();
                pS.drawCurvesFilled(vT, vSlo, vShi)
                    .fillColor("#4daf4a").fillIntensity(0.3).fillTransparent()
                    .lineColor("#4daf4a").lineWidth(0).labelNone();
                pS.drawCurve(vT, vS).lineColor("#4daf4a").lineWidth(2).label("");

                Plot2D pG; setup_analysis_plot(pG);
                if (log_temp) pG.gnuplot("set logscale x 10");
                pG.xlabel("{/Helvetica-Oblique T}");
                pG.ylabel("Gini coefficient  {/Helvetica-Oblique G}");
                pG.legend().hide();
                pG.drawCurvesFilled(vT, vGlo, vGhi)
                    .fillColor("#ff7f00").fillIntensity(0.3).fillTransparent()
                    .lineColor("#ff7f00").lineWidth(0).labelNone();
                pG.drawCurve(vT, vG).lineColor("#ff7f00").lineWidth(2).label("");

                Figure fig = {{pS, pG}};
                Canvas canvas = {{fig}};
                canvas.size(1400, 500);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/entropy_gini.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Non-self-averaging A_E ---
        {
            struct AELine { double T, A, Aerr; };
            std::vector<AELine> ae_rows;
            char aefile[512]; snprintf(aefile, sizeof(aefile), "%s/nsa_energy.dat", outdir);
            FILE* f = fopen(aefile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    AELine r;
                    if (sscanf(line, "%lf %lf %lf", &r.T, &r.A, &r.Aerr) == 3)
                        ae_rows.push_back(r);
                }
                fclose(f);
            }
            if (!ae_rows.empty()) {
                std::vector<double> vT, vA, vLo, vHi;
                for (auto& r : ae_rows) {
                    vT.push_back(r.T); vA.push_back(r.A);
                    vLo.push_back(r.A - r.Aerr); vHi.push_back(r.A + r.Aerr);
                }
                Plot2D plot; setup_analysis_plot(plot);
                if (log_temp) plot.gnuplot("set logscale x 10");
                plot.xlabel("{/Helvetica-Oblique T}");
                plot.ylabel("{/Helvetica-Oblique A}_E (non-self-averaging)");
                plot.legend().hide();
                plot.drawCurvesFilled(vT, vLo, vHi)
                    .fillColor("#a65628").fillIntensity(0.3).fillTransparent()
                    .lineColor("#a65628").lineWidth(0).labelNone();
                plot.drawCurve(vT, vA).lineColor("#a65628").lineWidth(2).label("");
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/nsa_energy.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Decorrelation time tau_E ---
        {
            struct TauLine { double T, tau, tauerr; };
            std::vector<TauLine> tau_rows;
            char taufile[512]; snprintf(taufile, sizeof(taufile), "%s/decorrelation_time.dat", outdir);
            FILE* f = fopen(taufile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    TauLine r;
                    if (sscanf(line, "%lf %lf %lf", &r.T, &r.tau, &r.tauerr) == 3)
                        tau_rows.push_back(r);
                }
                fclose(f);
            }
            if (!tau_rows.empty()) {
                std::vector<double> vT, vTau, vLo, vHi;
                for (auto& r : tau_rows) {
                    vT.push_back(r.T); vTau.push_back(r.tau);
                    vLo.push_back(r.tau - r.tauerr); vHi.push_back(r.tau + r.tauerr);
                }
                Plot2D plot; setup_analysis_plot(plot);
                if (log_temp) plot.gnuplot("set logscale x 10");
                plot.xlabel("{/Helvetica-Oblique T}");
                plot.ylabel("{/Symbol t}_E  (decorrelation time)");
                plot.legend().hide();
                plot.drawCurvesFilled(vT, vLo, vHi)
                    .fillColor("#e41a1c").fillIntensity(0.3).fillTransparent()
                    .lineColor("#e41a1c").lineWidth(0).labelNone();
                plot.drawCurve(vT, vTau).lineColor("#e41a1c").lineWidth(2).label("");
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/decorrelation_time.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- E2/E4 energy split ---
        {
            struct ESLine { double T, e2, e2err, e4, e4err; };
            std::vector<ESLine> es_rows;
            char esfile[512]; snprintf(esfile, sizeof(esfile), "%s/energy_split.dat", outdir);
            FILE* f = fopen(esfile, "r");
            if (f) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    ESLine r;
                    if (sscanf(line, "%lf %lf %lf %lf %lf",
                               &r.T, &r.e2, &r.e2err, &r.e4, &r.e4err) == 5)
                        es_rows.push_back(r);
                }
                fclose(f);
            }
            if (!es_rows.empty()) {
                std::vector<double> vT, v2, v2lo, v2hi, v4, v4lo, v4hi;
                for (auto& r : es_rows) {
                    vT.push_back(r.T);
                    v2.push_back(r.e2); v2lo.push_back(r.e2 - r.e2err); v2hi.push_back(r.e2 + r.e2err);
                    v4.push_back(r.e4); v4lo.push_back(r.e4 - r.e4err); v4hi.push_back(r.e4 + r.e4err);
                }
                Plot2D plot; setup_analysis_plot(plot);
                if (log_temp) plot.gnuplot("set logscale x 10");
                plot.xlabel("{/Helvetica-Oblique T}");
                plot.ylabel("{/Helvetica-Oblique E/N}");
                plot.drawCurvesFilled(vT, v2lo, v2hi)
                    .fillColor("#377eb8").fillIntensity(0.3).fillTransparent()
                    .lineColor("#377eb8").lineWidth(0).labelNone();
                plot.drawCurve(vT, v2).lineColor("#377eb8").lineWidth(2).label("E_2/N");
                plot.drawCurvesFilled(vT, v4lo, v4hi)
                    .fillColor("#e41a1c").fillIntensity(0.3).fillTransparent()
                    .lineColor("#e41a1c").lineWidth(0).labelNone();
                plot.drawCurve(vT, v4).lineColor("#e41a1c").lineWidth(2).label("E_4/N");
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/energy_split.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Autocorrelation C(tau) ---
        {
            // Plot all available autocorrelation files
            Plot2D plot; setup_analysis_plot(plot);
            plot.xlabel("{/Symbol t}  (lag)");
            plot.ylabel("{/Helvetica-Oblique C}({/Symbol t})");
            const char* acols[] = {"#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"};
            int nac = 0;
            for (int ti = 0; ti < ntemps && nac < 5; ti++) {
                int tsel[] = { 0, ntemps / 4, ntemps / 2, 3 * ntemps / 4, ntemps - 1 };
                bool found = false;
                for (int tt = 0; tt < 5 && !found; tt++)
                    if (tsel[tt] == ti) found = true;
                if (!found) continue;

                char acfile[512];
                snprintf(acfile, sizeof(acfile), "%s/autocorr_T%.4f.dat", outdir, temps[ti]);
                FILE* f = fopen(acfile, "r");
                if (!f) continue;

                std::vector<double> vlags, vC;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    int lag; double C, Cerr;
                    if (sscanf(line, "%d %lf %lf", &lag, &C, &Cerr) >= 2) {
                        vlags.push_back(lag); vC.push_back(C);
                    }
                }
                fclose(f);
                if (!vlags.empty()) {
                    char lbl[64]; snprintf(lbl, sizeof(lbl), "T=%.3f", temps[ti]);
                    plot.drawCurve(vlags, vC)
                        .lineColor(acols[nac % 5]).lineWidth(2).label(lbl);
                    nac++;
                }
            }
            if (nac > 0) {
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/autocorrelation.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Mode-mode eigenvalue spectrum ---
        {
            const char* tnames_mm[] = { "hot", "mid", "cold" };
            const char* ecols[] = {"#e41a1c", "#4daf4a", "#377eb8"};
            Plot2D plot; setup_analysis_plot(plot);
            plot.xlabel("Eigenvalue index");
            plot.ylabel("{/Symbol l} / {/Helvetica-Oblique N}");
            plot.gnuplot("set logscale y 10");
            int nec = 0;
            for (int tt = 0; tt < 3; tt++) {
                char evfile[512];
                snprintf(evfile, sizeof(evfile), "%s/eigvals_%s.dat", outdir, tnames_mm[tt]);
                FILE* f = fopen(evfile, "r");
                if (!f) continue;
                std::vector<double> vi, vl;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    int idx; double lam, lamN;
                    if (sscanf(line, "%d %lf %lf", &idx, &lam, &lamN) == 3) {
                        vi.push_back(idx); vl.push_back(lamN);
                    }
                }
                fclose(f);
                if (!vi.empty()) {
                    plot.drawCurve(vi, vl).lineColor(ecols[tt]).lineWidth(2).label(tnames_mm[tt]);
                    nec++;
                }
            }
            if (nec > 0) {
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/eigvals.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }

        // --- Mode-frequency correlation scatter ---
        {
            const char* mfnames[] = { "hot", "cold" };
            const char* mfcols[] = { "#e41a1c", "#377eb8" };
            Plot2D plot; setup_analysis_plot(plot);
            plot.xlabel("{/Symbol w}_k");
            plot.ylabel("<{/Helvetica-Oblique I}_k>");
            int nmc = 0;
            for (int tt = 0; tt < 2; tt++) {
                char mffile[512];
                snprintf(mffile, sizeof(mffile), "%s/mode_freq_corr_%s.dat", outdir, mfnames[tt]);
                FILE* f = fopen(mffile, "r");
                if (!f) continue;
                std::vector<double> vw, vI;
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (line[0] == '#' || line[0] == '\n') continue;
                    double w, I;
                    if (sscanf(line, "%lf %lf", &w, &I) == 2) {
                        vw.push_back(w); vI.push_back(I);
                    }
                }
                fclose(f);
                if (!vw.empty()) {
                    char lbl[64]; snprintf(lbl, sizeof(lbl), "T=%s", mfnames[tt]);
                    plot.drawPoints(vw, vI)
                        .lineColor(mfcols[tt]).pointType(7).pointSize(0.8).label(lbl);
                    nmc++;
                }
            }
            if (nmc > 0) {
                Figure fig = {{plot}};
                Canvas canvas = {{fig}};
                canvas.size(900, 600);
                char pf[512]; snprintf(pf, sizeof(pf), "%s/mode_freq_corr.png", plotdir);
                canvas.save(pf);
                printf("  Written %s\n", pf);
            }
        }
    }

    printf("\n── Done ────────────────────────────────────────────\n\n");
    return 0;
}
