#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

struct SimConfig {
    int save_freq;          // save every save_freq iterations
    int verbose;            // print progress to stdout
    int N;                  // number of spins
    int nrep;               // number of replicas
    double T;               // temperature
    double J;               // total coupling scale
    double J0;              // mean coupling scale
    double alpha;           // fraction of J going to 4-body: J4 = alpha*J, J2 = (1-alpha)*J
    double alpha0;          // fraction of J0 going to 4-body mean: J4_0 = alpha0*J0, J2_0 = (1-alpha0)*J0
    int mc_iterations;      // total MC iterations
    uint64_t seed;          // master RNG seed
    int label;              // sample label (-1 = no label)
    int dev;                // GPU device index (-1 = default)
    int fmc_mode;           // 0=fully-connected, 1=comb, 2=uniform
    double gamma;           // FMC bandwidth
};

// Default configuration
inline SimConfig default_config() {
    SimConfig cfg;
    cfg.N = 64;
    cfg.nrep = 1;
    cfg.T = 1.0;
    cfg.J = 2.0;
    cfg.J0 = 0.0;
    cfg.alpha = 0.5;
    cfg.alpha0 = 0.5;
    cfg.mc_iterations = 10000;
    cfg.seed = 42;
    cfg.save_freq = 100;
    cfg.verbose = 0;
    cfg.label = -1;
    cfg.dev = -1;
    cfg.fmc_mode = 0;
    cfg.gamma = 0.0;
    return cfg;
}

SimConfig parse_args(int argc, char** argv);

// Format a clean float tag for directory names (removes trailing zeros, keeps one decimal)
// e.g. 0.500000 -> "0.50", 1.000000 -> "1.00", 0.333333 -> "0.333333"
inline void fmt_param(char* buf, int sz, double v) {
    snprintf(buf, sz, "%.6g", v);
    // Ensure at least one decimal point for readability
    if (!strchr(buf, '.') && !strchr(buf, 'e')) {
        int n = (int)strlen(buf);
        if (n + 3 < sz) { buf[n] = '.'; buf[n+1] = '0'; buf[n+2] = '\0'; }
    }
}

// Build parameter-aware directory name for data:
//   data/{prefix}_N{N}_a{alpha}_R{R}_a0{alpha0}_NT{NT}_NR{nrep}_S{label}
// For analysis (label < 0):
//   analysis/{prefix}_N{N}_a{alpha}_R{R}_a0{alpha0}_NT{NT}_NR{nrep}
inline void make_run_dir(char* buf, int sz,
                         const char* base,   // "data" or "analysis"
                         const char* prefix,  // "PT" or "PTS"
                         int N, double alpha, double J, double J0, double alpha0,
                         int NT, int nrep, int label) {
    char sa[32], sR[32], sa0[32];
    fmt_param(sa,  sizeof(sa),  alpha);
    fmt_param(sa0, sizeof(sa0), alpha0);
    if (J0 == 0.0)
        snprintf(sR, sizeof(sR), "inf");
    else {
        double R = J / J0;
        fmt_param(sR, sizeof(sR), R);
    }
    if (label >= 0)
        snprintf(buf, sz, "%s/%s_N%d_a%s_R%s_a0%s_NT%d_NR%d_S%d",
                 base, prefix, N, sa, sR, sa0, NT, nrep, label);
    else
        snprintf(buf, sz, "%s/%s_N%d_a%s_R%s_a0%s_NT%d_NR%d",
                 base, prefix, N, sa, sR, sa0, NT, nrep);
}

#endif
