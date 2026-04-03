#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

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

#endif
