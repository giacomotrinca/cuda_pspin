#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

struct SimConfig {
    int save_freq;          // save every save_freq iterations
    int verbose;            // print progress to stdout
    int N;                  // number of spins
    int nrep;               // number of replicas
    double T;               // temperature
    double J;               // coupling scale (J^2 sets variance)
    int mc_iterations;      // total MC iterations
    uint64_t seed;          // master RNG seed
    int label;              // sample label (-1 = no label)
    int dev;                // GPU device index (-1 = default)
};

// Default configuration
inline SimConfig default_config() {
    SimConfig cfg;
    cfg.N = 64;
    cfg.nrep = 1;
    cfg.T = 1.0;
    cfg.J = 1.0;
    cfg.mc_iterations = 10000;
    cfg.seed = 42;
    cfg.save_freq = 100;
    cfg.verbose = 0;
    cfg.label = -1;
    cfg.dev = -1;
    return cfg;
}

SimConfig parse_args(int argc, char** argv);

#endif
