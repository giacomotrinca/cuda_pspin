#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

struct SimConfig {
    int N;                  // number of spins
    double T;               // temperature
    double J;               // coupling scale (J^2 sets variance)
    int n_sweeps;           // number of MC sweeps
    int n_therm;            // thermalization sweeps
    int measure_every;      // measurement interval
    uint64_t seed;          // RNG seed
    double delta;           // MC proposal amplitude
};

// Default configuration
inline SimConfig default_config() {
    SimConfig cfg;
    cfg.N = 64;
    cfg.T = 1.0;
    cfg.J = 1.0;
    cfg.n_sweeps = 10000;
    cfg.n_therm = 1000;
    cfg.measure_every = 10;
    cfg.seed = 42;
    cfg.delta = 0.1;
    return cfg;
}

SimConfig parse_args(int argc, char** argv);

#endif
