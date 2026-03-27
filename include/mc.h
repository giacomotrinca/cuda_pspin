#ifndef MC_H
#define MC_H

#include "config.h"
#include <cuComplex.h>

struct MCState {
    cuDoubleComplex* d_spins;    // device: spin configuration [N]
    cuDoubleComplex* d_g2;       // device: 2-body couplings [N*N]
    cuDoubleComplex* d_g4;       // device: 4-body couplings [C(N,4)]
    double* d_workspace;         // device: reduction workspace
    double energy;               // current total energy
    long long accepted;          // accepted moves counter
    long long proposed;          // proposed moves counter
    int N;
};

// Allocate and initialize the MC state (spins, disorder, energy)
MCState mc_init(const SimConfig& cfg);

// Free all device memory
void mc_free(MCState& state);

// Perform one full MC sweep:
// - iterate over all pairs (i,j), propose rotation, accept/reject
// - iterate over all quadruplets (i,j,k,l) — future: selection rule
void mc_sweep(MCState& state, const SimConfig& cfg,
              curandStatePhilox4_32_10_t* d_rng_states);

// Get acceptance ratio
double mc_acceptance_ratio(const MCState& state);

#endif
