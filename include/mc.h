#ifndef MC_H
#define MC_H

#include "config.h"
#include <cuComplex.h>
#include <curand_kernel.h>

struct MCState {
    // Shared disorder (same for all replicas)
    cuDoubleComplex* d_g2;       // device: 2-body couplings [N*N]
    cuDoubleComplex* d_g4;       // device: 4-body couplings [3 * C(N,4)]

    // Per-replica data (contiguous, indexed by replica)
    cuDoubleComplex* d_spins;    // device: spin configurations [nrep * N]
    curandStatePhilox4_32_10_t* d_rng; // device: RNG states [nrep]
    double* d_energies;          // device: energies [nrep]
    long long* d_accepted;       // device: accepted moves [nrep]
    long long* d_proposed;       // device: proposed moves [nrep]
    double* d_betas;             // device: per-replica inverse temperatures [nrep]

    double* h_omega;             // host: frequencies [N] (NULL if FC)
    long long n_pairs_active;    // active pairs after FMC
    long long n_quart_active;    // active quartets after FMC
    int h2_active;               // 1 if H2 terms exist, 0 if skipped (comb FMC)

    int N;
    int nrep;
};

// Allocate and initialize: disorder (shared), spins (per replica), energies
MCState mc_init(const SimConfig& cfg);

// Free all device memory
void mc_free(MCState& state);

// One full MC sweep: iterate over all pairs, each pair updates all replicas in parallel
void mc_sweep(MCState& state, const SimConfig& cfg);

// Copy energies and counters to host arrays (caller allocates)
void mc_get_results(const MCState& state, double* h_energies,
                    long long* h_accepted, long long* h_proposed);

// Copy all replica spins to host (caller allocates nrep*N cuDoubleComplex)
void mc_get_spins(const MCState& state, cuDoubleComplex* h_spins);

// Sweep using pre-set per-replica betas (call mc_set_betas first)
void mc_sweep_pt(MCState& state);

// Upload per-replica betas from host array to device
void mc_set_betas(MCState& state, const double* h_betas);

#endif
