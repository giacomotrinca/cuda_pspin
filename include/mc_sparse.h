#ifndef MC_SPARSE_H
#define MC_SPARSE_H

#include "config.h"
#include <cuComplex.h>
#include <curand_kernel.h>

// A single sparse H4 coupling entry: one (quartet, channel) pair
struct SparseQuartet {
    int i, j, k, l;        // sorted indices: i < j < k < l
    int ch;                 // interaction channel: 0, 1, or 2
    cuDoubleComplex g;      // coupling value (rescaled)
};

struct MCStateSparse {
    // Shared disorder
    cuDoubleComplex* d_g2;       // device: 2-body couplings [N*N]
    SparseQuartet*   d_quartets; // device: sparse H4 couplings [n_quartets]
    int              n_quartets; // number of selected sparse quartets

    // Per-replica data
    cuDoubleComplex* d_spins;    // device: spin configurations [nrep * N]
    curandStatePhilox4_32_10_t* d_rng;  // device: RNG states [nrep]
    double*    d_energies;       // device: energies  [nrep]
    long long* d_accepted;       // device: accepted  [nrep]
    long long* d_proposed;       // device: proposed  [nrep]
    double*    d_betas;          // device: per-replica inverse temps [nrep]

    double*   h_omega;           // host: frequencies [N] (NULL if FC)
    long long n_pairs_active;    // active pairs after FMC
    long long n_quart_active;    // total non-zero quartet entries after FMC (before subsampling)
    int       h2_active;         // 1 if H2 terms exist, 0 if skipped

    int N;
    int nrep;
};

// Allocate and initialize:
//  - disorder (g2 dense, g4 -> FMC -> sample N quartets)
//  - spins on smoothed-cube surface  sum|a_k|^4 = N
//  - initial energies
MCStateSparse mc_sparse_init(const SimConfig& cfg);

// Free all device (and host) memory
void mc_sparse_free(MCStateSparse& state);

// One MC sweep using pre-set per-replica betas (for parallel tempering)
void mc_sparse_sweep_pt(MCStateSparse& state);

// Upload per-replica betas from host array
void mc_sparse_set_betas(MCStateSparse& state, const double* h_betas);

// Copy energies and counters to host arrays
void mc_sparse_get_results(const MCStateSparse& state, double* h_energies,
                           long long* h_accepted, long long* h_proposed);

// Copy all replica spins to host
void mc_sparse_get_spins(const MCStateSparse& state, cuDoubleComplex* h_spins);

#endif
