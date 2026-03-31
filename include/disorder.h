#ifndef DISORDER_H
#define DISORDER_H

#include <cuComplex.h>

// Generate the 2-body couplings g2[i*N + j] for i < j
// Stored as upper-triangular in a flat N*N array.
// Generated with unit variance; call rescale_g2 after FMC filter to set
// Var = J^2 * N / n_surviving.
void generate_g2(cuDoubleComplex* d_g2, int N, double J, unsigned long long seed);

// Generate the 4-body couplings g4.
// Stored as a flat array indexed by the sorted quadruplet (i,j,k,l) with i<j<k<l.
// Total entries = C(N,4).
// Generated with unit variance; call rescale_g4 after FMC filter to set
// Var = J^2 * N / n_surviving.
void generate_g4(cuDoubleComplex* d_g4, int N, double J, unsigned long long seed);

// Number of 4-body terms C(N,4)
inline long long n_quartets(int N) {
    return (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
}

// Number of 2-body terms C(N,2)
inline long long n_pairs(int N) {
    return (long long)N * (N - 1) / 2;
}

// Rescale g2 couplings: sigma = J * sqrt(N / n_surviving)
void rescale_g2(cuDoubleComplex* d_g2, int N, double J, long long n_surviving);

// Rescale g4 couplings: sigma = J * sqrt(N / n_surviving)
void rescale_g4(cuDoubleComplex* d_g4, int N, double J, long long n_surviving);

// Apply FMC filter: zero out g2 pairs where |omega_i - omega_j| > gamma
void apply_fmc_g2(cuDoubleComplex* d_g2, int N, const double* d_omega, double gamma);

// Apply FMC filter: zero out g4 quartets (i<j<k<l) where |omega_j - omega_i + omega_k - omega_l| > gamma
void apply_fmc_g4(cuDoubleComplex* d_g4, int N, const double* d_omega, double gamma);

#endif
