#ifndef DISORDER_H
#define DISORDER_H

#include <cuComplex.h>

// Number of interaction channels per quartet.
// For sorted (i<j<k<l) there are 3 ways to partition into
// an unconjugated pair and a conjugated pair:
//   ch 0: unconj {i,j}, conj {k,l}   FMC: |w_i+w_j-w_k-w_l| <= gamma
//   ch 1: unconj {i,l}, conj {j,k}   FMC: |w_i+w_l-w_j-w_k| <= gamma
//   ch 2: unconj {i,k}, conj {j,l}   FMC: |w_i+w_k-w_j-w_l| <= gamma
// Each channel has an independent coupling.
#define N_G4_CHANNELS 3

// Generate the 2-body couplings g2[i*N + j] for i < j
// Stored as upper-triangular in a flat N*N array.
// Generated with unit variance; call rescale_g2 after FMC filter to set
// Var = J^2 * N / n_surviving.
void generate_g2(cuDoubleComplex* d_g2, int N, double J, unsigned long long seed);

// Generate the 4-body couplings g4.
// Layout: [ch0: C(N,4) | ch1: C(N,4) | ch2: C(N,4)]
// Total entries = 3 * C(N,4).  Each channel gets independent N(0,1) couplings.
// Call rescale_g4 after FMC filter with the total surviving count.
void generate_g4(cuDoubleComplex* d_g4, int N, double J, unsigned long long seed);

// Number of 4-body terms C(N,4)
inline long long n_quartets(int N) {
    return (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
}

// Total g4 entries: 3 channels * C(N,4)
inline long long n_g4_total(int N) {
    return (long long)N_G4_CHANNELS * n_quartets(N);
}

// Number of 2-body terms C(N,2)
inline long long n_pairs(int N) {
    return (long long)N * (N - 1) / 2;
}

// Rescale g2 couplings: sigma = J * sqrt(N / n_surviving)
void rescale_g2(cuDoubleComplex* d_g2, int N, double J, long long n_surviving);

// Rescale g4 couplings over all 3*C(N,4) entries: sigma = J * sqrt(N / n_surviving)
void rescale_g4(cuDoubleComplex* d_g4, int N, double J, long long n_surviving);

// Apply FMC filter: zero out g2 pairs where |omega_i - omega_j| > gamma
void apply_fmc_g2(cuDoubleComplex* d_g2, int N, const double* d_omega, double gamma);

// Apply FMC filter: 3-channel filter on g4 [3*C(N,4)] entries
// Each channel uses its own conservation condition.
void apply_fmc_g4(cuDoubleComplex* d_g4, int N, const double* d_omega, double gamma);

#endif
