#ifndef DISORDER_H
#define DISORDER_H

#include <cuComplex.h>
#include <cstdint>

// Number of interaction channels per quartet.
// For sorted (i<j<k<l) there are 3 ways to partition into
// an unconjugated pair and a conjugated pair:
//   ch 0: unconj {i,j}, conj {k,l}   FMC: |w_i+w_j-w_k-w_l| <= gamma
//   ch 1: unconj {i,l}, conj {j,k}   FMC: |w_i+w_l-w_j-w_k| <= gamma
//   ch 2: unconj {i,k}, conj {j,l}   FMC: |w_i+w_k-w_j-w_l| <= gamma
// The coupling g_{ijkl} is symmetric under permutations (same spatial
// overlap integral), so all 3 channels share the same coupling value.
// FMC independently selects which channels survive via a 3-bit mask
// per quartet (bit ch set ⇒ channel active).
#define N_G4_CHANNELS 3

// Generate the 2-body couplings g2[i*N + j] for i < j
// Stored as upper-triangular in a flat N*N array.
// Generated with unit variance; call rescale_g2 after FMC filter to set
// Var = J^2 * N / n_surviving.
void generate_g2(cuDoubleComplex* d_g2, int N, double J, unsigned long long seed);

// Generate the 4-body couplings g4[C(N,4)].
// One coupling per quartet (symmetric tensor: all 3 channels share it).
// Call init_g4_mask() to create the channel mask, then apply_fmc_g4()
// to filter it.  Rescale with rescale_g4() after counting survivors.
void generate_g4(cuDoubleComplex* d_g4, int N, double J, unsigned long long seed);

// Number of 4-body terms C(N,4)
inline long long n_quartets(int N) {
    return (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
}

// Total (quartet, channel) pairs: 3 * C(N,4).
// This is the maximum number of H4 terms (before FMC).
inline long long n_g4_total(int N) {
    return (long long)N_G4_CHANNELS * n_quartets(N);
}

// Number of 2-body terms C(N,2)
inline long long n_pairs(int N) {
    return (long long)N * (N - 1) / 2;
}

// Rescale g2 couplings: sigma = J * sqrt(N / n_surviving)
void rescale_g2(cuDoubleComplex* d_g2, int N, double J, long long n_surviving);

// Rescale g4 couplings over C(N,4) entries: sigma = J * sqrt(N / n_surviving)
// n_surviving = total active (quartet,channel) pairs = sum of popcount(mask).
void rescale_g4(cuDoubleComplex* d_g4, int N, double J, long long n_surviving);

// Shift the real part of non-zero couplings by +mean.
// Only entries with at least one non-zero component are shifted (respects FMC zeros).
void shift_mean_couplings(cuDoubleComplex* d_c, long long n, double mean);

// Apply FMC filter: zero out g2 pairs where |omega_i - omega_j| > gamma
void apply_fmc_g2(cuDoubleComplex* d_g2, int N, const double* d_omega, double gamma);

// Initialise g4 channel mask: all bits set (0x7 = all 3 channels active).
// d_mask must be pre-allocated with C(N,4) bytes.
void init_g4_mask(uint8_t* d_mask, int N);

// Apply FMC filter on the g4 channel mask.
// Clears mask bits for channels whose frequency condition is not met.
void apply_fmc_g4(uint8_t* d_mask, int N, const double* d_omega, double gamma);

// Count total active (quartet, channel) pairs = sum of popcount(mask[q]).
long long count_active_g4(const uint8_t* d_mask, int N);

#endif
