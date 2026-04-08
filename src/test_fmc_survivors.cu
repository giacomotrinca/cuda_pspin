// ===========================================================================
// Test: FMC survivor statistics
//
// For a given N and gamma, generates n_samples random frequency
// realizations (uniform in [0, 1), sorted — same as mc_init) and counts
// how many pairs and quartets survive the FMC filter.
//
// Uses existing GPU functions: init_g4_mask, apply_fmc_g4, count_active_g4.
// Pair counting is done on CPU (O(N^2), negligible for N <= 120).
//
// Output (one line per sample):
//   sample_id   pairs_active   pairs_total   quartets_active   quartets_total
//
// Usage:
//   test_fmc_survivors --N=18 --gamma=1.0 --samples=100 [--seed=42]
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#include "disorder.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Same splitmix64 used in mc.cu for frequency generation
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main(int argc, char** argv) {
    int N = 18;
    double gamma = 1.0;
    int n_samples = 100;
    unsigned long long seed = 42;

    for (int a = 1; a < argc; a++) {
        if (sscanf(argv[a], "--N=%d",       &N)         == 1) continue;
        if (sscanf(argv[a], "-N=%d",        &N)         == 1) continue;
        if (sscanf(argv[a], "--gamma=%lf",  &gamma)     == 1) continue;
        if (sscanf(argv[a], "-gamma=%lf",   &gamma)     == 1) continue;
        if (sscanf(argv[a], "--samples=%d", &n_samples) == 1) continue;
        if (sscanf(argv[a], "-samples=%d",  &n_samples) == 1) continue;
        if (sscanf(argv[a], "--seed=%llu",  &seed)      == 1) continue;
        if (sscanf(argv[a], "-seed=%llu",   &seed)      == 1) continue;
    }

    if (N < 4) { fprintf(stderr, "N must be >= 4\n"); return 1; }

    long long nq = n_quartets(N);
    long long np = n_pairs(N);

    // Allocate device arrays
    double*  d_omega;
    uint8_t* d_mask;
    CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mask,  nq * sizeof(uint8_t)));

    double* h_omega = new double[N];

    printf("# N=%d  gamma=%.6f  samples=%d  seed=%llu\n", N, gamma, n_samples, seed);
    printf("# sample  pairs_active  pairs_total  quartets_active  quartets_total\n");

    for (int s = 0; s < n_samples; s++) {
        // Generate random frequencies uniform in [0, 1), sorted
        // (same as mc_init fmc_mode==2: splitmix64 -> double -> sort)
        uint64_t freq_state = seed + 3000ULL + (uint64_t)s * 7919ULL;
        for (int k = 0; k < N; k++) {
            uint64_t z = splitmix64(&freq_state);
            h_omega[k] = (double)(z >> 11) / (double)(1ULL << 53);
        }
        std::sort(h_omega, h_omega + N);

        CUDA_CHECK(cudaMemcpy(d_omega, h_omega, N * sizeof(double),
                              cudaMemcpyHostToDevice));

        // --- Quartets: init mask -> apply FMC -> count ---
        init_g4_mask(d_mask, N);
        apply_fmc_g4(d_mask, N, d_omega, gamma);
        long long active_q = count_active_g4(d_mask, N);

        // --- Pairs: count on CPU (|omega_i - omega_j| <= gamma, including i=j) ---
        long long active_p = 0;
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
                if (fabs(h_omega[i] - h_omega[j]) <= gamma)
                    active_p++;

        printf("%d  %lld  %lld  %lld  %lld\n", s, active_p, np, active_q, nq);
    }

    delete[] h_omega;
    CUDA_CHECK(cudaFree(d_omega));
    CUDA_CHECK(cudaFree(d_mask));

    return 0;
}
