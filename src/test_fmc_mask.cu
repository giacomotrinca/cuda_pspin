// ===========================================================================
// Test 6: FMC mask correctness
//
// Generate random frequencies ω_i.  Compute the FMC g4 mask on GPU via
// init_g4_mask() + apply_fmc_g4().  Then compute the expected mask on CPU
// using the same frequency-matching conditions, and compare bit-by-bit.
//
// Channel conditions for sorted (ii < jj < kk < ll):
//   ch 0: |ω_ii + ω_jj - ω_kk - ω_ll| <= γ
//   ch 1: |ω_ii + ω_ll - ω_jj - ω_kk| <= γ
//   ch 2: |ω_ii + ω_kk - ω_jj - ω_ll| <= γ
//
// Also cross-check the g2 FMC filter.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand.h>

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

// CPU reference quartet decode (same formula as GPU)
static void decode_quartet(long long q, int* ii, int* jj, int* kk, int* ll) {
    int l = 3;
    while ((long long)l*(l-1)*(l-2)*(l-3)/24 <= q) l++;
    l--;
    long long rem = q - (long long)l*(l-1)*(l-2)*(l-3)/24;
    int k = 2;
    while ((long long)k*(k-1)*(k-2)/6 <= rem) k++;
    k--;
    rem -= (long long)k*(k-1)*(k-2)/6;
    int j = 1;
    while ((long long)j*(j-1)/2 <= rem) j++;
    j--;
    rem -= (long long)j*(j-1)/2;
    *ii = (int)rem; *jj = j; *kk = k; *ll = l;
}

int main(int argc, char** argv) {
    int N = 10;
    double gamma = 1.0;
    int fmc_mode = 1;   // 1 = comb, 2 = uniform

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) gamma = atof(argv[2]);
    if (argc > 3) fmc_mode = atoi(argv[3]);
    if (N < 4) N = 4;

    long long nq = n_quartets(N);
    long long np = n_pairs(N);

    printf("=== Test: FMC mask  N=%d  gamma=%.4f  mode=%d ===\n", N, gamma, fmc_mode);
    printf("    quartets=%lld  pairs=%lld\n\n", nq, np);

    // Generate frequencies on host
    double* h_omega = new double[N];
    if (fmc_mode == 1) {
        // Comb: ω_k = k
        for (int k = 0; k < N; k++) h_omega[k] = (double)k;
    } else {
        // Uniform random in [0, N)
        srand48(42);
        for (int k = 0; k < N; k++) h_omega[k] = drand48() * N;
    }

    // Upload frequencies to device
    double* d_omega;
    CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_omega, h_omega, N * sizeof(double), cudaMemcpyHostToDevice));

    // --- Test g4 mask ---
    uint8_t* d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, nq * sizeof(uint8_t)));
    init_g4_mask(d_mask, N);
    apply_fmc_g4(d_mask, N, d_omega, gamma);

    // Download GPU mask
    uint8_t* h_mask_gpu = new uint8_t[nq];
    CUDA_CHECK(cudaMemcpy(h_mask_gpu, d_mask, nq * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Compute CPU reference (single channel: first-pass-wins)
    uint8_t* h_mask_cpu = new uint8_t[nq];
    for (long long q = 0; q < nq; q++) {
        int ii, jj, kk, ll;
        decode_quartet(q, &ii, &jj, &kk, &ll);
        uint8_t m = 0;
        // ch 0: |ω_ii + ω_jj - ω_kk - ω_ll|
        if (fabs(h_omega[ii] + h_omega[jj] - h_omega[kk] - h_omega[ll]) <= gamma)
            m = (1u << 0);
        // ch 1: |ω_ii + ω_ll - ω_jj - ω_kk|
        else if (fabs(h_omega[ii] + h_omega[ll] - h_omega[jj] - h_omega[kk]) <= gamma)
            m = (1u << 1);
        // ch 2: |ω_ii + ω_kk - ω_jj - ω_ll|
        else if (fabs(h_omega[ii] + h_omega[kk] - h_omega[jj] - h_omega[ll]) <= gamma)
            m = (1u << 2);
        h_mask_cpu[q] = m;
    }

    int g4_mismatch = 0;
    for (long long q = 0; q < nq; q++) {
        if (h_mask_gpu[q] != h_mask_cpu[q]) {
            g4_mismatch++;
            if (g4_mismatch <= 5) {
                int ii, jj, kk, ll;
                decode_quartet(q, &ii, &jj, &kk, &ll);
                printf("  g4 mismatch q=%lld (%d,%d,%d,%d)  GPU=0x%x  CPU=0x%x\n",
                       q, ii, jj, kk, ll, h_mask_gpu[q], h_mask_cpu[q]);
            }
        }
    }

    // Count active
    long long gpu_active = 0, cpu_active = 0;
    for (long long q = 0; q < nq; q++) {
        gpu_active += __builtin_popcount(h_mask_gpu[q]);
        cpu_active += __builtin_popcount(h_mask_cpu[q]);
    }

    int g4_pass = (g4_mismatch == 0);
    printf("  g4 mask:  %d mismatches / %lld quartets  active: GPU=%lld CPU=%lld  %s\n",
           g4_mismatch, nq, gpu_active, cpu_active, g4_pass ? "PASS" : "FAIL");

    // --- Test g2 FMC filter ---
    // Generate dummy g2 with all entries = 1+0i, apply FMC, check which zeroed
    cuDoubleComplex* d_g2;
    CUDA_CHECK(cudaMalloc(&d_g2, (long long)N * N * sizeof(cuDoubleComplex)));
    cuDoubleComplex* h_g2_init = new cuDoubleComplex[(long long)N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_g2_init[i * N + j] = make_cuDoubleComplex(1.0, 0.0);
    CUDA_CHECK(cudaMemcpy(d_g2, h_g2_init, (long long)N * N * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    apply_fmc_g2(d_g2, N, d_omega, gamma);

    cuDoubleComplex* h_g2_gpu = new cuDoubleComplex[(long long)N * N];
    CUDA_CHECK(cudaMemcpy(h_g2_gpu, d_g2, (long long)N * N * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    int g2_mismatch = 0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
            bool gpu_zero = (cuCreal(h_g2_gpu[i * N + j]) == 0.0);
            bool cpu_zero = (i != j) && (fabs(h_omega[i] - h_omega[j]) > gamma);
            if (gpu_zero != cpu_zero) g2_mismatch++;
        }

    int g2_pass = (g2_mismatch == 0);
    printf("  g2 fmc:   %d mismatches / %lld pairs  %s\n",
           g2_mismatch, np, g2_pass ? "PASS" : "FAIL");

    // Cleanup
    delete[] h_omega;
    delete[] h_mask_gpu;
    delete[] h_mask_cpu;
    delete[] h_g2_init;
    delete[] h_g2_gpu;
    CUDA_CHECK(cudaFree(d_omega));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_g2));

    int all_pass = g4_pass && g2_pass;
    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
