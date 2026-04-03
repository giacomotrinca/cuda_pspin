// ===========================================================================
// Test 9: Mean coupling shift (J0)
//
// Verify that shift_mean_couplings() correctly adds J0 * N / n_surviving
// to the real part of every non-zero coupling entry.
//
// Procedure:
//   1. Generate g2 and g4 with known seed.
//   2. Optionally apply FMC to get a sparse mask.
//   3. Rescale couplings (standard pipeline).
//   4. Compute pre-shift mean of non-zero entries.
//   5. Apply shift_mean_couplings().
//   6. Compute post-shift mean.
//   7. Verify Δmean ≈ J0 * N / n_surviving.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuComplex.h>

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

// Compute mean of real parts of non-zero entries
static double mean_nonzero_re(const cuDoubleComplex* h, long long n) {
    double sum = 0.0;
    long long count = 0;
    for (long long i = 0; i < n; i++) {
        if (cuCreal(h[i]) != 0.0 || cuCimag(h[i]) != 0.0) {
            sum += cuCreal(h[i]);
            count++;
        }
    }
    return (count > 0) ? sum / count : 0.0;
}

static long long count_nonzero(const cuDoubleComplex* h, long long n) {
    long long c = 0;
    for (long long i = 0; i < n; i++)
        if (cuCreal(h[i]) != 0.0 || cuCimag(h[i]) != 0.0) c++;
    return c;
}

int main(int argc, char** argv) {
    int N      = 12;
    double J   = 2.0;
    double J0  = 0.5;
    double tol = 1e-10;

    if (argc > 1) N  = atoi(argv[1]);
    if (argc > 2) J0 = atof(argv[2]);
    if (N < 4) N = 4;

    printf("=== Test: mean coupling shift  N=%d  J=%.2f  J0=%.4f ===\n\n", N, J, J0);

    int all_pass = 1;

    // ======================== Test g2 shift ========================
    {
        printf("--- g2 shift ---\n");
        long long n = (long long)N * N;
        cuDoubleComplex* d_g2;
        CUDA_CHECK(cudaMalloc(&d_g2, n * sizeof(cuDoubleComplex)));
        generate_g2(d_g2, N, J, 42);

        // g2 is fully-connected (no FMC): all off-diag are non-zero
        long long n_surviving = n_pairs(N);  // N*(N-1)/2

        // Rescale
        rescale_g2(d_g2, N, J, n_surviving);

        // Download before shift
        cuDoubleComplex* h_pre = new cuDoubleComplex[n];
        CUDA_CHECK(cudaMemcpy(h_pre, d_g2, n * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double mean_pre = mean_nonzero_re(h_pre, n);

        // Apply shift
        double expected_shift = J0 * (double)N / (double)n_surviving;
        shift_mean_couplings(d_g2, n, expected_shift);

        // Download after shift
        cuDoubleComplex* h_post = new cuDoubleComplex[n];
        CUDA_CHECK(cudaMemcpy(h_post, d_g2, n * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double mean_post = mean_nonzero_re(h_post, n);

        double actual_shift = mean_post - mean_pre;
        double err = fabs(actual_shift - expected_shift);

        printf("  n_surviving = %lld\n", n_surviving);
        printf("  expected shift = %.12f\n", expected_shift);
        printf("  actual shift   = %.12f\n", actual_shift);
        printf("  error          = %.2e  %s\n", err, (err < tol) ? "PASS" : "FAIL");
        if (err >= tol) all_pass = 0;

        // Verify zero entries (diagonal) stayed zero
        int diag_fail = 0;
        for (int i = 0; i < N; i++) {
            if (cuCreal(h_post[i * N + i]) != 0.0 || cuCimag(h_post[i * N + i]) != 0.0) {
                diag_fail++;
            }
        }
        printf("  diagonal zeros preserved: %s\n", diag_fail == 0 ? "PASS" : "FAIL");
        if (diag_fail) all_pass = 0;

        delete[] h_pre;
        delete[] h_post;
        CUDA_CHECK(cudaFree(d_g2));
    }

    // ======================== Test g4 shift ========================
    {
        printf("\n--- g4 shift ---\n");
        long long nq = n_quartets(N);
        cuDoubleComplex* d_g4;
        CUDA_CHECK(cudaMalloc(&d_g4, nq * sizeof(cuDoubleComplex)));
        generate_g4(d_g4, N, J, 42);

        // Fully-connected: all quartets active with 3 channels
        long long n_surviving = n_g4_total(N);

        rescale_g4(d_g4, N, J, n_surviving);

        // Download before shift
        cuDoubleComplex* h_pre = new cuDoubleComplex[nq];
        CUDA_CHECK(cudaMemcpy(h_pre, d_g4, nq * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double mean_pre = mean_nonzero_re(h_pre, nq);

        // Apply shift
        double expected_shift = J0 * (double)N / (double)n_surviving;
        shift_mean_couplings(d_g4, nq, expected_shift);

        cuDoubleComplex* h_post = new cuDoubleComplex[nq];
        CUDA_CHECK(cudaMemcpy(h_post, d_g4, nq * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double mean_post = mean_nonzero_re(h_post, nq);

        double actual_shift = mean_post - mean_pre;
        double err = fabs(actual_shift - expected_shift);

        printf("  n_surviving = %lld\n", n_surviving);
        printf("  expected shift = %.12f\n", expected_shift);
        printf("  actual shift   = %.12f\n", actual_shift);
        printf("  error          = %.2e  %s\n", err, (err < tol) ? "PASS" : "FAIL");
        if (err >= tol) all_pass = 0;

        delete[] h_pre;
        delete[] h_post;
        CUDA_CHECK(cudaFree(d_g4));
    }

    // ==================== Test g4 with FMC mask ====================
    {
        printf("\n--- g4 shift with FMC (comb, gamma=1.0) ---\n");
        long long nq = n_quartets(N);

        // Generate g4
        cuDoubleComplex* d_g4;
        CUDA_CHECK(cudaMalloc(&d_g4, nq * sizeof(cuDoubleComplex)));
        generate_g4(d_g4, N, J, 42);

        // Apply FMC mask
        uint8_t* d_mask;
        CUDA_CHECK(cudaMalloc(&d_mask, nq * sizeof(uint8_t)));
        init_g4_mask(d_mask, N);

        double* h_omega = new double[N];
        for (int k = 0; k < N; k++) h_omega[k] = (double)k;  // comb
        double* d_omega;
        CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_omega, h_omega, N * sizeof(double), cudaMemcpyHostToDevice));
        apply_fmc_g4(d_mask, N, d_omega, 1.0);

        long long n_surviving = count_active_g4(d_mask, N);

        // Zero out masked entries in g4  (simulate what mc_init does NOT for g4;
        // the couplings themselves don't get zeroed, but shift is applied to ALL entries).
        // shift_mean_couplings shifts all non-zero entries including masked ones.
        // In the real code, masked quartets are still in g4[] but their mask bit is 0.
        // The shift is applied to the g4 array (all C(N,4) entries regardless of mask).
        // So n_surviving here is the total number of active (quartet, channel) pairs,
        // but the shift is per-entry, not per-(quartet,channel).

        // For g4, the shift formula in mc_init is:
        //   mean_g4 = J4_0 * N / n_quart_active;
        //   shift_mean_couplings(d_g4, n_quartets(N), mean_g4);
        // So the denominator is n_quart_active (active channels across all quartets),
        // but the shift is applied to ALL nq entries that are non-zero.
        // This means some entries that are masked (no active channel) still get shifted.

        rescale_g4(d_g4, N, J, n_surviving);

        cuDoubleComplex* h_pre = new cuDoubleComplex[nq];
        CUDA_CHECK(cudaMemcpy(h_pre, d_g4, nq * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        long long nz_pre = count_nonzero(h_pre, nq);
        double mean_pre = mean_nonzero_re(h_pre, nq);

        double expected_shift = J0 * (double)N / (double)n_surviving;
        shift_mean_couplings(d_g4, nq, expected_shift);

        cuDoubleComplex* h_post = new cuDoubleComplex[nq];
        CUDA_CHECK(cudaMemcpy(h_post, d_g4, nq * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double mean_post = mean_nonzero_re(h_post, nq);

        double actual_shift = mean_post - mean_pre;
        double err = fabs(actual_shift - expected_shift);

        printf("  C(N,4)      = %lld\n", nq);
        printf("  active ch   = %lld\n", n_surviving);
        printf("  non-zero g4 = %lld\n", nz_pre);
        printf("  expected shift = %.12f\n", expected_shift);
        printf("  actual shift   = %.12f\n", actual_shift);
        printf("  error          = %.2e  %s\n", err, (err < tol) ? "PASS" : "FAIL");
        if (err >= tol) all_pass = 0;

        delete[] h_omega;
        delete[] h_pre;
        delete[] h_post;
        CUDA_CHECK(cudaFree(d_g4));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_omega));
    }

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
