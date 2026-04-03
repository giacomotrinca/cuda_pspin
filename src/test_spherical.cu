// ===========================================================================
// Test 2: Spherical constraint preservation
//
// After init_spins_uniform(), verify  sum_k |a_k|^2 = N.
// After many MC sweeps, verify the constraint still holds for each replica.
// The pair-rotation move preserves |a_i|^2 + |a_j|^2, so the total
// sum |a|^2 = N must be an exact invariant (up to floating-point drift).
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "config.h"
#include "mc.h"
#include "hamiltonian.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

static double compute_norm_sq(const cuDoubleComplex* spins, int N) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double re = cuCreal(spins[i]);
        double im = cuCimag(spins[i]);
        s += re * re + im * im;
    }
    return s;
}

int main(int argc, char** argv) {
    int N    = 16;
    int nrep = 4;
    int sweeps = 500;
    double tol = 1e-8;

    if (argc > 1) N      = atoi(argv[1]);
    if (argc > 2) nrep   = atoi(argv[2]);
    if (argc > 3) sweeps = atoi(argv[3]);

    printf("=== Test: spherical constraint  N=%d  nrep=%d  sweeps=%d ===\n\n", N, nrep, sweeps);

    const char* outdir = getenv("TEST_OUTDIR");
    FILE* fdat = NULL;
    if (outdir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/spherical.dat", outdir);
        fdat = fopen(path, "w");
        if (fdat) fprintf(fdat, "# sweep max_err\n");
    }

    // Build a minimal SimConfig
    SimConfig cfg = default_config();
    cfg.N = N;
    cfg.nrep = nrep;
    cfg.T = 1.0;
    cfg.J = 2.0;
    cfg.J0 = 0.0;
    cfg.alpha = 0.5;
    cfg.alpha0 = 0.0;
    cfg.mc_iterations = sweeps;
    cfg.seed = 12345;
    cfg.verbose = 0;
    cfg.fmc_mode = 0;

    MCState state = mc_init(cfg);

    // Check constraint after init
    cuDoubleComplex* h_spins = new cuDoubleComplex[(long long)nrep * N];
    mc_get_spins(state, h_spins);

    int all_pass = 1;
    double max_err_init = 0.0;
    for (int r = 0; r < nrep; r++) {
        double norm_sq = compute_norm_sq(h_spins + (long long)r * N, N);
        double err = fabs(norm_sq - (double)N);
        if (err > max_err_init) max_err_init = err;
        if (err > tol) {
            printf("  INIT  rep %d: |a|^2 = %.12f  err = %.2e  FAIL\n", r, norm_sq, err);
            all_pass = 0;
        }
    }
    printf("  After init:   max |sum|a|^2 - N| = %.2e  %s\n",
           max_err_init, (max_err_init <= tol) ? "PASS" : "FAIL");
    if (fdat) fprintf(fdat, "0 %.12e\n", max_err_init);

    // Run sweeps, checking at regular intervals
    int check_interval = (sweeps >= 10) ? sweeps / 10 : 1;
    double max_err_sweep = 0.0;

    for (int s = 0; s < sweeps; s++) {
        mc_sweep(state, cfg);

        if ((s + 1) % check_interval == 0 || s == sweeps - 1) {
            mc_get_spins(state, h_spins);
            double checkpoint_err = 0.0;
            for (int r = 0; r < nrep; r++) {
                double norm_sq = compute_norm_sq(h_spins + (long long)r * N, N);
                double err = fabs(norm_sq - (double)N);
                if (err > checkpoint_err) checkpoint_err = err;
                if (err > max_err_sweep) max_err_sweep = err;
                if (err > tol) {
                    printf("  sweep %d  rep %d: |a|^2 = %.12f  err = %.2e  FAIL\n",
                           s + 1, r, norm_sq, err);
                    all_pass = 0;
                }
            }
            if (fdat) fprintf(fdat, "%d %.12e\n", s + 1, checkpoint_err);
        }
    }
    printf("  After %d sweeps: max |sum|a|^2 - N| = %.2e  %s\n",
           sweeps, max_err_sweep, (max_err_sweep <= tol) ? "PASS" : "FAIL");

    if (fdat) fclose(fdat);
    delete[] h_spins;
    mc_free(state);

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
