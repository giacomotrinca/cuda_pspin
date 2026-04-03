// ===========================================================================
// Test 4: Infinite-temperature acceptance rate
//
// At β = 0 every move is accepted (Metropolis: exp(0) = 1).
// Run many sweeps at T → ∞, verify acceptance ≈ 1.
// Also verify that the mean energy is close to 0 at high temperature
// (for zero-mean Gaussian couplings and J0 = 0).
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "config.h"
#include "mc.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    int N      = 12;
    int nrep   = 8;
    int sweeps = 500;
    double acc_tol = 0.001;   // acceptance must be > 1 - acc_tol

    if (argc > 1) N      = atoi(argv[1]);
    if (argc > 2) nrep   = atoi(argv[2]);
    if (argc > 3) sweeps = atoi(argv[3]);

    printf("=== Test: infinite temperature  N=%d  nrep=%d  sweeps=%d ===\n\n",
           N, nrep, sweeps);

    const char* outdir = getenv("TEST_OUTDIR");
    FILE* fdat = NULL;
    if (outdir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/inf_temp.dat", outdir);
        fdat = fopen(path, "w");
        if (fdat) fprintf(fdat, "# replica acceptance_rate energy_per_N\n");
    }

    SimConfig cfg = default_config();
    cfg.N    = N;
    cfg.nrep = nrep;
    cfg.T    = 1e30;      // effectively β = 0
    cfg.J    = 2.0;
    cfg.J0   = 0.0;
    cfg.alpha = 0.5;
    cfg.alpha0 = 0.0;
    cfg.mc_iterations = sweeps;
    cfg.seed = 31415;
    cfg.verbose = 0;
    cfg.fmc_mode = 0;

    MCState state = mc_init(cfg);

    // Run sweeps
    for (int s = 0; s < sweeps; s++)
        mc_sweep(state, cfg);

    // Read counters
    double*   h_energies = new double[nrep];
    long long* h_acc     = new long long[nrep];
    long long* h_prop    = new long long[nrep];
    mc_get_results(state, h_energies, h_acc, h_prop);

    int all_pass = 1;

    // Check acceptance rates
    double min_rate = 1.0;
    for (int r = 0; r < nrep; r++) {
        double rate = (h_prop[r] > 0) ? (double)h_acc[r] / h_prop[r] : 0.0;
        if (rate < min_rate) min_rate = rate;
        if (fdat) fprintf(fdat, "%d %.12f %.12f\n", r, rate, h_energies[r] / N);
        if (rate < 1.0 - acc_tol) {
            printf("  rep %d: acceptance = %.6f  FAIL\n", r, rate);
            all_pass = 0;
        }
    }
    printf("  Acceptance:  min = %.6f  (need > %.6f)  %s\n",
           min_rate, 1.0 - acc_tol, (min_rate >= 1.0 - acc_tol) ? "PASS" : "FAIL");

    // At β → 0, <E/N> → 0 for zero-mean couplings.
    // With finite sweeps the energy is a random walk; check |<E/N>| < moderate bound.
    double mean_eN = 0.0;
    for (int r = 0; r < nrep; r++) mean_eN += h_energies[r] / N;
    mean_eN /= nrep;
    double e_bound = 5.0;   // generous bound
    int e_pass = (fabs(mean_eN) < e_bound);
    printf("  <E/N>  = %.6f  (expect ~ 0, bound %.1f)  %s\n",
           mean_eN, e_bound, e_pass ? "PASS" : "FAIL");
    if (!e_pass) all_pass = 0;

    if (fdat) fclose(fdat);
    delete[] h_energies;
    delete[] h_acc;
    delete[] h_prop;
    mc_free(state);

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
