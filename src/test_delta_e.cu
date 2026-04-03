// ===========================================================================
// Test 3: Delta-E consistency
//
// The MC sweep kernel accumulates the energy via ΔE each time a move is
// accepted.  After one or more sweeps, the stored energy (d_energies)
// must agree with a full independent recomputation via compute_energy().
// Any drift signals a bug in the ΔE kernel.
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

int main(int argc, char** argv) {
    int N      = 10;
    int nrep   = 4;
    int sweeps = 200;
    double tol = 1e-4;   // absolute tolerance on E/N

    if (argc > 1) N      = atoi(argv[1]);
    if (argc > 2) nrep   = atoi(argv[2]);
    if (argc > 3) sweeps = atoi(argv[3]);

    printf("=== Test: delta-E consistency  N=%d  nrep=%d  sweeps=%d ===\n\n", N, nrep, sweeps);

    const char* outdir = getenv("TEST_OUTDIR");
    FILE* fdat = NULL;
    if (outdir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/delta_e.dat", outdir);
        fdat = fopen(path, "w");
        if (fdat) fprintf(fdat, "# sweep max_rel_err\n");
    }

    SimConfig cfg = default_config();
    cfg.N = N;
    cfg.nrep = nrep;
    cfg.T = 1.0;
    cfg.J = 2.0;
    cfg.J0 = 0.0;
    cfg.alpha = 0.5;
    cfg.alpha0 = 0.0;
    cfg.mc_iterations = sweeps;
    cfg.seed = 77777;
    cfg.verbose = 0;
    cfg.fmc_mode = 0;

    MCState state = mc_init(cfg);

    double* h_energies  = new double[nrep];
    long long* h_acc    = new long long[nrep];
    long long* h_prop   = new long long[nrep];

    int all_pass = 1;
    int check_interval = (sweeps >= 10) ? sweeps / 10 : 1;
    double max_rel_err = 0.0;

    for (int s = 0; s < sweeps; s++) {
        mc_sweep(state, cfg);

        if ((s + 1) % check_interval == 0 || s == sweeps - 1) {
            // Get stored energies
            mc_get_results(state, h_energies, h_acc, h_prop);

            // Recompute full energy for each replica
            double checkpoint_err = 0.0;
            for (int r = 0; r < nrep; r++) {
                cuDoubleComplex* d_rep = state.d_spins + (long long)r * N;
                double E_full = compute_energy(d_rep, state.d_g2, state.d_g4,
                                               state.d_g4_mask, N);
                double E_stored = h_energies[r];
                double diff = fabs(E_full - E_stored);
                double scale = fabs(E_full) > 1.0 ? fabs(E_full) : 1.0;
                double rel = diff / scale;
                if (rel > checkpoint_err) checkpoint_err = rel;
                if (rel > max_rel_err) max_rel_err = rel;

                if (diff / N > tol) {
                    printf("  sweep %d  rep %d:  stored=%.8f  full=%.8f  diff/N=%.2e  FAIL\n",
                           s + 1, r, E_stored / N, E_full / N, diff / N);
                    all_pass = 0;
                }
            }
            if (fdat) fprintf(fdat, "%d %.12e\n", s + 1, checkpoint_err);
        }
    }

    printf("  After %d sweeps:  max |E_stored - E_full| / max(|E|,1) = %.2e  %s\n",
           sweeps, max_rel_err, all_pass ? "PASS" : "FAIL");

    if (fdat) fclose(fdat);
    delete[] h_energies;
    delete[] h_acc;
    delete[] h_prop;
    mc_free(state);

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
