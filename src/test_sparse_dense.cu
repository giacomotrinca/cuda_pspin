// ===========================================================================
// Test 8: Sparse MC self-consistency
//
// Checks the sparse (smoothed-cube) MC code:
//   1. Cube constraint: ∑|a_k|⁴ = N preserved after init and many sweeps.
//   2. ΔE tracking: stored energy matches independent recomputation from
//      the sparse quartet list + dense g2.
//   3. Acceptance at T→∞ is ≈ 1.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "config.h"
#include "mc_sparse.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// CPU: compute sum |a_k|^4 (smoothed-cube norm)
static double compute_cube_norm(const cuDoubleComplex* spins, int N) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double re = cuCreal(spins[i]);
        double im = cuCimag(spins[i]);
        double r2 = re * re + im * im;
        s += r2 * r2;
    }
    return s;
}

// CPU: full energy from dense g2 + sparse quartets
static double compute_energy_cpu(const cuDoubleComplex* spins, int N,
                                 const cuDoubleComplex* g2,
                                 const SparseQuartet* quartets, int nq,
                                 int h2_active)
{
    double E = 0.0;

    // H2
    if (h2_active) {
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) {
                cuDoubleComplex gij = g2[i * N + j];
                cuDoubleComplex ai  = spins[i];
                cuDoubleComplex aj_c = cuConj(spins[j]);
                cuDoubleComplex prod = cuCmul(gij, cuCmul(ai, aj_c));
                E += -cuCreal(prod);
            }
    }

    // H4 (sparse)
    for (int q = 0; q < nq; q++) {
        const SparseQuartet& sq = quartets[q];
        cuDoubleComplex ai = spins[sq.i], aj = spins[sq.j];
        cuDoubleComplex ak = spins[sq.k], al = spins[sq.l];
        cuDoubleComplex s0, s1, s2, s3;
        if (sq.ch == 0)      { s0 = ai; s1 = aj;         s2 = cuConj(ak); s3 = cuConj(al); }
        else if (sq.ch == 1) { s0 = ai; s1 = cuConj(aj); s2 = cuConj(ak); s3 = al; }
        else                 { s0 = ai; s1 = cuConj(aj); s2 = ak;         s3 = cuConj(al); }
        cuDoubleComplex prod = cuCmul(sq.g, cuCmul(cuCmul(s0, s1), cuCmul(s2, s3)));
        E += -cuCreal(prod);
    }
    return E;
}

int main(int argc, char** argv) {
    int N      = 10;
    int nrep   = 4;
    int sweeps = 300;

    if (argc > 1) N      = atoi(argv[1]);
    if (argc > 2) nrep   = atoi(argv[2]);
    if (argc > 3) sweeps = atoi(argv[3]);

    printf("=== Test: sparse MC self-consistency  N=%d  nrep=%d  sweeps=%d ===\n\n",
           N, nrep, sweeps);

    const char* outdir = getenv("TEST_OUTDIR");
    FILE* fdat = NULL;
    if (outdir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/sparse_dense.dat", outdir);
        fdat = fopen(path, "w");
        if (fdat) fprintf(fdat, "# sweep max_cube_err max_e_err\n");
    }

    int all_pass = 1;

    // ---- Part A: cube constraint + ΔE at finite T ----
    {
        printf("--- Part A: cube constraint + energy tracking (T=1.0) ---\n");
        SimConfig cfg = default_config();
        cfg.N = N; cfg.nrep = nrep; cfg.T = 1.0;
        cfg.J = 2.0; cfg.J0 = 0.0; cfg.alpha = 0.5; cfg.alpha0 = 0.0;
        cfg.seed = 54321; cfg.verbose = 0; cfg.fmc_mode = 0;

        MCStateSparse state = mc_sparse_init(cfg);

        // Download sparse quartets for CPU energy check
        int n_sq = state.n_quartets;
        SparseQuartet* h_quartets = new SparseQuartet[n_sq];
        CUDA_CHECK(cudaMemcpy(h_quartets, state.d_quartets,
                              n_sq * sizeof(SparseQuartet), cudaMemcpyDeviceToHost));

        // Download g2
        cuDoubleComplex* h_g2 = new cuDoubleComplex[(long long)N * N];
        CUDA_CHECK(cudaMemcpy(h_g2, state.d_g2,
                              (long long)N * N * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));

        cuDoubleComplex* h_spins = new cuDoubleComplex[(long long)nrep * N];
        double*   h_energies = new double[nrep];
        long long* h_acc     = new long long[nrep];
        long long* h_prop    = new long long[nrep];

        // Set betas
        double beta = 1.0 / cfg.T;
        double* h_betas = new double[nrep];
        for (int r = 0; r < nrep; r++) h_betas[r] = beta;
        mc_sparse_set_betas(state, h_betas);

        // Check constraint after init
        mc_sparse_get_spins(state, h_spins);
        double max_cube_err = 0.0;
        double cube_tol = 1e-8;
        for (int r = 0; r < nrep; r++) {
            double cn = compute_cube_norm(h_spins + (long long)r * N, N);
            double err = fabs(cn - (double)N);
            if (err > max_cube_err) max_cube_err = err;
        }
        printf("  After init:  max |sum|a|^4 - N| = %.2e  %s\n",
               max_cube_err, (max_cube_err <= cube_tol) ? "PASS" : "FAIL");
        if (max_cube_err > cube_tol) all_pass = 0;
        if (fdat) fprintf(fdat, "0 %.12e 0.0\n", max_cube_err);

        // Run sweeps
        double e_tol = 1e-3;
        double max_e_err = 0.0;
        double max_cube_sweep = 0.0;
        int check_interval = (sweeps >= 10) ? sweeps / 10 : 1;

        for (int s = 0; s < sweeps; s++) {
            mc_sparse_sweep_pt(state);

            if ((s + 1) % check_interval == 0 || s == sweeps - 1) {
                mc_sparse_get_results(state, h_energies, h_acc, h_prop);
                mc_sparse_get_spins(state, h_spins);

                double cp_cube_err = 0.0, cp_e_err = 0.0;
                for (int r = 0; r < nrep; r++) {
                    // Cube constraint
                    double cn = compute_cube_norm(h_spins + (long long)r * N, N);
                    double cerr = fabs(cn - (double)N);
                    if (cerr > cp_cube_err) cp_cube_err = cerr;
                    if (cerr > max_cube_sweep) max_cube_sweep = cerr;

                    // Energy recomputation
                    double E_cpu = compute_energy_cpu(
                        h_spins + (long long)r * N, N,
                        h_g2, h_quartets, n_sq, state.h2_active);
                    double E_stored = h_energies[r];
                    double ediff = fabs(E_cpu - E_stored);
                    if (ediff / N > cp_e_err) cp_e_err = ediff / N;
                    if (ediff / N > max_e_err) max_e_err = ediff / N;
                    if (ediff / N > e_tol) {
                        printf("  sweep %d  rep %d:  stored=%.6f  cpu=%.6f  diff/N=%.2e  FAIL\n",
                               s + 1, r, E_stored / N, E_cpu / N, ediff / N);
                        all_pass = 0;
                    }
                }
                if (fdat) fprintf(fdat, "%d %.12e %.12e\n", s + 1, cp_cube_err, cp_e_err);
            }
        }

        printf("  After %d sweeps:  max cube err = %.2e  %s\n",
               sweeps, max_cube_sweep, (max_cube_sweep <= cube_tol) ? "PASS" : "FAIL");
        if (max_cube_sweep > cube_tol) all_pass = 0;
        printf("  Energy tracking:  max |E_stored - E_cpu|/N = %.2e  %s\n",
               max_e_err, (max_e_err <= e_tol) ? "PASS" : "FAIL");
        if (max_e_err > e_tol) all_pass = 0;

        delete[] h_quartets;
        delete[] h_g2;
        delete[] h_spins;
        delete[] h_energies;
        delete[] h_acc;
        delete[] h_prop;
        delete[] h_betas;
        mc_sparse_free(state);
    }

    // ---- Part B: acceptance at T → ∞ ----
    {
        printf("\n--- Part B: infinite temperature acceptance ---\n");
        SimConfig cfg = default_config();
        cfg.N = N; cfg.nrep = nrep; cfg.T = 1e30;
        cfg.J = 2.0; cfg.J0 = 0.0; cfg.alpha = 0.5; cfg.alpha0 = 0.0;
        cfg.seed = 98765; cfg.verbose = 0; cfg.fmc_mode = 0;

        MCStateSparse state = mc_sparse_init(cfg);

        double beta = 1.0 / cfg.T;
        double* h_betas = new double[nrep];
        for (int r = 0; r < nrep; r++) h_betas[r] = beta;
        mc_sparse_set_betas(state, h_betas);

        for (int s = 0; s < 200; s++)
            mc_sparse_sweep_pt(state);

        double*   h_e = new double[nrep];
        long long* h_a = new long long[nrep];
        long long* h_p = new long long[nrep];
        mc_sparse_get_results(state, h_e, h_a, h_p);

        double min_rate = 1.0;
        for (int r = 0; r < nrep; r++) {
            double rate = (h_p[r] > 0) ? (double)h_a[r] / h_p[r] : 0.0;
            if (rate < min_rate) min_rate = rate;
        }
        double acc_tol = 0.01;
        printf("  Acceptance: min = %.4f  (need > %.4f)  %s\n",
               min_rate, 1.0 - acc_tol, (min_rate >= 1.0 - acc_tol) ? "PASS" : "FAIL");
        if (min_rate < 1.0 - acc_tol) all_pass = 0;

        delete[] h_betas;
        delete[] h_e;
        delete[] h_a;
        delete[] h_p;
        mc_sparse_free(state);
    }

    if (fdat) fclose(fdat);
    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
