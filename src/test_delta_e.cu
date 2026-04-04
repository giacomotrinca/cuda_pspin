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

// CPU-side energy computation (gold standard)
static void compute_energy_cpu(const cuDoubleComplex* spins, int N,
                               const cuDoubleComplex* g2,
                               const cuDoubleComplex* g4,
                               const uint8_t* g4_mask,
                               double* out_h2, double* out_h4) {
    double E_h2 = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            cuDoubleComplex gij = g2[i * N + j];
            cuDoubleComplex prod = cuCmul(gij, cuCmul(spins[i], cuConj(spins[j])));
            E_h2 += -cuCreal(prod);
        }

    double E_h4 = 0.0;
    long long nq = n_quartets(N);
    for (long long q = 0; q < nq; q++) {
        uint8_t mask = g4_mask[q];
        if (!mask) continue;
        cuDoubleComplex gq = g4[q];

        // Decode q -> (ii, jj, kk, ll)
        int ll = 3;
        while ((long long)ll*(ll-1)*(ll-2)*(ll-3)/24 <= q) ll++;
        ll--;
        long long rem = q - (long long)ll*(ll-1)*(ll-2)*(ll-3)/24;
        int kk = 2;
        while ((long long)kk*(kk-1)*(kk-2)/6 <= rem) kk++;
        kk--;
        rem -= (long long)kk*(kk-1)*(kk-2)/6;
        int jj = 1;
        while ((long long)jj*(jj-1)/2 <= rem) jj++;
        jj--;
        rem -= (long long)jj*(jj-1)/2;
        int ii = (int)rem;

        cuDoubleComplex ai = spins[ii], aj = spins[jj];
        cuDoubleComplex ak = spins[kk], al = spins[ll];

        for (int ch = 0; ch < 3; ch++) {
            if (!(mask & (1 << ch))) continue;
            cuDoubleComplex s0, s1, s2, s3;
            if (ch == 0)      { s0=ai; s1=aj;         s2=cuConj(ak); s3=cuConj(al); }
            else if (ch == 1) { s0=ai; s1=cuConj(aj); s2=cuConj(ak); s3=al; }
            else              { s0=ai; s1=cuConj(aj); s2=ak;         s3=cuConj(al); }
            cuDoubleComplex prod = cuCmul(gq, cuCmul(cuCmul(s0,s1), cuCmul(s2,s3)));
            E_h4 += -cuCreal(prod);
        }
    }
    *out_h2 = E_h2;
    *out_h4 = E_h4;
}

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

    // ---- Sweep-0 diagnostic: CPU vs init_energy vs compute_energy ----
    {
        mc_get_results(state, h_energies, h_acc, h_prop);

        // Download all arrays for CPU computation
        long long nq = n_quartets(N);
        cuDoubleComplex* h_spins = new cuDoubleComplex[(long long)nrep * N];
        cuDoubleComplex* h_g2    = new cuDoubleComplex[(long long)N * N];
        cuDoubleComplex* h_g4    = new cuDoubleComplex[nq];
        uint8_t*         h_g4m   = new uint8_t[nq];
        CUDA_CHECK(cudaMemcpy(h_spins, state.d_spins,
                              (long long)nrep * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_g2, state.d_g2,
                              (long long)N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_g4, state.d_g4,
                              nq * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_g4m, state.d_g4_mask,
                              nq * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        printf("  ---- Sweep-0 diagnostic (CPU vs GPU) ----\n");
        for (int r = 0; r < nrep; r++) {
            double h2_cpu, h4_cpu;
            compute_energy_cpu(h_spins + (long long)r * N, N, h_g2, h_g4, h_g4m,
                               &h2_cpu, &h4_cpu);
            double E_cpu  = h2_cpu + h4_cpu;
            double E_init = h_energies[r];
            cuDoubleComplex* d_rep = state.d_spins + (long long)r * N;
            double E_gpu  = compute_energy(d_rep, state.d_g2, state.d_g4,
                                           state.d_g4_mask, N);

            printf("  rep %d:  H2_cpu=%.8f  H4_cpu=%.8f  E_cpu=%.8f\n",
                   r, h2_cpu / N, h4_cpu / N, E_cpu / N);
            printf("          E_init=%.8f  E_gpu=%.8f\n",
                   E_init / N, E_gpu / N);
            printf("          init-cpu=%.2e  gpu-cpu=%.2e  init-gpu=%.2e\n",
                   (E_init - E_cpu) / N, (E_gpu - E_cpu) / N, (E_init - E_gpu) / N);

            double diff_init = fabs(E_init - E_cpu) / N;
            double diff_gpu  = fabs(E_gpu  - E_cpu) / N;
            if (diff_init > tol || diff_gpu > tol) {
                printf("          FAIL\n");
                all_pass = 0;
            } else {
                printf("          ok\n");
            }
        }
        printf("\n");

        delete[] h_spins;
        delete[] h_g2;
        delete[] h_g4;
        delete[] h_g4m;
    }

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
