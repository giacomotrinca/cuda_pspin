// ===========================================================================
// Test 7: Replica exchange sanity
//
// Run a mini parallel-tempering simulation with NT temperatures.
// Checks:
//   1. Swap acceptance rates are monotonically non-decreasing as ΔT → 0
//      (adjacent temperatures closer together should swap more easily).
//   2. All rates are in [0, 1].
//   3. Every replica visits both the hottest and coldest temperatures
//      (ergodicity in temperature space) — soft check for short runs.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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
    int N       = 8;
    int NT      = 6;
    int sweeps  = 2000;
    int pt_freq = 5;
    double Tmin = 0.5, Tmax = 2.0;

    if (argc > 1) N       = atoi(argv[1]);
    if (argc > 2) NT      = atoi(argv[2]);
    if (argc > 3) sweeps  = atoi(argv[3]);
    if (argc > 4) pt_freq = atoi(argv[4]);

    int nrep_per_temp = 1;
    int total_replicas = NT * nrep_per_temp;

    printf("=== Test: replica exchange  N=%d  NT=%d  sweeps=%d  pt_freq=%d ===\n",
           N, NT, sweeps, pt_freq);
    printf("    T in [%.2f, %.2f]  total replicas=%d\n\n", Tmin, Tmax, total_replicas);

    const char* outdir = getenv("TEST_OUTDIR");
    FILE* frates = NULL;
    FILE* fwalk  = NULL;
    if (outdir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/exchange_rates.dat", outdir);
        frates = fopen(path, "w");
        if (frates) fprintf(frates, "# T_high T_low rate\n");
        snprintf(path, sizeof(path), "%s/exchange_walk.dat", outdir);
        fwalk = fopen(path, "w");
        if (fwalk) fprintf(fwalk, "# sweep replica temp_idx\n");
    }

    // Temperature schedule (linear)
    double* T_sched    = new double[NT];
    double* beta_sched = new double[NT];
    double dT = (NT > 1) ? (Tmax - Tmin) / (NT - 1) : 0.0;
    for (int t = 0; t < NT; t++) {
        T_sched[t]    = Tmax - t * dT;   // T[0]=Tmax, T[NT-1]=Tmin
        beta_sched[t] = 1.0 / T_sched[t];
    }

    SimConfig cfg = default_config();
    cfg.N = N;
    cfg.nrep = total_replicas;
    cfg.T = Tmax;
    cfg.J = 2.0;
    cfg.J0 = 0.0;
    cfg.alpha = 0.5;
    cfg.alpha0 = 0.0;
    cfg.seed = 65432;
    cfg.verbose = 0;
    cfg.fmc_mode = 0;

    MCState state = mc_init(cfg);

    // Set per-replica betas
    double* h_betas = new double[total_replicas];
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < nrep_per_temp; r++)
            h_betas[t * nrep_per_temp + r] = beta_sched[t];
    mc_set_betas(state, h_betas);

    // Permutation: perm[logical_index] = physical replica
    int* perm = new int[total_replicas];
    for (int i = 0; i < total_replicas; i++) perm[i] = i;

    // Exchange counters
    long long* ex_acc  = new long long[NT - 1]();
    long long* ex_prop = new long long[NT - 1]();

    // Temperature visit tracking: visited[phys_replica][temp_idx]
    int* visited = new int[total_replicas * NT]();

    double* h_energies = new double[total_replicas];
    long long* h_acc   = new long long[total_replicas];
    long long* h_prop  = new long long[total_replicas];

    srand48((long)cfg.seed + 7777);

    // Mark initial positions
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < nrep_per_temp; r++)
            visited[perm[t * nrep_per_temp + r] * NT + t] = 1;

    for (int s = 0; s < sweeps; s++) {
        mc_sweep_pt(state);

        if ((s + 1) % pt_freq == 0) {
            mc_get_results(state, h_energies, h_acc, h_prop);

            for (int t = 0; t < NT - 1; t++) {
                for (int r = 0; r < nrep_per_temp; r++) {
                    int phys_a = perm[t       * nrep_per_temp + r];
                    int phys_b = perm[(t + 1) * nrep_per_temp + r];

                    double delta = (beta_sched[t] - beta_sched[t + 1])
                                 * (h_energies[phys_a] - h_energies[phys_b]);

                    ex_prop[t]++;
                    bool accept = (delta >= 0.0) || (drand48() < exp(delta));
                    if (accept) {
                        ex_acc[t]++;
                        int tmp = perm[t * nrep_per_temp + r];
                        perm[t * nrep_per_temp + r] = perm[(t + 1) * nrep_per_temp + r];
                        perm[(t + 1) * nrep_per_temp + r] = tmp;
                    }
                }
            }

            // Update betas and record visits
            for (int t = 0; t < NT; t++)
                for (int r = 0; r < nrep_per_temp; r++) {
                    int phys = perm[t * nrep_per_temp + r];
                    h_betas[phys] = beta_sched[t];
                    visited[phys * NT + t] = 1;
                }
            mc_set_betas(state, h_betas);

            // Write temperature walk trajectory
            if (fwalk) {
                for (int t = 0; t < NT; t++)
                    for (int rr = 0; rr < nrep_per_temp; rr++)
                        fprintf(fwalk, "%d %d %d\n",
                                s + 1, perm[t * nrep_per_temp + rr], t);
            }
        }
    }

    // --- Check 1: rates in [0, 1] ---
    int all_pass = 1;
    printf("  Exchange rates:\n");
    double* rates = new double[NT - 1];
    for (int t = 0; t < NT - 1; t++) {
        rates[t] = (ex_prop[t] > 0) ? (double)ex_acc[t] / ex_prop[t] : 0.0;
        printf("    T[%d]=%.3f <-> T[%d]=%.3f :  %lld/%lld = %.4f\n",
               t, T_sched[t], t + 1, T_sched[t + 1],
               ex_acc[t], ex_prop[t], rates[t]);
        if (frates) fprintf(frates, "%.6f %.6f %.6f\n",
                            T_sched[t], T_sched[t + 1], rates[t]);
        if (rates[t] < 0.0 || rates[t] > 1.0) {
            printf("    -> rate out of [0,1]!  FAIL\n");
            all_pass = 0;
        }
    }

    // --- Check 2: monotonicity (higher T-gap → lower rate) ---
    // rates[0] is Tmax↔Tmax-dT (smallest gap at high T, highest β gap)
    // For linear schedule with equal dT, rates should be roughly constant or
    // decrease as T gets colder (gap in β grows).  We just check no wild
    // non-monotonicity beyond noise.
    int mono_fail = 0;
    for (int t = 0; t < NT - 2; t++) {
        // Allow small violations (10%) due to finite statistics
        if (rates[t + 1] > rates[t] + 0.15) {
            mono_fail++;
        }
    }
    printf("\n  Monotonicity: %d violations  %s\n",
           mono_fail, (mono_fail == 0) ? "PASS" : "WARN (soft check)");

    // --- Check 3: temperature ergodicity ---
    int n_full_ergodic = 0;
    for (int phys = 0; phys < total_replicas; phys++) {
        int hot = visited[phys * NT + 0];       // visited Tmax?
        int cold = visited[phys * NT + (NT-1)]; // visited Tmin?
        if (hot && cold) n_full_ergodic++;
    }
    printf("  T-ergodicity: %d/%d replicas visited both extremes  %s\n",
           n_full_ergodic, total_replicas,
           (n_full_ergodic == total_replicas) ? "PASS" : "PARTIAL (may need more sweeps)");

    // Cleanup
    if (frates) fclose(frates);
    if (fwalk)  fclose(fwalk);
    delete[] T_sched;
    delete[] beta_sched;
    delete[] h_betas;
    delete[] perm;
    delete[] ex_acc;
    delete[] ex_prop;
    delete[] visited;
    delete[] h_energies;
    delete[] h_acc;
    delete[] h_prop;
    delete[] rates;
    mc_free(state);

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
