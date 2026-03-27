#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

// Init RNG states on device
__global__ void init_rng_states(curandStatePhilox4_32_10_t* states,
                                 unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

int main(int argc, char** argv) {
    // Parse configuration
    SimConfig cfg = parse_args(argc, argv);

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("========================================\n");
    printf("p-Spin 2+4 Complex Spherical MC\n");
    printf("========================================\n");
    printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N = %d\n", cfg.N);
    printf("T = %.4f\n", cfg.T);
    printf("J = %.4f\n", cfg.J);
    printf("Sweeps = %d (therm = %d)\n", cfg.n_sweeps, cfg.n_therm);
    printf("Delta = %.4f\n", cfg.delta);
    printf("Seed = %llu\n", (unsigned long long)cfg.seed);
    printf("Pairs: %lld\n", n_pairs(cfg.N));
    printf("Quartets: %lld\n", n_quartets(cfg.N));
    printf("========================================\n\n");

    // Memory estimate
    long long mem_g2 = (long long)cfg.N * cfg.N * sizeof(cuDoubleComplex);
    long long mem_g4 = n_quartets(cfg.N) * sizeof(cuDoubleComplex);
    long long mem_spins = cfg.N * sizeof(cuDoubleComplex);
    printf("Memory: g2=%.2f MB, g4=%.2f MB, spins=%.2f KB\n",
           mem_g2 / 1e6, mem_g4 / 1e6, mem_spins / 1e3);
    printf("Total GPU memory needed: ~%.2f MB\n\n",
           (mem_g2 + mem_g4 + mem_spins) / 1e6);

    // Check available memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.2f MB free / %.2f MB total\n\n",
           free_mem / 1e6, total_mem / 1e6);

    if ((long long)(mem_g2 + mem_g4 + mem_spins) > (long long)free_mem * 0.9) {
        fprintf(stderr, "ERROR: Not enough GPU memory!\n");
        return 1;
    }

    // Initialize MC state
    MCState state = mc_init(cfg);

    // Initialize RNG states on device
    int n_rng = 1;  // For now, single thread proposes moves
    curandStatePhilox4_32_10_t* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, n_rng * sizeof(curandStatePhilox4_32_10_t)));
    init_rng_states<<<1, 1>>>(d_rng_states, cfg.seed + 5000, n_rng);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Thermalization
    printf("Thermalizing (%d sweeps)...\n", cfg.n_therm);
    for (int s = 0; s < cfg.n_therm; s++) {
        mc_sweep(state, cfg, d_rng_states);
        if ((s + 1) % 100 == 0) {
            printf("  therm sweep %d/%d: E/N = %.6f, acc = %.4f\n",
                   s + 1, cfg.n_therm, state.energy / cfg.N,
                   mc_acceptance_ratio(state));
        }
    }

    // Reset counters after thermalization
    state.accepted = 0;
    state.proposed = 0;

    // Production sweeps
    printf("\nProduction (%d sweeps)...\n", cfg.n_sweeps);
    double E_sum = 0.0, E2_sum = 0.0;
    int n_measurements = 0;

    for (int s = 0; s < cfg.n_sweeps; s++) {
        mc_sweep(state, cfg, d_rng_states);

        if ((s + 1) % cfg.measure_every == 0) {
            double e = state.energy / cfg.N;
            E_sum += e;
            E2_sum += e * e;
            n_measurements++;

            if ((s + 1) % (cfg.measure_every * 100) == 0) {
                double E_avg = E_sum / n_measurements;
                double E2_avg = E2_sum / n_measurements;
                double cv = (E2_avg - E_avg * E_avg) * cfg.N / (cfg.T * cfg.T);
                printf("  sweep %d/%d: E/N = %.6f, <E/N> = %.6f, Cv/N = %.6f, acc = %.4f\n",
                       s + 1, cfg.n_sweeps, e, E_avg, cv / cfg.N,
                       mc_acceptance_ratio(state));
            }
        }
    }

    // Final results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    if (n_measurements > 0) {
        double E_avg = E_sum / n_measurements;
        double E2_avg = E2_sum / n_measurements;
        double cv = (E2_avg - E_avg * E_avg) * cfg.N / (cfg.T * cfg.T);
        printf("<E/N> = %.6f\n", E_avg);
        printf("Cv/N  = %.6f\n", cv / cfg.N);
    }
    printf("Acceptance ratio: %.4f\n", mc_acceptance_ratio(state));
    printf("========================================\n");

    // Verify energy by full recomputation
    double E_check = compute_energy(state.d_spins, state.d_g2, state.d_g4, cfg.N);
    printf("Energy check: tracked=%.6f recomputed=%.6f diff=%.2e\n",
           state.energy, E_check, fabs(state.energy - E_check));

    // Cleanup
    CUDA_CHECK(cudaFree(d_rng_states));
    mc_free(state);

    return 0;
}
