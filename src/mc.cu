#include "mc.h"
#include "spins.h"
#include "disorder.h"
#include "hamiltonian.h"
#include <cstdio>
#include <cmath>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

MCState mc_init(const SimConfig& cfg) {
    MCState state;
    state.N = cfg.N;
    int N = cfg.N;

    // Allocate spins
    CUDA_CHECK(cudaMalloc(&state.d_spins, N * sizeof(cuDoubleComplex)));

    // Allocate 2-body couplings (full N×N matrix, only upper triangle used)
    CUDA_CHECK(cudaMalloc(&state.d_g2, (long long)N * N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(state.d_g2, 0, (long long)N * N * sizeof(cuDoubleComplex)));

    // Allocate 4-body couplings
    long long nq = n_quartets(N);
    CUDA_CHECK(cudaMalloc(&state.d_g4, nq * sizeof(cuDoubleComplex)));

    // Allocate workspace for reductions
    // Need max(N, nq) / block_size blocks
    long long max_terms = (nq > N) ? nq : N;
    long long ws_size = (max_terms + 255) / 256;
    CUDA_CHECK(cudaMalloc(&state.d_workspace, ws_size * sizeof(double)));

    // Initialize spins uniformly on hypersphere
    init_spins_uniform(state.d_spins, N, cfg.seed);

    // Generate disorder
    generate_g2(state.d_g2, N, cfg.J, cfg.seed + 1000);
    generate_g4(state.d_g4, N, cfg.J, cfg.seed + 2000);

    // Compute initial energy
    state.energy = compute_energy(state.d_spins, state.d_g2, state.d_g4, N);

    state.accepted = 0;
    state.proposed = 0;

    printf("Initial energy: %.6f\n", state.energy);
    printf("Energy per spin: %.6f\n", state.energy / N);

    return state;
}

void mc_free(MCState& state) {
    if (state.d_spins) CUDA_CHECK(cudaFree(state.d_spins));
    if (state.d_g2) CUDA_CHECK(cudaFree(state.d_g2));
    if (state.d_g4) CUDA_CHECK(cudaFree(state.d_g4));
    if (state.d_workspace) CUDA_CHECK(cudaFree(state.d_workspace));
    state.d_spins = nullptr;
    state.d_g2 = nullptr;
    state.d_g4 = nullptr;
    state.d_workspace = nullptr;
}

// Kernel to init curand states
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* states,
                                unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel: propose and evaluate one pair move per thread
// Each thread handles one pair (i, j) sequentially for the MC update
// NOTE: For now, we serialize accept/reject on the host. This is the
// simplest correct implementation. Future: parallel updates with checkerboard.
__global__ void propose_move_kernel(
    cuDoubleComplex* spins,
    int N, int i0, int j0,
    double delta,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* proposed  // [2]: proposed a_i, a_j
) {
    cuDoubleComplex a_i = spins[i0];
    cuDoubleComplex a_j = spins[j0];

    cuDoubleComplex a_i_new, a_j_new;
    propose_pair_rotation(a_i, a_j, delta, rng_state, &a_i_new, &a_j_new);

    proposed[0] = a_i_new;
    proposed[1] = a_j_new;
}

void mc_sweep(MCState& state, const SimConfig& cfg,
              curandStatePhilox4_32_10_t* d_rng_states) {
    int N = state.N;
    double beta = 1.0 / cfg.T;

    // Temporary storage for proposed spins
    cuDoubleComplex* d_proposed;
    CUDA_CHECK(cudaMalloc(&d_proposed, 2 * sizeof(cuDoubleComplex)));

    cuDoubleComplex h_proposed[2];

    // Host-side RNG for accept/reject (separate from GPU RNG)
    // We use a simple host-side approach for now
    static unsigned long long host_rng_state = cfg.seed + 999;

    // Iterate over ALL pairs (i, j) with i < j
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            state.proposed++;

            // Propose move on GPU (single thread)
            propose_move_kernel<<<1, 1>>>(
                state.d_spins, N, i, j, cfg.delta,
                &d_rng_states[0], d_proposed);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy proposed values to host
            CUDA_CHECK(cudaMemcpy(h_proposed, d_proposed,
                                  2 * sizeof(cuDoubleComplex),
                                  cudaMemcpyDeviceToHost));

            // Compute delta_E on GPU
            double dE = compute_delta_E_pair(
                state.d_spins, state.d_g2, state.d_g4, N,
                i, j, h_proposed[0], h_proposed[1],
                state.d_workspace);

            // Metropolis accept/reject
            bool accept = false;
            if (dE <= 0.0) {
                accept = true;
            } else {
                // Simple LCG for host-side uniform random
                host_rng_state = host_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
                double u = (double)(host_rng_state >> 33) / (double)(1ULL << 31);
                accept = (u < exp(-beta * dE));
            }

            if (accept) {
                state.accepted++;
                state.energy += dE;

                // Update spins on device
                CUDA_CHECK(cudaMemcpy(&state.d_spins[i], &h_proposed[0],
                                      sizeof(cuDoubleComplex),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(&state.d_spins[j], &h_proposed[1],
                                      sizeof(cuDoubleComplex),
                                      cudaMemcpyHostToDevice));
            }
        }
    }

    CUDA_CHECK(cudaFree(d_proposed));
}

double mc_acceptance_ratio(const MCState& state) {
    if (state.proposed == 0) return 0.0;
    return (double)state.accepted / (double)state.proposed;
}
