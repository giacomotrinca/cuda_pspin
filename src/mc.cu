#include "mc.h"
#include "spins.h"
#include "disorder.h"
#include "hamiltonian.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
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

// ============================================================================
// Seed generation from master seed (splitmix64 on host)
// ============================================================================
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ============================================================================
// RNG init kernel
// ============================================================================
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* states,
                                unsigned long long* seeds, int nrep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrep) {
        curand_init(seeds[idx], 0, 0, &states[idx]);
    }
}

// ============================================================================
// Fused MC sweep kernel: one block per replica, all C(N,2) pairs internal.
// Spins cached in shared memory. Warp-shuffle reduction.
// Differential H4 for single-spin-change quartets (Types 1 & 2).
// Three-type enumeration: no wasted iterations.
// ============================================================================
__global__ void mc_sweep_kernel(
    cuDoubleComplex* all_spins,   // [nrep * N]
    const cuDoubleComplex* g2,    // [N * N] shared
    const cuDoubleComplex* g4,    // [C(N,4)] shared
    int N, int nrep,
    const double* betas,
    curandStatePhilox4_32_10_t* rng_states,  // [nrep]
    double* energies,             // [nrep]
    long long* accepted,          // [nrep]
    long long* proposed           // [nrep]
) {
    int rep = blockIdx.x;
    if (rep >= nrep) return;

    double beta = betas[rep];

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int nwarps = bdim >> 5;

    // Shared memory: [N spins][2 proposal][nwarps warp sums]
    extern __shared__ char shared_buf[];
    cuDoubleComplex* s_spins = (cuDoubleComplex*)shared_buf;
    cuDoubleComplex* s_prop  = s_spins + N;
    double* s_warp = (double*)(s_prop + 2);

    cuDoubleComplex* g_spins = all_spins + (long long)rep * N;

    // Load spins into shared memory
    for (int i = tid; i < N; i += bdim)
        s_spins[i] = g_spins[i];
    __syncthreads();

    // Thread 0: cache RNG state, energy, counters in registers
    curandStatePhilox4_32_10_t rng;
    double cur_energy;
    long long loc_acc = 0, loc_prop = 0;
    if (tid == 0) {
        rng = rng_states[rep];
        cur_energy = energies[rep];
    }

    // H4 work counts (constant for the sweep)
    int Nm2 = N - 2;
    long long n_type12 = (long long)Nm2 * (Nm2 - 1) * (Nm2 - 2) / 6; // C(N-2,3)
    long long n_type3  = (long long)Nm2 * (Nm2 - 1) / 2;             // C(N-2,2)
    long long n_h4     = 2 * n_type12 + n_type3;

    // ======== Sweep over all C(N,2) pairs ========
    for (int i0 = 0; i0 < N; i0++) {
      for (int j0 = i0 + 1; j0 < N; j0++) {

        // --- Propose rotation (thread 0) ---
        if (tid == 0) {
            propose_pair_rotation(s_spins[i0], s_spins[j0], &rng,
                                  &s_prop[0], &s_prop[1]);
            loc_prop++;
        }
        __syncthreads();

        cuDoubleComplex a_i_new = s_prop[0];
        cuDoubleComplex a_j_new = s_prop[1];
        cuDoubleComplex a_i_old = s_spins[i0];
        cuDoubleComplex a_j_old = s_spins[j0];
        cuDoubleComplex delta_i = cuCsub(a_i_new, a_i_old);
        cuDoubleComplex delta_j = cuCsub(a_j_new, a_j_old);

        double local_dE = 0.0;

        // --- H2: factored computation ---
        // dE_H2 = -Re(delta_i * Sum_k g_{i0,k} conj(a_k)) + same for j0 + pair
        cuDoubleComplex sum_i = make_cuDoubleComplex(0.0, 0.0);
        cuDoubleComplex sum_j = make_cuDoubleComplex(0.0, 0.0);
        for (int k = tid; k < N; k += bdim) {
            if (k != i0 && k != j0) {
                cuDoubleComplex ak_c = cuConj(s_spins[k]);
                int ri = (i0 < k) ? i0 : k, ci = (i0 < k) ? k : i0;
                sum_i = cuCadd(sum_i, cuCmul(g2[ri * N + ci], ak_c));
                int rj = (j0 < k) ? j0 : k, cj = (j0 < k) ? k : j0;
                sum_j = cuCadd(sum_j, cuCmul(g2[rj * N + cj], ak_c));
            }
        }
        local_dE -= cuCreal(cuCmul(delta_i, sum_i));
        local_dE -= cuCreal(cuCmul(delta_j, sum_j));

        if (tid == 0) {
            cuDoubleComplex gij = g2[i0 * N + j0]; // i0 < j0
            cuDoubleComplex old_p = cuCmul(gij, cuCmul(a_i_old, cuConj(a_j_old)));
            cuDoubleComplex new_p = cuCmul(gij, cuCmul(a_i_new, cuConj(a_j_new)));
            local_dE -= (cuCreal(new_p) - cuCreal(old_p));
        }

        // --- H4: three-type enumeration with differential for Types 1&2 ---
        for (long long t = tid; t < n_h4; t += bdim) {
            int idx[4];
            int is_type3 = 0;

            if (t < n_type12) {
                // Type 1: {i0, a, b, c}  a<b<c from {0..N-1}\{i0,j0}
                long long tt = t;
                int cc = (int)cbrt(6.0 * (double)tt);
                if (cc < 2) cc = 2;
                while ((long long)cc * (cc-1) * (cc-2) / 6 > tt) cc--;
                while ((long long)(cc+1) * cc * (cc-1) / 6 <= tt) cc++;
                long long rem = tt - (long long)cc * (cc-1) * (cc-2) / 6;
                int bb = (int)sqrt(2.0 * (double)rem);
                if (bb < 1) bb = 1;
                while ((long long)bb * (bb-1) / 2 > rem) bb--;
                while ((long long)(bb+1) * bb / 2 <= rem) bb++;
                int aa = (int)(rem - (long long)bb * (bb-1) / 2);
                int a = aa; if (a >= i0) a++; if (a >= j0) a++;
                int b = bb; if (b >= i0) b++; if (b >= j0) b++;
                int c = cc; if (c >= i0) c++; if (c >= j0) c++;
                idx[0] = i0; idx[1] = a; idx[2] = b; idx[3] = c;
            } else if (t < 2 * n_type12) {
                // Type 2: {j0, a, b, c}
                long long tt = t - n_type12;
                int cc = (int)cbrt(6.0 * (double)tt);
                if (cc < 2) cc = 2;
                while ((long long)cc * (cc-1) * (cc-2) / 6 > tt) cc--;
                while ((long long)(cc+1) * cc * (cc-1) / 6 <= tt) cc++;
                long long rem = tt - (long long)cc * (cc-1) * (cc-2) / 6;
                int bb = (int)sqrt(2.0 * (double)rem);
                if (bb < 1) bb = 1;
                while ((long long)bb * (bb-1) / 2 > rem) bb--;
                while ((long long)(bb+1) * bb / 2 <= rem) bb++;
                int aa = (int)(rem - (long long)bb * (bb-1) / 2);
                int a = aa; if (a >= i0) a++; if (a >= j0) a++;
                int b = bb; if (b >= i0) b++; if (b >= j0) b++;
                int c = cc; if (c >= i0) c++; if (c >= j0) c++;
                idx[0] = j0; idx[1] = a; idx[2] = b; idx[3] = c;
            } else {
                // Type 3: {i0, j0, a, b}  a<b from {0..N-1}\{i0,j0}
                is_type3 = 1;
                long long tt = t - 2 * n_type12;
                int bb = (int)sqrt(2.0 * (double)tt);
                if (bb < 1) bb = 1;
                while ((long long)bb * (bb-1) / 2 > tt) bb--;
                while ((long long)(bb+1) * bb / 2 <= tt) bb++;
                int aa = (int)(tt - (long long)bb * (bb-1) / 2);
                int a = aa; if (a >= i0) a++; if (a >= j0) a++;
                int b = bb; if (b >= i0) b++; if (b >= j0) b++;
                idx[0] = i0; idx[1] = j0; idx[2] = a; idx[3] = b;
            }

            // Sort 4 elements
            if (idx[0] > idx[1]) { int tmp = idx[0]; idx[0] = idx[1]; idx[1] = tmp; }
            if (idx[2] > idx[3]) { int tmp = idx[2]; idx[2] = idx[3]; idx[3] = tmp; }
            if (idx[0] > idx[2]) { int tmp = idx[0]; idx[0] = idx[2]; idx[2] = tmp; }
            if (idx[1] > idx[3]) { int tmp = idx[1]; idx[1] = idx[3]; idx[3] = tmp; }
            if (idx[1] > idx[2]) { int tmp = idx[1]; idx[1] = idx[2]; idx[2] = tmp; }

            int ii = idx[0], jj = idx[1], kk = idx[2], ll = idx[3];
            long long q = (long long)ll*(ll-1)*(ll-2)*(ll-3)/24
                        + (long long)kk*(kk-1)*(kk-2)/6
                        + (long long)jj*(jj-1)/2 + ii;
            cuDoubleComplex gq = g4[q];

            if (is_type3) {
                // Both i0 and j0 change: full old & new
                cuDoubleComplex oi = s_spins[ii], oj_c = cuConj(s_spins[jj]);
                cuDoubleComplex ok = s_spins[kk], ol_c = cuConj(s_spins[ll]);
                cuDoubleComplex old_prod = cuCmul(gq, cuCmul(cuCmul(oi, oj_c), cuCmul(ok, ol_c)));
                cuDoubleComplex ni = (ii==i0) ? a_i_new : ((ii==j0) ? a_j_new : s_spins[ii]);
                cuDoubleComplex nj = (jj==i0) ? a_i_new : ((jj==j0) ? a_j_new : s_spins[jj]);
                cuDoubleComplex nk = (kk==i0) ? a_i_new : ((kk==j0) ? a_j_new : s_spins[kk]);
                cuDoubleComplex nl = (ll==i0) ? a_i_new : ((ll==j0) ? a_j_new : s_spins[ll]);
                nj = cuConj(nj); nl = cuConj(nl);  // positions 1,3 conjugated
                cuDoubleComplex new_prod = cuCmul(gq, cuCmul(cuCmul(ni, nj), cuCmul(nk, nl)));
                local_dE -= (cuCreal(new_prod) - cuCreal(old_prod));
            } else {
                // Single spin changes: differential (halves cuCmul count)
                int changed = (t < n_type12) ? i0 : j0;
                cuDoubleComplex delta = (changed == i0) ? delta_i : delta_j;
                // Positions 0,2 unconjugated; positions 1,3 conjugated
                cuDoubleComplex f0 = (ii == changed) ? delta : s_spins[ii];
                cuDoubleComplex f1 = (jj == changed) ? cuConj(delta) : cuConj(s_spins[jj]);
                cuDoubleComplex f2 = (kk == changed) ? delta : s_spins[kk];
                cuDoubleComplex f3 = (ll == changed) ? cuConj(delta) : cuConj(s_spins[ll]);
                cuDoubleComplex diff = cuCmul(gq, cuCmul(cuCmul(f0, f1), cuCmul(f2, f3)));
                local_dE -= cuCreal(diff);
            }
        }

        // --- Warp-shuffle reduction ---
        double val = local_dE;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if ((tid & 31) == 0) s_warp[tid >> 5] = val;
        __syncthreads();

        // --- Accept / reject (thread 0) ---
        if (tid == 0) {
            double dE = 0.0;
            for (int w = 0; w < nwarps; w++) dE += s_warp[w];
            bool accept;
            if (beta == 0.0) {
                accept = true;  // T = inf: accept everything
            } else {
                accept = (dE <= 0.0);
                if (!accept) {
                    double r = curand_uniform_double(&rng);
                    accept = (r < exp(-beta * dE));
                }
            }
            if (accept) {
                s_spins[i0] = a_i_new;
                s_spins[j0] = a_j_new;
                cur_energy += dE;
                loc_acc++;
            }
        }
        __syncthreads();
      }
    }

    // Write spins back to global memory
    for (int i = tid; i < N; i += bdim)
        g_spins[i] = s_spins[i];

    if (tid == 0) {
        rng_states[rep] = rng;
        energies[rep] = cur_energy;
        accepted[rep] += loc_acc;
        proposed[rep] += loc_prop;
    }
}

// ============================================================================
// Init energy kernel: compute energy for each replica
// ============================================================================
__global__ void init_energy_kernel(
    const cuDoubleComplex* all_spins,
    const cuDoubleComplex* g2,
    const cuDoubleComplex* g4,
    int N, int nrep,
    double* energies
) {
    int rep = blockIdx.y;
    if (rep >= nrep) return;

    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const cuDoubleComplex* spins = all_spins + (long long)rep * N;
    double local_sum = 0.0;

    // H2
    long long total_pairs = (long long)N * (N - 1) / 2;
    for (long long p = (long long)blockIdx.x * bdim + tid; p < total_pairs;
         p += (long long)gridDim.x * bdim) {
        int j = (int)(0.5 + sqrt(0.25 + 2.0 * p));
        int i = (int)(p - (long long)j * (j - 1) / 2);
        if (i >= j) { j++; i = (int)(p - (long long)j * (j - 1) / 2); }

        cuDoubleComplex gij = g2[i * N + j];
        cuDoubleComplex prod = cuCmul(gij, cuCmul(spins[i], cuConj(spins[j])));
        local_sum += -cuCreal(prod);
    }

    // H4
    long long total_q = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    for (long long q = (long long)blockIdx.x * bdim + tid; q < total_q;
         q += (long long)gridDim.x * bdim) {
        int ii, jj, kk, ll;
        ll = 3;
        while ((long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24 <= q) ll++;
        ll--;
        long long rem = q - (long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24;
        kk = 2;
        while ((long long)kk * (kk - 1) * (kk - 2) / 6 <= rem) kk++;
        kk--;
        rem -= (long long)kk * (kk - 1) * (kk - 2) / 6;
        jj = 1;
        while ((long long)jj * (jj - 1) / 2 <= rem) jj++;
        jj--;
        rem -= (long long)jj * (jj - 1) / 2;
        ii = (int)rem;

        cuDoubleComplex gq = g4[q];
        cuDoubleComplex prod = cuCmul(gq, cuCmul(
            cuCmul(spins[ii], cuConj(spins[jj])),
            cuCmul(spins[kk], cuConj(spins[ll]))));
        local_sum += -cuCreal(prod);
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&energies[rep], sdata[0]);
}

// ============================================================================
// Public API
// ============================================================================

MCState mc_init(const SimConfig& cfg) {
    MCState state;
    state.N = cfg.N;
    state.nrep = cfg.nrep;
    int N = cfg.N;
    int nrep = cfg.nrep;

    // Generate per-replica seeds from master seed
    uint64_t master = cfg.seed;
    uint64_t* h_seeds = new uint64_t[nrep];
    for (int r = 0; r < nrep; r++) {
        h_seeds[r] = splitmix64(&master);
    }

    // Allocate shared disorder
    CUDA_CHECK(cudaMalloc(&state.d_g2, (long long)N * N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(state.d_g2, 0, (long long)N * N * sizeof(cuDoubleComplex)));

    long long nq = n_quartets(N);
    CUDA_CHECK(cudaMalloc(&state.d_g4, nq * sizeof(cuDoubleComplex)));

    // Generate disorder (same for all replicas, use master seed)
    generate_g2(state.d_g2, N, cfg.J, cfg.seed + 1000);
    generate_g4(state.d_g4, N, cfg.J, cfg.seed + 2000);

    // FMC filtering
    if (cfg.fmc_mode > 0) {
        state.h_omega = new double[N];
        if (cfg.fmc_mode == 1) { // comb: omega_k = k/(N-1), equispaziato in [0,1]
            for (int i = 0; i < N; i++) state.h_omega[i] = (double)i / (double)(N - 1);
        } else { // uniform: omega_i ~ U[0,1]
            uint64_t freq_state = cfg.seed + 3000;
            for (int i = 0; i < N; i++) {
                uint64_t z = splitmix64(&freq_state);
                state.h_omega[i] = (double)(z >> 11) / (double)(1ULL << 53);
            }
        }
        double* d_omega;
        CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_omega, state.h_omega, N * sizeof(double), cudaMemcpyHostToDevice));
        apply_fmc_g2(state.d_g2, N, d_omega, cfg.gamma);
        apply_fmc_g4(state.d_g4, N, d_omega, cfg.gamma);
        CUDA_CHECK(cudaFree(d_omega));

        // Count surviving terms
        cuDoubleComplex* h_g2 = new cuDoubleComplex[N * N];
        CUDA_CHECK(cudaMemcpy(h_g2, state.d_g2, (long long)N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        state.n_pairs_active = 0;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if (h_g2[i * N + j].x != 0.0 || h_g2[i * N + j].y != 0.0)
                    state.n_pairs_active++;
        delete[] h_g2;

        long long nq_count = n_quartets(N);
        cuDoubleComplex* h_g4 = new cuDoubleComplex[nq_count];
        CUDA_CHECK(cudaMemcpy(h_g4, state.d_g4, nq_count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        state.n_quart_active = 0;
        for (long long q = 0; q < nq_count; q++)
            if (h_g4[q].x != 0.0 || h_g4[q].y != 0.0)
                state.n_quart_active++;
        delete[] h_g4;
    } else {
        state.h_omega = nullptr;
        state.n_pairs_active = n_pairs(N);
        state.n_quart_active = n_quartets(N);
    }

    // Allocate per-replica spins
    CUDA_CHECK(cudaMalloc(&state.d_spins, (long long)nrep * N * sizeof(cuDoubleComplex)));

    // Initialize each replica's spins with its own seed
    for (int r = 0; r < nrep; r++) {
        init_spins_uniform(state.d_spins + (long long)r * N, N, h_seeds[r]);
    }

    // Allocate and init RNG states
    CUDA_CHECK(cudaMalloc(&state.d_rng, nrep * sizeof(curandStatePhilox4_32_10_t)));
    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, nrep * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds, nrep * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    int rng_blocks = (nrep + 255) / 256;
    init_rng_kernel<<<rng_blocks, 256>>>(state.d_rng, d_seeds, nrep);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_seeds));
    delete[] h_seeds;

    // Allocate energies and counters
    CUDA_CHECK(cudaMalloc(&state.d_energies, nrep * sizeof(double)));
    CUDA_CHECK(cudaMemset(state.d_energies, 0, nrep * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_accepted, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMemset(state.d_accepted, 0, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&state.d_proposed, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMemset(state.d_proposed, 0, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&state.d_betas, nrep * sizeof(double)));

    // Compute initial energies on GPU
    long long max_terms = (nq > n_pairs(N)) ? nq : n_pairs(N);
    int e_blocks = (int)((max_terms + 255) / 256);
    if (e_blocks > 1024) e_blocks = 1024;  // cap grid size in x
    dim3 e_grid(e_blocks, nrep);
    init_energy_kernel<<<e_grid, 256, 256 * sizeof(double)>>>(
        state.d_spins, state.d_g2, state.d_g4, N, nrep, state.d_energies);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print initial energies if verbose (highest-T and lowest-energy only)
    if (cfg.verbose) {
        double* h_energies = new double[nrep];
        CUDA_CHECK(cudaMemcpy(h_energies, state.d_energies, nrep * sizeof(double),
                              cudaMemcpyDeviceToHost));
        int i_hot = 0;   // replica at highest T (index 0)
        int i_low = 0;   // replica with lowest energy
        for (int r = 1; r < nrep; r++) {
            if (h_energies[r] < h_energies[i_low]) i_low = r;
        }
        printf("  initial E/N  highest-T (rep %d): %.6f\n", i_hot, h_energies[i_hot] / N);
        printf("  initial E/N  lowest-E  (rep %d): %.6f\n", i_low, h_energies[i_low] / N);
        delete[] h_energies;
    }

    return state;
}

void mc_free(MCState& state) {
    delete[] state.h_omega;
    if (state.d_spins)    CUDA_CHECK(cudaFree(state.d_spins));
    if (state.d_g2)       CUDA_CHECK(cudaFree(state.d_g2));
    if (state.d_g4)       CUDA_CHECK(cudaFree(state.d_g4));
    if (state.d_rng)      CUDA_CHECK(cudaFree(state.d_rng));
    if (state.d_energies) CUDA_CHECK(cudaFree(state.d_energies));
    if (state.d_accepted) CUDA_CHECK(cudaFree(state.d_accepted));
    if (state.d_proposed) CUDA_CHECK(cudaFree(state.d_proposed));
    if (state.d_betas)    CUDA_CHECK(cudaFree(state.d_betas));
}

static __global__ void fill_double_kernel(double* arr, double val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

void mc_sweep(MCState& state, const SimConfig& cfg) {
    int N = state.N;
    int nrep = state.nrep;
    double beta = 1.0 / cfg.T;

    // Fill per-replica betas with uniform value
    fill_double_kernel<<<(nrep + 255) / 256, 256>>>(state.d_betas, beta, nrep);

    int block_size = 256;
    int nwarps = block_size >> 5;
    size_t shared_bytes = (N + 2) * sizeof(cuDoubleComplex) + nwarps * sizeof(double);

    mc_sweep_kernel<<<nrep, block_size, shared_bytes>>>(
        state.d_spins, state.d_g2, state.d_g4,
        N, nrep, state.d_betas,
        state.d_rng, state.d_energies,
        state.d_accepted, state.d_proposed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mc_get_results(const MCState& state, double* h_energies,
                    long long* h_accepted, long long* h_proposed) {
    CUDA_CHECK(cudaMemcpy(h_energies, state.d_energies,
                          state.nrep * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accepted, state.d_accepted,
                          state.nrep * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_proposed, state.d_proposed,
                          state.nrep * sizeof(long long), cudaMemcpyDeviceToHost));
}

void mc_get_spins(const MCState& state, cuDoubleComplex* h_spins) {
    CUDA_CHECK(cudaMemcpy(h_spins, state.d_spins,
                          (long long)state.nrep * state.N * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
}

void mc_sweep_pt(MCState& state) {
    int N = state.N;
    int nrep = state.nrep;

    int block_size = 256;
    int nwarps = block_size >> 5;
    size_t shared_bytes = (N + 2) * sizeof(cuDoubleComplex) + nwarps * sizeof(double);

    mc_sweep_kernel<<<nrep, block_size, shared_bytes>>>(
        state.d_spins, state.d_g2, state.d_g4,
        N, nrep, state.d_betas,
        state.d_rng, state.d_energies,
        state.d_accepted, state.d_proposed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mc_set_betas(MCState& state, const double* h_betas) {
    CUDA_CHECK(cudaMemcpy(state.d_betas, h_betas,
                          state.nrep * sizeof(double), cudaMemcpyHostToDevice));
}
