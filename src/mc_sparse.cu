#include "mc_sparse.h"
#include "spins_sparse.h"
#include "disorder.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
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

// atomicAdd(double*,double) fallback for sm < 6.0
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ============================================================================
// Seed generation from master seed (splitmix64)
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
__global__ void sparse_init_rng_kernel(curandStatePhilox4_32_10_t* states,
                                       unsigned long long* seeds, int nrep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrep)
        curand_init(seeds[idx], 0, 0, &states[idx]);
}

// ============================================================================
// Sparse MC sweep kernel
//
// One block per replica, N/2 pair proposals per sweep.
// H2: factored computation on dense g2 (same as the spherical kernel).
// H4: iterate over the n_quartets sparse entries.
// Spin proposal: propose_pair_rotation_cube  (conserves |a_i|^4+|a_j|^4).
// ============================================================================
__global__ void mc_sparse_sweep_kernel(
    cuDoubleComplex* __restrict__ all_spins,  // [nrep * N]
    const cuDoubleComplex* __restrict__ g2,   // [N * N]
    const SparseQuartet*   __restrict__ quartets, // [n_quartets]
    int n_quartets,
    int N, int nrep,
    const double* __restrict__ betas,
    curandStatePhilox4_32_10_t* rng_states,
    double*    energies,
    long long* accepted,
    long long* proposed,
    int h2_active
) {
    int rep = blockIdx.x;
    if (rep >= nrep) return;

    double beta = betas[rep];
    int tid  = threadIdx.x;
    int bdim = blockDim.x;
    int nwarps = bdim >> 5;

    // Shared memory layout: [N spins][2 proposal][nwarps warp sums][2 pair idx]
    extern __shared__ char shared_buf[];
    cuDoubleComplex* s_spins = (cuDoubleComplex*)shared_buf;
    cuDoubleComplex* s_prop  = s_spins + N;
    double* s_warp = (double*)(s_prop + 2);
    int*    s_pair = (int*)(s_warp + nwarps);

    cuDoubleComplex* g_spins = all_spins + (long long)rep * N;

    // Load spins into shared memory
    for (int i = tid; i < N; i += bdim)
        s_spins[i] = g_spins[i];
    __syncthreads();

    // Thread-0 caches RNG state and energy
    curandStatePhilox4_32_10_t rng;
    double cur_energy;
    long long loc_acc = 0, loc_prop = 0;
    if (tid == 0) {
        rng = rng_states[rep];
        cur_energy = energies[rep];
    }

    // Conjugation masks per channel
    // ch 0: conj positions {2,3} -> mask 0xC
    // ch 1: conj positions {1,2} -> mask 0x6
    // ch 2: conj positions {1,3} -> mask 0xA
    const int conj_masks[3] = {0xC, 0x6, 0xA};

    int n_steps = N / 2;

    for (int step = 0; step < n_steps; step++) {

        // --- Thread 0: choose random pair & propose cube-constrained move ---
        if (tid == 0) {
            int ri = (int)(curand_uniform_double(&rng) * N);
            if (ri >= N) ri = N - 1;
            int rj = (int)(curand_uniform_double(&rng) * (N - 1));
            if (rj >= N - 1) rj = N - 2;
            if (rj >= ri) rj++;
            int i0 = (ri < rj) ? ri : rj;
            int j0 = (ri < rj) ? rj : ri;
            s_pair[0] = i0;
            s_pair[1] = j0;
            propose_pair_rotation_cube(s_spins[i0], s_spins[j0], &rng,
                                       &s_prop[0], &s_prop[1]);
            loc_prop++;
        }
        __syncthreads();

        int i0 = s_pair[0];
        int j0 = s_pair[1];

        cuDoubleComplex a_i_new = s_prop[0];
        cuDoubleComplex a_j_new = s_prop[1];
        cuDoubleComplex a_i_old = s_spins[i0];
        cuDoubleComplex a_j_old = s_spins[j0];
        cuDoubleComplex delta_i = cuCsub(a_i_new, a_i_old);
        cuDoubleComplex delta_j = cuCsub(a_j_new, a_j_old);

        double local_dE = 0.0;

        // ----- H2 (dense, skip zeros naturally) -----
        if (h2_active) {
            cuDoubleComplex sum_i = make_cuDoubleComplex(0.0, 0.0);
            cuDoubleComplex sum_j = make_cuDoubleComplex(0.0, 0.0);
            for (int k = tid; k < N; k += bdim) {
                if (k != i0 && k != j0) {
                    cuDoubleComplex ak_c = cuConj(s_spins[k]);
                    sum_i = cuCadd(sum_i, cuCmul(g2[i0 * N + k], ak_c));
                    sum_j = cuCadd(sum_j, cuCmul(g2[j0 * N + k], ak_c));
                }
            }
            local_dE -= cuCreal(cuCmul(delta_i, sum_i));
            local_dE -= cuCreal(cuCmul(delta_j, sum_j));

            if (tid == 0) {
                cuDoubleComplex gij = g2[i0 * N + j0];
                cuDoubleComplex old_p = cuCmul(gij, cuCmul(a_i_old, cuConj(a_j_old)));
                cuDoubleComplex new_p = cuCmul(gij, cuCmul(a_i_new, cuConj(a_j_new)));
                local_dE -= (cuCreal(new_p) - cuCreal(old_p));
            }
        }

        // ----- Sparse H4: iterate over selected quartets -----
        for (int q = tid; q < n_quartets; q += bdim) {
            SparseQuartet sq = quartets[q];

            bool inv_i = (sq.i == i0 || sq.j == i0 || sq.k == i0 || sq.l == i0);
            bool inv_j = (sq.i == j0 || sq.j == j0 || sq.k == j0 || sq.l == j0);

            if (!inv_i && !inv_j) continue;

            int ids[4] = {sq.i, sq.j, sq.k, sq.l};
            int cm = conj_masks[sq.ch];

            if (inv_i && inv_j) {
                // Both i0 and j0 in this quartet: full old/new recomputation
                cuDoubleComplex old_f[4], new_f[4];
                for (int p = 0; p < 4; p++) {
                    cuDoubleComplex oval = s_spins[ids[p]];
                    cuDoubleComplex nval;
                    if      (ids[p] == i0) nval = a_i_new;
                    else if (ids[p] == j0) nval = a_j_new;
                    else                   nval = oval;
                    old_f[p] = ((cm >> p) & 1) ? cuConj(oval) : oval;
                    new_f[p] = ((cm >> p) & 1) ? cuConj(nval) : nval;
                }
                cuDoubleComplex old_prod = cuCmul(sq.g, cuCmul(cuCmul(old_f[0], old_f[1]),
                                                               cuCmul(old_f[2], old_f[3])));
                cuDoubleComplex new_prod = cuCmul(sq.g, cuCmul(cuCmul(new_f[0], new_f[1]),
                                                               cuCmul(new_f[2], new_f[3])));
                local_dE -= (cuCreal(new_prod) - cuCreal(old_prod));
            } else {
                // Only one of i0, j0 in this quartet: differential update
                int changed = inv_i ? i0 : j0;
                cuDoubleComplex delta = (changed == i0) ? delta_i : delta_j;
                cuDoubleComplex f[4];
                for (int p = 0; p < 4; p++) {
                    cuDoubleComplex base = (ids[p] == changed) ? delta : s_spins[ids[p]];
                    f[p] = ((cm >> p) & 1) ? cuConj(base) : base;
                }
                cuDoubleComplex diff = cuCmul(sq.g, cuCmul(cuCmul(f[0], f[1]),
                                                           cuCmul(f[2], f[3])));
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
                accept = true;
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

    // Write spins back to global memory
    for (int i = tid; i < N; i += bdim)
        g_spins[i] = s_spins[i];

    if (tid == 0) {
        rng_states[rep] = rng;
        energies[rep]   = cur_energy;
        accepted[rep]  += loc_acc;
        proposed[rep]  += loc_prop;
    }
}

// ============================================================================
// Init energy kernel (H2 dense + H4 sparse)
// ============================================================================
__global__ void init_energy_sparse_kernel(
    const cuDoubleComplex* __restrict__ all_spins,
    const cuDoubleComplex* __restrict__ g2,
    const SparseQuartet*   __restrict__ quartets,
    int n_quartets,
    int N, int nrep,
    double* energies,
    int h2_active
) {
    int rep = blockIdx.y;
    if (rep >= nrep) return;

    extern __shared__ double sdata[];
    int tid  = threadIdx.x;
    int bdim = blockDim.x;

    const cuDoubleComplex* spins = all_spins + (long long)rep * N;
    double local_sum = 0.0;

    // H2: same as dense version
    if (h2_active) {
        long long total_pairs = (long long)N * (N - 1) / 2;
        for (long long p = (long long)blockIdx.x * bdim + tid; p < total_pairs;
             p += (long long)gridDim.x * bdim) {
            int j = (int)(0.5 + sqrt(0.25 + 2.0 * p));
            int i = (int)(p - (long long)j * (j - 1) / 2);
            if (i >= j) { j++; i = (int)(p - (long long)j * (j - 1) / 2); }

            cuDoubleComplex gij  = g2[i * N + j];
            cuDoubleComplex prod = cuCmul(gij, cuCmul(spins[i], cuConj(spins[j])));
            local_sum += -cuCreal(prod);
        }
    }

    // Sparse H4
    for (int q = (int)((long long)blockIdx.x * bdim + tid); q < n_quartets;
         q += (int)((long long)gridDim.x * bdim)) {
        SparseQuartet sq = quartets[q];
        cuDoubleComplex ai = spins[sq.i], aj = spins[sq.j];
        cuDoubleComplex ak = spins[sq.k], al = spins[sq.l];

        cuDoubleComplex s0, s1, s2, s3;
        if (sq.ch == 0)      { s0 = ai; s1 = aj;         s2 = cuConj(ak); s3 = cuConj(al); }
        else if (sq.ch == 1) { s0 = ai; s1 = cuConj(aj); s2 = cuConj(ak); s3 = al; }
        else                 { s0 = ai; s1 = cuConj(aj); s2 = ak;         s3 = cuConj(al); }

        cuDoubleComplex prod = cuCmul(sq.g, cuCmul(cuCmul(s0, s1), cuCmul(s2, s3)));
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
// Combinatorial index decoder (host side)
// ============================================================================
static void decode_quartet_index(long long q, int* ii, int* jj, int* kk, int* ll) {
    int l = 3;
    while ((long long)l * (l - 1) * (l - 2) * (l - 3) / 24 <= q) l++;
    l--;
    long long rem = q - (long long)l * (l - 1) * (l - 2) * (l - 3) / 24;
    int k = 2;
    while ((long long)k * (k - 1) * (k - 2) / 6 <= rem) k++;
    k--;
    rem -= (long long)k * (k - 1) * (k - 2) / 6;
    int j = 1;
    while ((long long)j * (j - 1) / 2 <= rem) j++;
    j--;
    rem -= (long long)j * (j - 1) / 2;
    int i = (int)rem;
    *ii = i; *jj = j; *kk = k; *ll = l;
}

// ============================================================================
// Fisher-Yates partial shuffle: select n_select from count entries
// ============================================================================
static void partial_shuffle(long long* arr, long long count, int n_select, uint64_t seed) {
    uint64_t rng = seed;
    for (int i = 0; i < n_select && i < (int)count; i++) {
        uint64_t r = splitmix64(&rng);
        long long j_idx = i + (long long)(r % (uint64_t)(count - i));
        long long tmp = arr[i];
        arr[i]     = arr[j_idx];
        arr[j_idx] = tmp;
    }
}

// ============================================================================
// Optimal block size via occupancy API
// ============================================================================
static size_t sparse_sweep_smem(int N, int bs) {
    int nw = bs >> 5;
    return (size_t)(N + 2) * sizeof(cuDoubleComplex)
         + (size_t)nw * sizeof(double)
         + 2 * sizeof(int);
}

static int optimal_block_size_sparse(int N) {
    static int cached_N  = -1;
    static int cached_bs = 256;
    if (N == cached_N) return cached_bs;

    static const int candidates[] = {32, 64, 128, 256, 512, 1024};
    int best_bs = 256;
    int best_occ = 0;
    for (int c = 0; c < 6; ++c) {
        int bs = candidates[c];
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nb, mc_sparse_sweep_kernel,
            bs, sparse_sweep_smem(N, bs));
        int occ = nb * bs;
        if (occ > best_occ) { best_occ = occ; best_bs = bs; }
    }
    if (best_bs < 32) best_bs = 32;

    printf("[auto-tune] mc_sparse_sweep  N=%d  -> block_size=%d\n", N, best_bs);
    cached_N  = N;
    cached_bs = best_bs;
    return best_bs;
}

// ============================================================================
// Public API
// ============================================================================

MCStateSparse mc_sparse_init(const SimConfig& cfg) {
    MCStateSparse state;
    state.N    = cfg.N;
    state.nrep = cfg.nrep;
    int N    = cfg.N;
    int nrep = cfg.nrep;
    int n_sparse_quartets = N;  // always N

    // Per-replica seeds
    uint64_t master = cfg.seed;
    uint64_t* h_seeds = new uint64_t[nrep];
    for (int r = 0; r < nrep; r++)
        h_seeds[r] = splitmix64(&master);

    // ---- G2 (dense, same as spherical) ----
    CUDA_CHECK(cudaMalloc(&state.d_g2, (long long)N * N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(state.d_g2, 0, (long long)N * N * sizeof(cuDoubleComplex)));

    // ---- Temporary dense G4 + mask for FMC filtering ----
    long long nq     = n_quartets(N);
    cuDoubleComplex* d_g4_tmp;
    CUDA_CHECK(cudaMalloc(&d_g4_tmp, nq * sizeof(cuDoubleComplex)));
    uint8_t* d_g4_mask_tmp;
    CUDA_CHECK(cudaMalloc(&d_g4_mask_tmp, nq * sizeof(uint8_t)));

    // Compute effective coupling scales from alpha parametrization
    double J2   = (1.0 - cfg.alpha)  * cfg.J;
    double J4   = cfg.alpha           * cfg.J;
    double J2_0 = (1.0 - cfg.alpha0) * cfg.J0;
    double J4_0 = cfg.alpha0          * cfg.J0;

    // Generate disorder
    if (cfg.fmc_mode != 1)
        generate_g2(state.d_g2, N, J2, cfg.seed + 1000);
    generate_g4(d_g4_tmp, N, J4, cfg.seed + 2000);
    init_g4_mask(d_g4_mask_tmp, N);

    // ---- FMC filtering ----
    if (cfg.fmc_mode > 0) {
        state.h_omega = new double[N];
        if (cfg.fmc_mode == 1) {
            for (int i = 0; i < N; i++) state.h_omega[i] = (double)i;
        } else {
            uint64_t freq_state = cfg.seed + 3000;
            for (int i = 0; i < N; i++) {
                uint64_t z = splitmix64(&freq_state);
                state.h_omega[i] = (double)(z >> 11) / (double)(1ULL << 53);
            }
            std::sort(state.h_omega, state.h_omega + N);
        }
        double* d_omega;
        CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_omega, state.h_omega, N * sizeof(double),
                              cudaMemcpyHostToDevice));

        if (cfg.fmc_mode == 1) {
            state.n_pairs_active = 0;
        } else {
            apply_fmc_g2(state.d_g2, N, d_omega, cfg.gamma);
            cuDoubleComplex* h_g2 = new cuDoubleComplex[(long long)N * N];
            CUDA_CHECK(cudaMemcpy(h_g2, state.d_g2,
                                  (long long)N * N * sizeof(cuDoubleComplex),
                                  cudaMemcpyDeviceToHost));
            state.n_pairs_active = 0;
            for (int i = 0; i < N; i++)
                for (int j = i; j < N; j++)
                    if (h_g2[i * N + j].x != 0.0 || h_g2[i * N + j].y != 0.0)
                        state.n_pairs_active++;
            delete[] h_g2;
        }

        apply_fmc_g4(d_g4_mask_tmp, N, d_omega, cfg.gamma);
        CUDA_CHECK(cudaFree(d_omega));
    } else {
        state.h_omega = nullptr;
        state.n_pairs_active = n_pairs(N);
    }

    // Rescale g2 (same normalization as spherical code)
    rescale_g2(state.d_g2, N, J2, state.n_pairs_active);
    // Add g2 mean shift
    if (J2_0 != 0.0 && state.n_pairs_active > 0) {
        double mean_g2 = J2_0 * (double)N / (double)state.n_pairs_active;
        shift_mean_couplings(state.d_g2, (long long)N * N, mean_g2);
    }
    state.h2_active = (state.n_pairs_active > 0) ? 1 : 0;

    // ---- Sparse H4 sampling ----
    // Copy g4 couplings and mask to host
    cuDoubleComplex* h_g4 = new cuDoubleComplex[nq];
    CUDA_CHECK(cudaMemcpy(h_g4, d_g4_tmp, nq * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_g4_tmp));  // temporary dense g4 freed

    uint8_t* h_mask = new uint8_t[nq];
    CUDA_CHECK(cudaMemcpy(h_mask, d_g4_mask_tmp, nq * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_g4_mask_tmp));  // temporary mask freed

    // Collect non-zero entry indices as (q, ch) pairs encoded as q*3+ch
    // Each active (quartet, channel) pair is a potential sparse entry.
    long long nq_tot = n_g4_total(N);  // C(N,4) — max possible (1 channel per quartet)
    long long* nonzero_idx = new long long[nq_tot];
    long long n_nonzero = 0;
    for (long long q = 0; q < nq; q++) {
        if (h_g4[q].x == 0.0 && h_g4[q].y == 0.0) continue;
        uint8_t m = h_mask[q];
        for (int ch = 0; ch < 3; ch++)
            if (m & (1 << ch))
                nonzero_idx[n_nonzero++] = q * 3 + ch;
    }

    state.n_quart_active = n_nonzero;

    if (n_nonzero < n_sparse_quartets) {
        fprintf(stderr, "WARNING: only %lld non-zero quartets after FMC, "
                "requested %d. Using all.\n", n_nonzero, n_sparse_quartets);
        n_sparse_quartets = (int)n_nonzero;
    }

    // Partial Fisher-Yates shuffle to select n_sparse_quartets
    partial_shuffle(nonzero_idx, n_nonzero, n_sparse_quartets, cfg.seed + 4000);

    // Build SparseQuartet array
    state.n_quartets = n_sparse_quartets;
    SparseQuartet* h_sq = new SparseQuartet[n_sparse_quartets];

    // Coupling variance: sigma = J4 * sqrt(N / n_selected)
    double sigma = J4 * sqrt((double)N / (double)n_sparse_quartets);
    // Coupling mean shift for g4
    double mean_g4 = (J4_0 != 0.0) ? J4_0 * (double)N / (double)n_sparse_quartets : 0.0;

    for (int s = 0; s < n_sparse_quartets; s++) {
        long long encoded = nonzero_idx[s];
        int ch     = (int)(encoded % 3);
        long long q = encoded / 3;

        int ii, jj, kk, ll;
        decode_quartet_index(q, &ii, &jj, &kk, &ll);

        h_sq[s].i  = ii;
        h_sq[s].j  = jj;
        h_sq[s].k  = kk;
        h_sq[s].l  = ll;
        h_sq[s].ch = ch;
        // Coupling: original unit-variance value * sigma + mean
        h_sq[s].g  = make_cuDoubleComplex(cuCreal(h_g4[q]) * sigma + mean_g4, 0.0);
    }

    delete[] h_g4;
    delete[] h_mask;
    delete[] nonzero_idx;

    // Copy sparse quartets to device
    CUDA_CHECK(cudaMalloc(&state.d_quartets,
                          n_sparse_quartets * sizeof(SparseQuartet)));
    CUDA_CHECK(cudaMemcpy(state.d_quartets, h_sq,
                          n_sparse_quartets * sizeof(SparseQuartet),
                          cudaMemcpyHostToDevice));
    delete[] h_sq;

    // ---- Allocate per-replica spins, init on cube surface ----
    CUDA_CHECK(cudaMalloc(&state.d_spins,
                          (long long)nrep * N * sizeof(cuDoubleComplex)));
    for (int r = 0; r < nrep; r++)
        init_spins_cube(state.d_spins + (long long)r * N, N, h_seeds[r]);

    // ---- RNG states ----
    CUDA_CHECK(cudaMalloc(&state.d_rng,
                          nrep * sizeof(curandStatePhilox4_32_10_t)));
    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, nrep * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds,
                          nrep * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    int rng_blocks = (nrep + 255) / 256;
    sparse_init_rng_kernel<<<rng_blocks, 256>>>(state.d_rng, d_seeds, nrep);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_seeds));
    delete[] h_seeds;

    // ---- Energies and counters ----
    CUDA_CHECK(cudaMalloc(&state.d_energies, nrep * sizeof(double)));
    CUDA_CHECK(cudaMemset(state.d_energies, 0, nrep * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_accepted, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMemset(state.d_accepted, 0, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&state.d_proposed, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMemset(state.d_proposed, 0, nrep * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&state.d_betas, nrep * sizeof(double)));

    // ---- Compute initial energies ----
    long long max_terms = n_pairs(N);
    if (n_sparse_quartets > max_terms) max_terms = n_sparse_quartets;
    int e_blocks = (int)((max_terms + 255) / 256);
    if (e_blocks > 1024) e_blocks = 1024;
    dim3 e_grid(e_blocks, nrep);
    init_energy_sparse_kernel<<<e_grid, 256, 256 * sizeof(double)>>>(
        state.d_spins, state.d_g2, state.d_quartets, state.n_quartets,
        N, nrep, state.d_energies, state.h2_active);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cfg.verbose) {
        double* h_en = new double[nrep];
        CUDA_CHECK(cudaMemcpy(h_en, state.d_energies,
                              nrep * sizeof(double), cudaMemcpyDeviceToHost));
        int i_low = 0;
        for (int r = 1; r < nrep; r++)
            if (h_en[r] < h_en[i_low]) i_low = r;
        printf("  initial E/N  highest-T (rep 0): %.6f\n", h_en[0] / N);
        printf("  initial E/N  lowest-E  (rep %d): %.6f\n", i_low, h_en[i_low] / N);
        delete[] h_en;
    }

    return state;
}

void mc_sparse_free(MCStateSparse& state) {
    delete[] state.h_omega;
    if (state.d_spins)    CUDA_CHECK(cudaFree(state.d_spins));
    if (state.d_g2)       CUDA_CHECK(cudaFree(state.d_g2));
    if (state.d_quartets) CUDA_CHECK(cudaFree(state.d_quartets));
    if (state.d_rng)      CUDA_CHECK(cudaFree(state.d_rng));
    if (state.d_energies) CUDA_CHECK(cudaFree(state.d_energies));
    if (state.d_accepted) CUDA_CHECK(cudaFree(state.d_accepted));
    if (state.d_proposed) CUDA_CHECK(cudaFree(state.d_proposed));
    if (state.d_betas)    CUDA_CHECK(cudaFree(state.d_betas));
}

void mc_sparse_sweep_pt(MCStateSparse& state) {
    int N    = state.N;
    int nrep = state.nrep;

    int block_size = optimal_block_size_sparse(N);
    int nwarps = block_size >> 5;
    size_t shared_bytes = (size_t)(N + 2) * sizeof(cuDoubleComplex)
                        + (size_t)nwarps * sizeof(double)
                        + 2 * sizeof(int);

    mc_sparse_sweep_kernel<<<nrep, block_size, shared_bytes>>>(
        state.d_spins, state.d_g2,
        state.d_quartets, state.n_quartets,
        N, nrep, state.d_betas,
        state.d_rng, state.d_energies,
        state.d_accepted, state.d_proposed,
        state.h2_active);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mc_sparse_set_betas(MCStateSparse& state, const double* h_betas) {
    CUDA_CHECK(cudaMemcpy(state.d_betas, h_betas,
                          state.nrep * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void mc_sparse_get_results(const MCStateSparse& state, double* h_energies,
                           long long* h_accepted, long long* h_proposed) {
    CUDA_CHECK(cudaMemcpy(h_energies, state.d_energies,
                          state.nrep * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accepted, state.d_accepted,
                          state.nrep * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_proposed, state.d_proposed,
                          state.nrep * sizeof(long long), cudaMemcpyDeviceToHost));
}

void mc_sparse_get_spins(const MCStateSparse& state, cuDoubleComplex* h_spins) {
    CUDA_CHECK(cudaMemcpy(h_spins, state.d_spins,
                          (long long)state.nrep * state.N * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Split energy kernel (sparse H4): computes H2 and H4 separately
// ============================================================================
__global__ void split_energy_sparse_kernel(
    const cuDoubleComplex* __restrict__ all_spins,
    const cuDoubleComplex* __restrict__ g2,
    const SparseQuartet*   __restrict__ quartets,
    int n_quartets_total,
    int N, int nrep,
    double* energies_h2,
    double* energies_h4,
    int h2_active
) {
    int rep = blockIdx.y;
    if (rep >= nrep) return;

    extern __shared__ double sdata[];
    int tid  = threadIdx.x;
    int bdim = blockDim.x;
    double* sdata_h2 = sdata;
    double* sdata_h4 = sdata + bdim;

    const cuDoubleComplex* spins = all_spins + (long long)rep * N;
    double local_h2 = 0.0, local_h4 = 0.0;

    // H2: dense
    if (h2_active) {
        long long total_pairs = (long long)N * (N - 1) / 2;
        for (long long p = (long long)blockIdx.x * bdim + tid; p < total_pairs;
             p += (long long)gridDim.x * bdim) {
            int j = (int)(0.5 + sqrt(0.25 + 2.0 * p));
            int i = (int)(p - (long long)j * (j - 1) / 2);
            if (i >= j) { j++; i = (int)(p - (long long)j * (j - 1) / 2); }
            cuDoubleComplex gij  = g2[i * N + j];
            cuDoubleComplex prod = cuCmul(gij, cuCmul(spins[i], cuConj(spins[j])));
            local_h2 += -cuCreal(prod);
        }
    }

    // Sparse H4
    for (int q = (int)((long long)blockIdx.x * bdim + tid); q < n_quartets_total;
         q += (int)((long long)gridDim.x * bdim)) {
        SparseQuartet sq = quartets[q];
        cuDoubleComplex ai = spins[sq.i], aj = spins[sq.j];
        cuDoubleComplex ak = spins[sq.k], al = spins[sq.l];

        cuDoubleComplex s0, s1, s2, s3;
        if (sq.ch == 0)      { s0 = ai; s1 = aj;         s2 = cuConj(ak); s3 = cuConj(al); }
        else if (sq.ch == 1) { s0 = ai; s1 = cuConj(aj); s2 = cuConj(ak); s3 = al; }
        else                 { s0 = ai; s1 = cuConj(aj); s2 = ak;         s3 = cuConj(al); }

        cuDoubleComplex prod = cuCmul(sq.g, cuCmul(cuCmul(s0, s1), cuCmul(s2, s3)));
        local_h4 += -cuCreal(prod);
    }

    sdata_h2[tid] = local_h2;
    sdata_h4[tid] = local_h4;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_h2[tid] += sdata_h2[tid + s];
            sdata_h4[tid] += sdata_h4[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&energies_h2[rep], sdata_h2[0]);
        atomicAdd(&energies_h4[rep], sdata_h4[0]);
    }
}

void mc_sparse_compute_split_energies(const MCStateSparse& state, double* h_e2, double* h_e4) {
    int N = state.N;
    int nrep = state.nrep;

    double* d_e2;
    double* d_e4;
    CUDA_CHECK(cudaMalloc(&d_e2, nrep * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_e4, nrep * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_e2, 0, nrep * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_e4, 0, nrep * sizeof(double)));

    long long max_terms = (state.n_quartets > n_pairs(N))
                        ? state.n_quartets : n_pairs(N);
    int e_blocks = (int)((max_terms + 255) / 256);
    if (e_blocks > 1024) e_blocks = 1024;
    dim3 grid(e_blocks, nrep);
    split_energy_sparse_kernel<<<grid, 256, 2 * 256 * sizeof(double)>>>(
        state.d_spins, state.d_g2, state.d_quartets, state.n_quartets,
        N, nrep, d_e2, d_e4, state.h2_active);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_e2, d_e2, nrep * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e4, d_e4, nrep * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_e2));
    CUDA_CHECK(cudaFree(d_e4));
}
