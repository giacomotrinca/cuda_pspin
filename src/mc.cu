#include "mc.h"
#include "spins.h"
#include "disorder.h"
#include "hamiltonian.h"
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

// atomicAdd(double*,double) is only hardware-native on sm_60+.
// Provide a CAS-based fallback for older architectures (e.g. sm_30).
// The guard uses defined(__CUDA_ARCH__) so it is emitted only during device
// compilation, avoiding conflicts with CUDA's host-side declarations.
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
    const uint8_t* g4_mask,       // [C(N,4)] 3-bit channel mask
    int N, int nrep,
    const double* betas,
    curandStatePhilox4_32_10_t* rng_states,  // [nrep]
    double* energies,             // [nrep]
    long long* accepted,          // [nrep]
    long long* proposed,          // [nrep]
    int h2_active                 // whether H2 terms are present
) {
    int rep = blockIdx.x;
    if (rep >= nrep) return;

    double beta = betas[rep];

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int nwarps = bdim >> 5;

    // Shared memory: [N spins][2 proposal][nwarps warp sums][2 pair indices]
    extern __shared__ char shared_buf[];
    cuDoubleComplex* s_spins = (cuDoubleComplex*)shared_buf;
    cuDoubleComplex* s_prop  = s_spins + N;
    double* s_warp = (double*)(s_prop + 2);
    int* s_pair    = (int*)(s_warp + nwarps);  // s_pair[0]=i0, s_pair[1]=j0

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

    // Three-type enumeration work counts:
    //   Type 1: {i0, a, b, c}  with a<b<c from {0..N-1}\{i0,j0}  → C(N-2,3) quartets
    //   Type 2: {j0, a, b, c}  same count
    //   Type 3: {i0, j0, a, b} with a<b from {0..N-1}\{i0,j0}    → C(N-2,2) quartets
    long long n_type12 = (long long)(N - 2) * (N - 3) * (N - 4) / 6;  // C(N-2,3)
    long long n_type3  = (long long)(N - 2) * (N - 3) / 2;            // C(N-2,2)
    long long n_h4     = 2 * n_type12 + n_type3;  // total involved quartets

    // Conjugation masks per channel (bit p set = position p conjugated)
    // ch 0: conj {kk,ll}  mask=0xC   ch 1: conj {jj,kk} mask=0x6   ch 2: conj {jj,ll} mask=0xA
    const int conj_masks[3] = {0xC, 0x6, 0xA};

    int n_steps = N / 2;  // Each step updates 2 spins → N total per sweep

    // ======== Sweep: N/2 random pair proposals ========
    for (int step = 0; step < n_steps; step++) {

        // --- Choose random pair and propose rotation (thread 0) ---
        if (tid == 0) {
            // Pick random i0 in [0, N-1]
            int ri = (int)(curand_uniform_double(&rng) * N);
            if (ri >= N) ri = N - 1;
            // Pick random j0 in [0, N-2], skip i0
            int rj = (int)(curand_uniform_double(&rng) * (N - 1));
            if (rj >= N - 1) rj = N - 2;
            if (rj >= ri) rj++;
            // Ensure i0 < j0 for canonical ordering
            int i0 = (ri < rj) ? ri : rj;
            int j0 = (ri < rj) ? rj : ri;
            s_pair[0] = i0;
            s_pair[1] = j0;
            propose_pair_rotation(s_spins[i0], s_spins[j0], &rng,
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

        // --- H2: factored computation (skipped entirely when no pairs survive FMC) ---
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
                cuDoubleComplex gij = g2[i0 * N + j0]; // i0 < j0
                cuDoubleComplex old_p = cuCmul(gij, cuCmul(a_i_old, cuConj(a_j_old)));
                cuDoubleComplex new_p = cuCmul(gij, cuCmul(a_i_new, cuConj(a_j_new)));
                local_dE -= (cuCreal(new_p) - cuCreal(old_p));
            }
        }

        // --- H4: three-type enumeration, full old/new for all types ---
        for (long long t = tid; t < n_h4; t += bdim) {
            int idx[4];

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

            // Look up coupling once (shared across all 3 channels)
            cuDoubleComplex gq = g4[q];
            uint8_t qmask = g4_mask[q];
            // Skip if coupling is zero or all channels filtered
            if ((gq.x == 0.0 && gq.y == 0.0) || qmask == 0) continue;

            // Loop over 3 interaction channels
            for (int ch = 0; ch < 3; ch++) {
                if (!(qmask & (1 << ch))) continue;

                int cm = conj_masks[ch];

                // Full old/new computation for all types
                int ids[4] = {ii, jj, kk, ll};
                cuDoubleComplex old_f[4], new_f[4];
                for (int p = 0; p < 4; p++) {
                    cuDoubleComplex oval = s_spins[ids[p]];
                    cuDoubleComplex nval;
                    if (ids[p] == i0)      nval = a_i_new;
                    else if (ids[p] == j0) nval = a_j_new;
                    else                   nval = oval;
                    old_f[p] = ((cm >> p) & 1) ? cuConj(oval) : oval;
                    new_f[p] = ((cm >> p) & 1) ? cuConj(nval) : nval;
                }
                cuDoubleComplex old_prod = cuCmul(gq, cuCmul(cuCmul(old_f[0], old_f[1]),
                                                             cuCmul(old_f[2], old_f[3])));
                cuDoubleComplex new_prod = cuCmul(gq, cuCmul(cuCmul(new_f[0], new_f[1]),
                                                             cuCmul(new_f[2], new_f[3])));
                local_dE -= (cuCreal(new_prod) - cuCreal(old_prod));
            } // end channel loop
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
    const uint8_t* g4_mask,
    int N, int nrep,
    double* energies,
    int h2_active
) {
    int rep = blockIdx.y;
    if (rep >= nrep) return;

    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const cuDoubleComplex* spins = all_spins + (long long)rep * N;
    double local_sum = 0.0;

    // H2 (skipped when h2_active==0, i.e. comb FMC)
    if (h2_active) {
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
    }

    // H4: iterate over C(N,4) quartets, inner loop on 3 channels using mask
    long long nq = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    for (long long q = (long long)blockIdx.x * bdim + tid; q < nq;
         q += (long long)gridDim.x * bdim) {
        cuDoubleComplex gq = g4[q];
        uint8_t qmask = g4_mask[q];
        if ((gq.x == 0.0 && gq.y == 0.0) || qmask == 0) continue;

        int ii, jj, kk, ll;
        {
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
            ii = (int)rem; jj = j; kk = k; ll = l;
        }

        cuDoubleComplex ai = spins[ii], aj = spins[jj];
        cuDoubleComplex ak = spins[kk], al = spins[ll];

        for (int ch = 0; ch < 3; ch++) {
            if (!(qmask & (1 << ch))) continue;

            cuDoubleComplex s0, s1, s2, s3;
            if (ch == 0)      { s0 = ai; s1 = aj;         s2 = cuConj(ak); s3 = cuConj(al); }
            else if (ch == 1) { s0 = ai; s1 = cuConj(aj); s2 = cuConj(ak); s3 = al; }
            else              { s0 = ai; s1 = cuConj(aj); s2 = ak;         s3 = cuConj(al); }

            cuDoubleComplex prod = cuCmul(gq, cuCmul(cuCmul(s0, s1), cuCmul(s2, s3)));
            local_sum += -cuCreal(prod);
        }
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
    CUDA_CHECK(cudaMalloc(&state.d_g4_mask, nq * sizeof(uint8_t)));

    // Compute effective coupling scales from alpha parametrization
    //   J2 = (1-alpha)*J,  J4 = alpha*J       (variance)
    //   J2_0 = (1-alpha0)*J0,  J4_0 = alpha0*J0  (mean)
    double J2   = (1.0 - cfg.alpha)  * cfg.J;
    double J4   = cfg.alpha           * cfg.J;
    double J2_0 = (1.0 - cfg.alpha0) * cfg.J0;
    double J4_0 = cfg.alpha0          * cfg.J0;

    // Generate disorder (same for all replicas, use master seed)
    // With comb frequencies (fmc_mode==1), H2 pairs never survive FMC → skip g2
    if (cfg.fmc_mode != 1)
        generate_g2(state.d_g2, N, J2, cfg.seed + 1000);
    generate_g4(state.d_g4, N, J4, cfg.seed + 2000);
    init_g4_mask(state.d_g4_mask, N);

    // FMC filtering
    if (cfg.fmc_mode > 0) {
        state.h_omega = new double[N];
        if (cfg.fmc_mode == 1) { // comb: omega_k = k (interi, come nel legacy)
            for (int i = 0; i < N; i++) state.h_omega[i] = (double)i;
        } else { // uniform: omega_i ~ U[0,1], then sort
            uint64_t freq_state = cfg.seed + 3000;
            for (int i = 0; i < N; i++) {
                uint64_t z = splitmix64(&freq_state);
                state.h_omega[i] = (double)(z >> 11) / (double)(1ULL << 53);
            }
            std::sort(state.h_omega, state.h_omega + N);
        }
        double* d_omega;
        CUDA_CHECK(cudaMalloc(&d_omega, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_omega, state.h_omega, N * sizeof(double), cudaMemcpyHostToDevice));

        // Comb mode: g2 is identically zero, skip both filter and counting
        if (cfg.fmc_mode == 1) {
            state.n_pairs_active = 0;
        } else {
            apply_fmc_g2(state.d_g2, N, d_omega, cfg.gamma);
            cuDoubleComplex* h_g2 = new cuDoubleComplex[N * N];
            CUDA_CHECK(cudaMemcpy(h_g2, state.d_g2, (long long)N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            state.n_pairs_active = 0;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++)
                    if (h_g2[i * N + j].x != 0.0 || h_g2[i * N + j].y != 0.0)
                        state.n_pairs_active++;
            delete[] h_g2;
        }

        apply_fmc_g4(state.d_g4_mask, N, d_omega, cfg.gamma);
        CUDA_CHECK(cudaFree(d_omega));

        state.n_quart_active = count_active_g4(state.d_g4_mask, N);
    } else {
        state.h_omega = nullptr;
        state.n_pairs_active = n_pairs(N);
        state.n_quart_active = n_g4_total(N);  // all 3 channels active
    }

    // Rescale couplings: Var = J_eff^2 * N / n_surviving
    rescale_g2(state.d_g2, N, J2, state.n_pairs_active);
    rescale_g4(state.d_g4, N, J4, state.n_quart_active);

    // Add mean shift: mean = J0_eff * N / n_surviving
    if (J2_0 != 0.0 && state.n_pairs_active > 0) {
        double mean_g2 = J2_0 * (double)N / (double)state.n_pairs_active;
        shift_mean_couplings(state.d_g2, (long long)N * N, mean_g2);
    }
    if (J4_0 != 0.0 && state.n_quart_active > 0) {
        double mean_g4 = J4_0 * (double)N / (double)state.n_quart_active;
        shift_mean_couplings(state.d_g4, n_quartets(N), mean_g4);
    }

    state.h2_active = (state.n_pairs_active > 0) ? 1 : 0;

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
        state.d_spins, state.d_g2, state.d_g4, state.d_g4_mask,
        N, nrep, state.d_energies,
        state.h2_active);
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
    if (state.d_g4_mask)  CUDA_CHECK(cudaFree(state.d_g4_mask));
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

// ============================================================================
// Compute optimal block size for mc_sweep_kernel via occupancy API.
// shared_bytes depends on block_size (nwarps = block_size / 32), so we
// manually probe candidate sizes with cudaOccupancyMaxActiveBlocksPerMultiprocessor.
// Result is cached per N value.
// ============================================================================
static size_t sweep_smem(int N, int bs) {
    int nw = bs >> 5;
    return (size_t)(N + 2) * sizeof(cuDoubleComplex)
         + (size_t)nw * sizeof(double)
         + 2 * sizeof(int);
}

static int optimal_block_size_for_sweep(int N) {
    static int cached_N = -1;
    static int cached_bs = 256;
    if (N == cached_N) return cached_bs;

    static const int candidates[] = {32, 64, 128, 256, 512, 1024};
    int best_bs = 256;
    int best_occupancy = 0;
    for (int c = 0; c < 6; ++c) {
        int bs = candidates[c];
        int num_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks, mc_sweep_kernel,
            bs, sweep_smem(N, bs));
        int occupancy = num_blocks * bs;  // active threads per SM
        if (occupancy > best_occupancy) {
            best_occupancy = occupancy;
            best_bs = bs;
        }
    }

    if (best_bs < 32) best_bs = 32;

    printf("[auto-tune] mc_sweep_kernel  N=%d  -> block_size=%d\n", N, best_bs);

    cached_N  = N;
    cached_bs = best_bs;
    return best_bs;
}

void mc_sweep(MCState& state, const SimConfig& cfg) {
    int N = state.N;
    int nrep = state.nrep;
    double beta = 1.0 / cfg.T;

    // Fill per-replica betas with uniform value
    fill_double_kernel<<<(nrep + 255) / 256, 256>>>(state.d_betas, beta, nrep);

    int block_size = optimal_block_size_for_sweep(N);
    int nwarps = block_size >> 5;
    size_t shared_bytes = (N + 2) * sizeof(cuDoubleComplex) + nwarps * sizeof(double)
                        + 2 * sizeof(int);

    mc_sweep_kernel<<<nrep, block_size, shared_bytes>>>(
        state.d_spins, state.d_g2, state.d_g4, state.d_g4_mask,
        N, nrep, state.d_betas,
        state.d_rng, state.d_energies,
        state.d_accepted, state.d_proposed,
        state.h2_active);
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

    int block_size = optimal_block_size_for_sweep(N);
    int nwarps = block_size >> 5;
    size_t shared_bytes = (N + 2) * sizeof(cuDoubleComplex) + nwarps * sizeof(double)
                        + 2 * sizeof(int);

    mc_sweep_kernel<<<nrep, block_size, shared_bytes>>>(
        state.d_spins, state.d_g2, state.d_g4, state.d_g4_mask,
        N, nrep, state.d_betas,
        state.d_rng, state.d_energies,
        state.d_accepted, state.d_proposed,
        state.h2_active);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mc_set_betas(MCState& state, const double* h_betas) {
    CUDA_CHECK(cudaMemcpy(state.d_betas, h_betas,
                          state.nrep * sizeof(double), cudaMemcpyHostToDevice));
}
