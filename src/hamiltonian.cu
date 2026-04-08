#include "hamiltonian.h"
#include "disorder.h"
#include <cstdio>
#include <cmath>
#include <cstdint>

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
// Full energy computation
// ============================================================================

// Kernel: compute H2 partial sums
// H2 = -sum_{i<=j} Re[ g2_ij * a_i * conj(a_j) ]  (includes diagonal i=j)
__global__ void energy_h2_kernel(const cuDoubleComplex* spins,
                                  const cuDoubleComplex* g2,
                                  int N, double* partial_sums) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long total_pairs = (long long)N * (N + 1) / 2;
    double local_sum = 0.0;

    if (pair_idx < total_pairs) {
        // Map linear index to (i,j) with i <= j
        // T(j) = j*(j+1)/2 pairs up to row j
        int j = (int)floor(sqrt(2.0 * pair_idx + 0.25) - 0.5);
        int i = pair_idx - (long long)j * (j + 1) / 2;
        if (i > j) { j++; i = pair_idx - (long long)j * (j + 1) / 2; }

        cuDoubleComplex gij = g2[i * N + j];
        cuDoubleComplex ai = spins[i];
        cuDoubleComplex aj_conj = cuConj(spins[j]);

        // g2_ij * a_i * conj(a_j)
        cuDoubleComplex prod = cuCmul(gij, cuCmul(ai, aj_conj));
        local_sum = -cuCreal(prod);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

// Kernel: compute H4 partial sums over C(N,4) quartets (single channel via mask)
// Channel-dependent conjugation:
//   ch 0: -Re[g4 * a_ii * a_jj * conj(a_kk) * conj(a_ll)]
//   ch 1: -Re[g4 * a_ii * conj(a_jj) * conj(a_kk) * a_ll]
//   ch 2: -Re[g4 * a_ii * conj(a_jj) * a_kk * conj(a_ll)]
__global__ void energy_h4_kernel(const cuDoubleComplex* spins,
                                  const cuDoubleComplex* g4,
                                  const uint8_t* g4_mask,
                                  int N, double* partial_sums) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long nq = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    double local_sum = 0.0;

    if (q < nq) {
        uint8_t qmask = g4_mask[q];
        if (qmask) {
            // Decode combinatorial index
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
            cuDoubleComplex ai = spins[ii], aj = spins[jj];
            cuDoubleComplex ak = spins[kk], al = spins[ll];

            for (int ch = 0; ch < 3; ch++) {
                if (!(qmask & (1 << ch))) continue;

                cuDoubleComplex s0, s1, s2, s3;
                if (ch == 0)      { s0 = ai; s1 = aj;           s2 = cuConj(ak); s3 = cuConj(al); }
                else if (ch == 1) { s0 = ai; s1 = cuConj(aj);   s2 = cuConj(ak); s3 = al; }
                else              { s0 = ai; s1 = cuConj(aj);   s2 = ak;         s3 = cuConj(al); }

                cuDoubleComplex prod = cuCmul(gq, cuCmul(cuCmul(s0, s1), cuCmul(s2, s3)));
                local_sum += -cuCreal(prod);
            }
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

// Host-side reduction of partial sums
static double reduce_partial_sums(double* d_partial, int n_blocks) {
    double* h_partial = new double[n_blocks];
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, n_blocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < n_blocks; i++) total += h_partial[i];
    delete[] h_partial;
    return total;
}

// ---------------------------------------------------------------------------
// Auto-tune block sizes via occupancy API (cached per kernel)
// ---------------------------------------------------------------------------

static int optimal_bs_h2() {
    static int bs = 0;
    if (bs) return bs;
    int mg; cudaOccupancyMaxPotentialBlockSize(&mg, &bs, energy_h2_kernel,
        (size_t)0, 0);
    if (bs < 32) bs = 32;
    // Round down to nearest power of 2: tree reduction requires it
    int p = 32; while (p * 2 <= bs) p *= 2; bs = p;
    return bs;
}
static int optimal_bs_h4() {
    static int bs = 0;
    if (bs) return bs;
    int mg; cudaOccupancyMaxPotentialBlockSize(&mg, &bs, energy_h4_kernel,
        (size_t)0, 0);
    if (bs < 32) bs = 32;
    // Round down to nearest power of 2: tree reduction requires it
    int p = 32; while (p * 2 <= bs) p *= 2; bs = p;
    return bs;
}

double compute_energy(const cuDoubleComplex* d_spins,
                      const cuDoubleComplex* d_g2,
                      const cuDoubleComplex* d_g4,
                      const uint8_t* d_g4_mask,
                      int N) {
    double energy = 0.0;

    // H2 contribution
    {
        int bs = optimal_bs_h2();
        long long np = n_pairs(N);
        int n_blocks = (int)((np + bs - 1) / bs);
        double* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(double)));

        energy_h2_kernel<<<n_blocks, bs, bs * sizeof(double)>>>(
            d_spins, d_g2, N, d_partial);
        CUDA_CHECK(cudaDeviceSynchronize());

        energy += reduce_partial_sums(d_partial, n_blocks);
        CUDA_CHECK(cudaFree(d_partial));
    }

    // H4 contribution
    {
        int bs = optimal_bs_h4();
        long long nq = n_quartets(N);  // C(N,4)
        int n_blocks = (int)((nq + bs - 1) / bs);
        double* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(double)));

        energy_h4_kernel<<<n_blocks, bs, bs * sizeof(double)>>>(
            d_spins, d_g4, d_g4_mask, N, d_partial);
        CUDA_CHECK(cudaDeviceSynchronize());

        energy += reduce_partial_sums(d_partial, n_blocks);
        CUDA_CHECK(cudaFree(d_partial));
    }

    return energy;
}

// ============================================================================
// Delta E computation for a pair update (i0, j0)
// ============================================================================

// Kernel: compute delta_E from 2-body terms involving sites i0 or j0
__global__ void delta_e_h2_kernel(
    const cuDoubleComplex* spins,
    const cuDoubleComplex* g2,
    int N, int i0, int j0,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    double* partial_sums
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    double local_dE = 0.0;

    if (k < N) {
        cuDoubleComplex a_k = spins[k];
        cuDoubleComplex a_i_old = spins[i0];
        cuDoubleComplex a_j_old = spins[j0];

        // Off-diagonal terms involving i0 or j0 (k != i0 and k != j0)
        if (k != i0 && k != j0) {
            cuDoubleComplex gik = g2[i0 * N + k];

            // Old contribution: -Re[ g * a_i * conj(a_k) ]
            cuDoubleComplex old_prod = cuCmul(gik, cuCmul(a_i_old, cuConj(a_k)));
            cuDoubleComplex new_prod = cuCmul(gik, cuCmul(a_i_new, cuConj(a_k)));
            local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));

            cuDoubleComplex gjk = g2[j0 * N + k];

            old_prod = cuCmul(gjk, cuCmul(a_j_old, cuConj(a_k)));
            new_prod = cuCmul(gjk, cuCmul(a_j_new, cuConj(a_k)));
            local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));
        }

        // Diagonal term (i0, i0): g2_ii * |a_i|^2
        if (k == i0) {
            cuDoubleComplex gii = g2[i0 * N + i0];
            cuDoubleComplex old_prod = cuCmul(gii, cuCmul(a_i_old, cuConj(a_i_old)));
            cuDoubleComplex new_prod = cuCmul(gii, cuCmul(a_i_new, cuConj(a_i_new)));
            local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));
        }

        // Diagonal term (j0, j0): g2_jj * |a_j|^2
        if (k == j0) {
            cuDoubleComplex gjj = g2[j0 * N + j0];
            cuDoubleComplex old_prod = cuCmul(gjj, cuCmul(a_j_old, cuConj(a_j_old)));
            cuDoubleComplex new_prod = cuCmul(gjj, cuCmul(a_j_new, cuConj(a_j_new)));
            local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));
        }

        // The (i0, j0) pair itself: only count once (thread k == 0 handles it)
        if (k == 0) {
            cuDoubleComplex gij = g2[i0 * N + j0];

            cuDoubleComplex old_prod = cuCmul(gij, cuCmul(a_i_old, cuConj(a_j_old)));
            cuDoubleComplex new_prod = cuCmul(gij, cuCmul(a_i_new, cuConj(a_j_new)));
            local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));
        }
    }

    sdata[tid] = local_dE;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

// Kernel: compute delta_E from 4-body terms involving sites i0 or j0
// Iterates over C(N,4) quartets; mask selects active channels.
__global__ void delta_e_h4_kernel(
    const cuDoubleComplex* spins,
    const cuDoubleComplex* g4,
    const uint8_t* g4_mask,
    int N, int i0, int j0,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    double* partial_sums
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long nq = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    double local_dE = 0.0;

    if (q < nq) {
        uint8_t qmask = g4_mask[q];
        if (qmask) {
            // Decode
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

            bool involves = (ii == i0 || jj == i0 || kk == i0 || ll == i0 ||
                             ii == j0 || jj == j0 || kk == j0 || ll == j0);

            if (involves) {
                cuDoubleComplex gq = g4[q];

                int ids[4] = {ii, jj, kk, ll};

                for (int ch = 0; ch < 3; ch++) {
                    if (!(qmask & (1 << ch))) continue;

                    // Conjugation mask: ch 0→0xC, ch 1→0x6, ch 2→0xA
                    int cm = (ch == 0) ? 0xC : ((ch == 1) ? 0x6 : 0xA);

                    cuDoubleComplex old_f[4], new_f[4];
                    for (int p = 0; p < 4; p++) {
                        cuDoubleComplex oval = spins[ids[p]];
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
                    local_dE += -(cuCreal(new_prod) - cuCreal(old_prod));
                }
            }
        }
    }

    sdata[tid] = local_dE;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

static int optimal_bs_dh2() {
    static int bs = 0;
    if (bs) return bs;
    int mg; cudaOccupancyMaxPotentialBlockSize(&mg, &bs, delta_e_h2_kernel,
        (size_t)0, 0);
    if (bs < 32) bs = 32;
    return bs;
}
static int optimal_bs_dh4() {
    static int bs = 0;
    if (bs) return bs;
    int mg; cudaOccupancyMaxPotentialBlockSize(&mg, &bs, delta_e_h4_kernel,
        (size_t)0, 0);
    if (bs < 32) bs = 32;
    return bs;
}

double compute_delta_E_pair(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    const uint8_t* d_g4_mask,
    int N,
    int i, int j,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    double* d_workspace
) {
    double dE = 0.0;

    // H2 delta
    {
        int bs = optimal_bs_dh2();
        int n_blocks = (N + bs - 1) / bs;
        delta_e_h2_kernel<<<n_blocks, bs, bs * sizeof(double)>>>(
            d_spins, d_g2, N, i, j, a_i_new, a_j_new, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
        dE += reduce_partial_sums(d_workspace, n_blocks);
    }

    // H4 delta
    {
        int bs = optimal_bs_dh4();
        long long nq = n_quartets(N);  // C(N,4)
        int n_blocks = (int)((nq + bs - 1) / bs);

        // Reuse workspace (ensure it's large enough)
        delta_e_h4_kernel<<<n_blocks, bs, bs * sizeof(double)>>>(
            d_spins, d_g4, d_g4_mask, N, i, j, a_i_new, a_j_new, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
        dE += reduce_partial_sums(d_workspace, n_blocks);
    }

    return dE;
}
