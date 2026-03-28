#include "hamiltonian.h"
#include "disorder.h"
#include <cstdio>
#include <cmath>

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
// H2 = -sum_{i<j} [ g2_ij * a_i * conj(a_j) + conj(g2_ij * a_i * conj(a_j)) ]
//     = -2 * sum_{i<j} Re[ g2_ij * a_i * conj(a_j) ]
__global__ void energy_h2_kernel(const cuDoubleComplex* spins,
                                  const cuDoubleComplex* g2,
                                  int N, double* partial_sums) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long total_pairs = (long long)N * (N - 1) / 2;
    double local_sum = 0.0;

    if (pair_idx < total_pairs) {
        // Map linear index to (i,j) with i < j
        // Using the inverse triangular number formula
        int j = (int)(0.5 + sqrt(0.25 + 2.0 * pair_idx));
        int i = pair_idx - (long long)j * (j - 1) / 2;
        if (i >= j) { j++; i = pair_idx - (long long)j * (j - 1) / 2; }

        cuDoubleComplex gij = g2[i * N + j];
        cuDoubleComplex ai = spins[i];
        cuDoubleComplex aj_conj = cuConj(spins[j]);

        // g2_ij * a_i * conj(a_j)
        cuDoubleComplex prod = cuCmul(gij, cuCmul(ai, aj_conj));
        local_sum = -2.0 * cuCreal(prod);  // -( prod + conj(prod) ) = -2*Re(prod)
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

// Kernel: compute H4 partial sums
// H4 = -sum_{i<j<k<l} [ g4_ijkl * a_i * conj(a_j) * a_k * conj(a_l) + c.c. ]
//     = -2 * sum Re[ g4_ijkl * a_i * conj(a_j) * a_k * conj(a_l) ]
__global__ void energy_h4_kernel(const cuDoubleComplex* spins,
                                  const cuDoubleComplex* g4,
                                  int N, double* partial_sums) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long long q_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long total_quartets = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    double local_sum = 0.0;

    if (q_idx < total_quartets) {
        // Map linear index to (i,j,k,l) with i<j<k<l
        // Decode using combinatorial number system
        int ii, jj, kk, ll;

        // Find ll: largest such that C(ll,4) <= q_idx
        ll = 3;
        while ((long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24 <= q_idx) ll++;
        ll--;
        long long rem = q_idx - (long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24;

        // Find kk: largest such that C(kk,3) <= rem
        kk = 2;
        while ((long long)kk * (kk - 1) * (kk - 2) / 6 <= rem) kk++;
        kk--;
        rem -= (long long)kk * (kk - 1) * (kk - 2) / 6;

        // Find jj: largest such that C(jj,2) <= rem
        jj = 1;
        while ((long long)jj * (jj - 1) / 2 <= rem) jj++;
        jj--;
        rem -= (long long)jj * (jj - 1) / 2;

        ii = (int)rem;

        cuDoubleComplex gq = g4[q_idx];
        cuDoubleComplex ai = spins[ii];
        cuDoubleComplex aj_conj = cuConj(spins[jj]);
        cuDoubleComplex ak = spins[kk];
        cuDoubleComplex al_conj = cuConj(spins[ll]);

        // g4 * a_i * conj(a_j) * a_k * conj(a_l)
        cuDoubleComplex prod = cuCmul(gq, cuCmul(cuCmul(ai, aj_conj), cuCmul(ak, al_conj)));
        local_sum = -2.0 * cuCreal(prod);
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

double compute_energy(const cuDoubleComplex* d_spins,
                      const cuDoubleComplex* d_g2,
                      const cuDoubleComplex* d_g4,
                      int N) {
    const int block_size = 256;
    double energy = 0.0;

    // H2 contribution
    {
        long long np = n_pairs(N);
        int n_blocks = (int)((np + block_size - 1) / block_size);
        double* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(double)));

        energy_h2_kernel<<<n_blocks, block_size, block_size * sizeof(double)>>>(
            d_spins, d_g2, N, d_partial);
        CUDA_CHECK(cudaDeviceSynchronize());

        energy += reduce_partial_sums(d_partial, n_blocks);
        CUDA_CHECK(cudaFree(d_partial));
    }

    // H4 contribution
    {
        long long nq = n_quartets(N);
        int n_blocks = (int)((nq + block_size - 1) / block_size);
        double* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(double)));

        energy_h4_kernel<<<n_blocks, block_size, block_size * sizeof(double)>>>(
            d_spins, d_g4, N, d_partial);
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

        // Terms involving i0: g2[i0,k] or g2[k,i0]
        if (k != i0 && k != j0) {
            // Pair (min(i0,k), max(i0,k))
            int r = (i0 < k) ? i0 : k;
            int c = (i0 < k) ? k : i0;
            cuDoubleComplex gik = g2[r * N + c];

            // Old contribution: -2 Re[ g * a_i * conj(a_k) ]
            cuDoubleComplex old_prod = cuCmul(gik, cuCmul(a_i_old, cuConj(a_k)));
            cuDoubleComplex new_prod = cuCmul(gik, cuCmul(a_i_new, cuConj(a_k)));
            local_dE += -2.0 * (cuCreal(new_prod) - cuCreal(old_prod));

            // Pair (min(j0,k), max(j0,k))
            r = (j0 < k) ? j0 : k;
            c = (j0 < k) ? k : j0;
            cuDoubleComplex gjk = g2[r * N + c];

            old_prod = cuCmul(gjk, cuCmul(a_j_old, cuConj(a_k)));
            new_prod = cuCmul(gjk, cuCmul(a_j_new, cuConj(a_k)));
            local_dE += -2.0 * (cuCreal(new_prod) - cuCreal(old_prod));
        }

        // The (i0, j0) pair itself: only count once (thread k == 0 handles it)
        if (k == 0) {
            int r = (i0 < j0) ? i0 : j0;
            int c = (i0 < j0) ? j0 : i0;
            cuDoubleComplex gij = g2[r * N + c];

            cuDoubleComplex old_prod = cuCmul(gij, cuCmul(a_i_old, cuConj(a_j_old)));
            cuDoubleComplex new_prod = cuCmul(gij, cuCmul(a_i_new, cuConj(a_j_new)));
            local_dE += -2.0 * (cuCreal(new_prod) - cuCreal(old_prod));
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
// This iterates over ALL quadruplets that contain i0 or j0 (or both).
__global__ void delta_e_h4_kernel(
    const cuDoubleComplex* spins,
    const cuDoubleComplex* g4,
    int N, int i0, int j0,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    double* partial_sums
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long long q_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long total_quartets = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    double local_dE = 0.0;

    if (q_idx < total_quartets) {
        // Decode quadruplet index to (ii, jj, kk, ll) with ii < jj < kk < ll
        int ii, jj, kk, ll;

        ll = 3;
        while ((long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24 <= q_idx) ll++;
        ll--;
        long long rem = q_idx - (long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24;

        kk = 2;
        while ((long long)kk * (kk - 1) * (kk - 2) / 6 <= rem) kk++;
        kk--;
        rem -= (long long)kk * (kk - 1) * (kk - 2) / 6;

        jj = 1;
        while ((long long)jj * (jj - 1) / 2 <= rem) jj++;
        jj--;
        rem -= (long long)jj * (jj - 1) / 2;

        ii = (int)rem;

        // Check if this quadruplet involves i0 or j0
        bool involves = (ii == i0 || jj == i0 || kk == i0 || ll == i0 ||
                         ii == j0 || jj == j0 || kk == j0 || ll == j0);

        if (involves) {
            cuDoubleComplex gq = g4[q_idx];

            // Old contribution
            cuDoubleComplex oi = spins[ii], oj = spins[jj];
            cuDoubleComplex ok = cuConj(spins[kk]), ol = cuConj(spins[ll]);
            cuDoubleComplex old_prod = cuCmul(gq, cuCmul(cuCmul(oi, oj), cuCmul(ok, ol)));

            // New contribution: replace spins i0 and j0 with new values
            cuDoubleComplex ni = (ii == i0) ? a_i_new : ((ii == j0) ? a_j_new : spins[ii]);
            cuDoubleComplex nj = (jj == i0) ? a_i_new : ((jj == j0) ? a_j_new : spins[jj]);
            cuDoubleComplex nk = (kk == i0) ? a_i_new : ((kk == j0) ? a_j_new : spins[kk]);
            cuDoubleComplex nl = (ll == i0) ? a_i_new : ((ll == j0) ? a_j_new : spins[ll]);
            nk = cuConj(nk);
            nl = cuConj(nl);

            cuDoubleComplex new_prod = cuCmul(gq, cuCmul(cuCmul(ni, nj), cuCmul(nk, nl)));

            local_dE = -2.0 * (cuCreal(new_prod) - cuCreal(old_prod));
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

double compute_delta_E_pair(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    int N,
    int i, int j,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    double* d_workspace
) {
    const int block_size = 256;
    double dE = 0.0;

    // H2 delta
    {
        int n_blocks = (N + block_size - 1) / block_size;
        delta_e_h2_kernel<<<n_blocks, block_size, block_size * sizeof(double)>>>(
            d_spins, d_g2, N, i, j, a_i_new, a_j_new, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
        dE += reduce_partial_sums(d_workspace, n_blocks);
    }

    // H4 delta
    {
        long long nq = n_quartets(N);
        int n_blocks = (int)((nq + block_size - 1) / block_size);

        // Reuse workspace (ensure it's large enough)
        delta_e_h4_kernel<<<n_blocks, block_size, block_size * sizeof(double)>>>(
            d_spins, d_g4, N, i, j, a_i_new, a_j_new, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
        dE += reduce_partial_sums(d_workspace, n_blocks);
    }

    return dE;
}
