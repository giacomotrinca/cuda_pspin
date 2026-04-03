#include "disorder.h"
#include <curand.h>
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

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)

// Kernel to scale generated normals to the correct variance
__global__ void scale_couplings_kernel(cuDoubleComplex* couplings, long long n, double sigma) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double re = cuCreal(couplings[idx]) * sigma;
        double im = cuCimag(couplings[idx]) * sigma;
        couplings[idx] = make_cuDoubleComplex(re, im);
    }
}

// Kernel to zero out imaginary parts (couplings are real)
__global__ void zero_imag_kernel(cuDoubleComplex* couplings, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        couplings[idx] = make_cuDoubleComplex(cuCreal(couplings[idx]), 0.0);
    }
}

// Kernel to symmetrize g2: copy upper triangle to lower, zero diagonal
__global__ void symmetrize_g2_kernel(cuDoubleComplex* g2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    if (i == j) {
        g2[idx] = make_cuDoubleComplex(0.0, 0.0);
    } else if (i > j) {
        g2[i * N + j] = g2[j * N + i];
    }
}

void generate_g2(cuDoubleComplex* d_g2, int N, double J, unsigned long long seed) {
    long long n = (long long)N * N;

    // Generate standard normal complex numbers
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    double* d_raw = reinterpret_cast<double*>(d_g2);
    // curandGenerateNormalDouble needs even count; 2*N*N is always even for N>=2
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, 2 * n, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Generate with unit variance; actual scaling deferred until after FMC filter
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    // Zero out imaginary parts: g2 are real
    zero_imag_kernel<<<(int)grid_size, block_size>>>(d_g2, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Symmetrize: g2[j*N+i] = g2[i*N+j], zero diagonal (no self-interaction)
    symmetrize_g2_kernel<<<(int)grid_size, block_size>>>(d_g2, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void rescale_g2(cuDoubleComplex* d_g2, int N, double J, long long n_surviving) {
    if (n_surviving <= 0) return;  // all couplings zeroed by FMC, nothing to rescale
    long long n = (long long)N * N;
    // Var(g2) = J^2 * N / n_surviving
    double sigma = J * sqrt((double)N / (double)n_surviving);
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    scale_couplings_kernel<<<(int)grid_size, block_size>>>(d_g2, n, sigma);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void generate_g4(cuDoubleComplex* d_g4, int N, double J, unsigned long long seed) {
    long long nq   = n_quartets(N);   // C(N,4)

    // Generate a single coupling per quartet (symmetric tensor: g_{ijkl}
    // is the same regardless of which indices are conjugated).
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed + 12345));

    double* d_raw = reinterpret_cast<double*>(d_g4);
    long long count = 2 * nq;
    if (count % 2 != 0) count++;
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, count, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    int block_size = 256;
    long long grid_size = (nq + block_size - 1) / block_size;
    // Zero out imaginary parts: g4 are real
    zero_imag_kernel<<<(int)grid_size, block_size>>>(d_g4, nq);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void rescale_g4(cuDoubleComplex* d_g4, int N, double J, long long n_surviving) {
    if (n_surviving <= 0) return;  // all couplings zeroed by FMC, nothing to rescale
    long long nq = n_quartets(N);  // C(N,4)
    // Var(g4) = J^2 * N / n_surviving (n_surviving = total active quartet-channel pairs)
    double sigma = J * sqrt((double)N / (double)n_surviving);
    int block_size = 256;
    long long grid_size = (nq + block_size - 1) / block_size;
    scale_couplings_kernel<<<(int)grid_size, block_size>>>(d_g4, nq, sigma);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// FMC filter kernels
// ============================================================================

__global__ void fmc_filter_g2_kernel(cuDoubleComplex* g2, const double* omega,
                                     int N, double gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    if (i < j) {
        double dw = fabs(omega[i] - omega[j]);
        if (dw > gamma) {
            g2[i * N + j] = make_cuDoubleComplex(0.0, 0.0);
            g2[j * N + i] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

// 3-channel FMC mask filter: processes C(N,4) quartets.
// For each quartet, clears the mask bit of channels whose FMC condition
// is not satisfied, leaving the coupling array untouched.
// Channel-dependent conservation conditions (sorted ii<jj<kk<ll):
//   ch 0: |w_ii+w_jj - w_kk-w_ll| <= gamma
//   ch 1: |w_ii+w_ll - w_jj-w_kk| <= gamma
//   ch 2: |w_ii+w_kk - w_jj-w_ll| <= gamma
__global__ void fmc_filter_g4_mask_kernel(uint8_t* mask, const double* omega,
                                          int N, double gamma, long long nq) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;

    // Decode combinatorial index to sorted (ii, jj, kk, ll) with ii < jj < kk < ll
    int ll = 3;
    while ((long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24 <= q) ll++;
    ll--;
    long long rem = q - (long long)ll * (ll - 1) * (ll - 2) * (ll - 3) / 24;
    int kk = 2;
    while ((long long)kk * (kk - 1) * (kk - 2) / 6 <= rem) kk++;
    kk--;
    rem -= (long long)kk * (kk - 1) * (kk - 2) / 6;
    int jj = 1;
    while ((long long)jj * (jj - 1) / 2 <= rem) jj++;
    jj--;
    rem -= (long long)jj * (jj - 1) / 2;
    int ii = (int)rem;

    uint8_t m = mask[q];

    // ch 0: |w_ii + w_jj - w_kk - w_ll|
    if (fabs(omega[ii] + omega[jj] - omega[kk] - omega[ll]) > gamma)
        m &= ~(1u << 0);
    // ch 1: |w_ii + w_ll - w_jj - w_kk|
    if (fabs(omega[ii] + omega[ll] - omega[jj] - omega[kk]) > gamma)
        m &= ~(1u << 1);
    // ch 2: |w_ii + w_kk - w_jj - w_ll|
    if (fabs(omega[ii] + omega[kk] - omega[jj] - omega[ll]) > gamma)
        m &= ~(1u << 2);

    mask[q] = m;
}

// Kernel: set all mask entries to 0x7 (all 3 channels active)
__global__ void init_g4_mask_kernel(uint8_t* mask, long long nq) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nq) mask[idx] = 0x7;
}

void apply_fmc_g2(cuDoubleComplex* d_g2, int N, const double* d_omega, double gamma) {
    int total = N * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    fmc_filter_g2_kernel<<<grid_size, block_size>>>(d_g2, d_omega, N, gamma);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel: shift real part of non-zero entries by +mean
__global__ void shift_mean_kernel(cuDoubleComplex* c, long long n, double mean) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double re = cuCreal(c[idx]);
        double im = cuCimag(c[idx]);
        if (re != 0.0 || im != 0.0)
            c[idx] = make_cuDoubleComplex(re + mean, im);
    }
}

void shift_mean_couplings(cuDoubleComplex* d_c, long long n, double mean) {
    if (n <= 0 || mean == 0.0) return;
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    shift_mean_kernel<<<(int)grid_size, block_size>>>(d_c, n, mean);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void init_g4_mask(uint8_t* d_mask, int N) {
    long long nq = n_quartets(N);
    int block_size = 256;
    long long grid_size = (nq + block_size - 1) / block_size;
    init_g4_mask_kernel<<<(int)grid_size, block_size>>>(d_mask, nq);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void apply_fmc_g4(uint8_t* d_mask, int N, const double* d_omega, double gamma) {
    long long nq = n_quartets(N);
    int block_size = 256;
    long long grid_size = (nq + block_size - 1) / block_size;
    fmc_filter_g4_mask_kernel<<<(int)grid_size, block_size>>>(d_mask, d_omega, N, gamma, nq);
    CUDA_CHECK(cudaDeviceSynchronize());
}

long long count_active_g4(const uint8_t* d_mask, int N) {
    long long nq = n_quartets(N);
    uint8_t* h_mask = new uint8_t[nq];
    CUDA_CHECK(cudaMemcpy(h_mask, d_mask, nq * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    long long count = 0;
    for (long long q = 0; q < nq; q++)
        count += __builtin_popcount(h_mask[q]);
    delete[] h_mask;
    return count;
}
