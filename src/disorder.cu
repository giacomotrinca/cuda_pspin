#include "disorder.h"
#include <curand.h>
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
    long long ntot = n_g4_total(N);  // 3 * C(N,4)

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed + 12345));

    double* d_raw = reinterpret_cast<double*>(d_g4);
    // Need 2*ntot doubles; pad to even if necessary
    long long count = 2 * ntot;
    if (count % 2 != 0) count++;
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, count, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Generate with unit variance; actual scaling deferred until after FMC filter
    int block_size = 256;
    long long grid_size = (ntot + block_size - 1) / block_size;
    // Zero out imaginary parts: g4 are real
    zero_imag_kernel<<<(int)grid_size, block_size>>>(d_g4, ntot);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void rescale_g4(cuDoubleComplex* d_g4, int N, double J, long long n_surviving) {
    if (n_surviving <= 0) return;  // all couplings zeroed by FMC, nothing to rescale
    long long ntot = n_g4_total(N);  // 3 * C(N,4)
    // Var(g4) = J^2 * N / n_surviving (total across all 3 channels)
    double sigma = J * sqrt((double)N / (double)n_surviving);
    int block_size = 256;
    long long grid_size = (ntot + block_size - 1) / block_size;
    scale_couplings_kernel<<<(int)grid_size, block_size>>>(d_g4, ntot, sigma);
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
        }
    }
}

// 3-channel FMC filter: processes 3*C(N,4) entries.
// Layout: [ch0: C(N,4) | ch1: C(N,4) | ch2: C(N,4)]
// Channel-dependent conservation conditions (sorted ii<jj<kk<ll):
//   ch 0: |w_ii+w_jj - w_kk-w_ll| <= gamma
//   ch 1: |w_ii+w_ll - w_jj-w_kk| <= gamma
//   ch 2: |w_ii+w_kk - w_jj-w_ll| <= gamma
__global__ void fmc_filter_g4_kernel(cuDoubleComplex* g4, const double* omega,
                                     int N, double gamma, long long nq_per_ch) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nq_total = 3 * nq_per_ch;
    if (idx >= nq_total) return;

    // Determine channel and quartet index
    int ch;
    long long q;
    if (idx < nq_per_ch)          { ch = 0; q = idx; }
    else if (idx < 2 * nq_per_ch) { ch = 1; q = idx - nq_per_ch; }
    else                          { ch = 2; q = idx - 2 * nq_per_ch; }

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

    // FMC condition depends on channel
    double dw;
    if (ch == 0)      dw = fabs(omega[ii] + omega[jj] - omega[kk] - omega[ll]);
    else if (ch == 1) dw = fabs(omega[ii] + omega[ll] - omega[jj] - omega[kk]);
    else              dw = fabs(omega[ii] + omega[kk] - omega[jj] - omega[ll]);

    if (dw > gamma) {
        g4[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

void apply_fmc_g2(cuDoubleComplex* d_g2, int N, const double* d_omega, double gamma) {
    int total = N * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    fmc_filter_g2_kernel<<<grid_size, block_size>>>(d_g2, d_omega, N, gamma);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void apply_fmc_g4(cuDoubleComplex* d_g4, int N, const double* d_omega, double gamma) {
    long long nq_per_ch = n_quartets(N);
    long long ntot = 3 * nq_per_ch;
    int block_size = 256;
    long long grid_size = (ntot + block_size - 1) / block_size;
    fmc_filter_g4_kernel<<<(int)grid_size, block_size>>>(d_g4, d_omega, N, gamma, nq_per_ch);
    CUDA_CHECK(cudaDeviceSynchronize());
}
