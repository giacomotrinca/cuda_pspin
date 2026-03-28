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

    // Variance of g2_ij: J^2 * 2! / (2*N) = J^2 / N
    // Each component (re, im) gets variance J^2/(2N) so sigma_component = J/sqrt(2N)
    // But we want variance of |g2|^2 = J^2/N, so sigma per component = J/sqrt(2N)
    double sigma = J / sqrt(2.0 * N);

    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    scale_couplings_kernel<<<(int)grid_size, block_size>>>(d_g2, n, sigma);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void generate_g4(cuDoubleComplex* d_g4, int N, double J, unsigned long long seed) {
    long long nq = n_quartets(N);

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed + 12345));

    double* d_raw = reinterpret_cast<double*>(d_g4);
    // Need 2*nq doubles; pad to even if necessary
    long long count = 2 * nq;
    if (count % 2 != 0) count++;
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, count, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Variance of g4_ijkl: J^2 * 4! / (2 * N^3) = 12 * J^2 / N^3
    // sigma per component = sqrt(12 * J^2 / (2 * N^3)) = J * sqrt(6 / N^3)
    double sigma = J * sqrt(6.0 / ((double)N * N * N));

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
        }
    }
}

__global__ void fmc_filter_g4_kernel(cuDoubleComplex* g4, const double* omega,
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
    // FMC: for sorted ii<jj<kk<ll, condition on frequencies
    // |omega_jj - omega_ii + omega_kk - omega_ll| > gamma → zero out
    // equivalently |(omega_jj + omega_kk) - (omega_ii + omega_ll)| > gamma
    double dw = fabs(omega[jj] - omega[ii] + omega[kk] - omega[ll]);
    if (dw > gamma) {
        g4[q] = make_cuDoubleComplex(0.0, 0.0);
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
    long long nq = n_quartets(N);
    int block_size = 256;
    long long grid_size = (nq + block_size - 1) / block_size;
    fmc_filter_g4_kernel<<<(int)grid_size, block_size>>>(d_g4, d_omega, N, gamma, nq);
    CUDA_CHECK(cudaDeviceSynchronize());
}
