#include "spins_sparse.h"
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

// Normalize spins so that sum |a_i|^4 = N  (smoothed-cube constraint)
// Single-block kernel (sufficient for N <= 1024)
__global__ void normalize_spins_cube_kernel(cuDoubleComplex* spins, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < N) {
        double re = cuCreal(spins[i]);
        double im = cuCimag(spins[i]);
        double r2 = re * re + im * im;
        local_sum = r2 * r2;   // |a_i|^4
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        double sum_r4 = sdata[0];
        // scale = (N / sum_r4)^{1/4}
        sdata[0] = pow((double)N / sum_r4, 0.25);
    }
    __syncthreads();

    double scale = sdata[0];
    if (i < N) {
        double re = cuCreal(spins[i]) * scale;
        double im = cuCimag(spins[i]) * scale;
        spins[i] = make_cuDoubleComplex(re, im);
    }
}

void init_spins_cube(cuDoubleComplex* d_spins, int N, unsigned long long seed) {
    // Generate 2*N Gaussian random numbers (re, im for each spin)
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    double* d_raw = reinterpret_cast<double*>(d_spins);
    long long count = 2 * (long long)N;
    if (count % 2 != 0) count++;  // cuRAND needs even count
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, count, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Project onto the smoothed-cube surface: sum |a_i|^4 = N
    int block_size = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;
    if (block_size < 32) block_size = 32;
    normalize_spins_cube_kernel<<<1, block_size, block_size * sizeof(double)>>>(d_spins, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
