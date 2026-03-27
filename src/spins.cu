#include "spins.h"
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

// Kernel to normalize spins so that sum |a_i|^2 = N
__global__ void normalize_spins_kernel(cuDoubleComplex* spins, int N, double target_norm_sq) {
    // First pass: compute partial sums of |a_i|^2
    // We use a simple approach: single block reduction for moderate N
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < N) {
        double re = cuCreal(spins[i]);
        double im = cuCimag(spins[i]);
        local_sum = re * re + im * im;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block 0, thread 0 has the total (for single block)
    if (tid == 0) {
        // Store norm^2 in the last element's position temporarily
        // We'll use atomicAdd for multi-block, but for now assume single block
        double norm_sq = sdata[0];
        double scale = sqrt(target_norm_sq / norm_sq);
        // Write scale factor to shared memory
        sdata[0] = scale;
    }
    __syncthreads();

    double scale = sdata[0];
    if (i < N) {
        double re = cuCreal(spins[i]) * scale;
        double im = cuCimag(spins[i]) * scale;
        spins[i] = make_cuDoubleComplex(re, im);
    }
}

void init_spins_uniform(cuDoubleComplex* d_spins, int N, unsigned long long seed) {
    // Generate 2*N random normal numbers (real and imaginary parts)
    // Then project onto the hypersphere

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // Generate as doubles: treat d_spins as array of 2*N doubles
    double* d_raw = reinterpret_cast<double*>(d_spins);
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_raw, 2 * N, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Normalize to the hypersphere: sum |a_i|^2 = N
    int block_size = (N < 1024) ? N : 1024;
    // For simplicity, use a single block if N <= 1024
    if (N <= 1024) {
        normalize_spins_kernel<<<1, block_size, block_size * sizeof(double)>>>(
            d_spins, N, (double)N
        );
    } else {
        // Multi-block: compute norm on host, then scale
        cuDoubleComplex* h_spins = new cuDoubleComplex[N];
        CUDA_CHECK(cudaMemcpy(h_spins, d_spins, N * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double norm_sq = 0.0;
        for (int i = 0; i < N; i++) {
            double re = cuCreal(h_spins[i]);
            double im = cuCimag(h_spins[i]);
            norm_sq += re * re + im * im;
        }
        double scale = sqrt((double)N / norm_sq);
        for (int i = 0; i < N; i++) {
            double re = cuCreal(h_spins[i]) * scale;
            double im = cuCimag(h_spins[i]) * scale;
            h_spins[i] = make_cuDoubleComplex(re, im);
        }
        CUDA_CHECK(cudaMemcpy(d_spins, h_spins, N * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        delete[] h_spins;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Device function: propose pair rotation preserving |a_i|^2 + |a_j|^2
__device__ void propose_pair_rotation(
    cuDoubleComplex a_i, cuDoubleComplex a_j,
    double delta,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* a_i_new, cuDoubleComplex* a_j_new
) {
    // Rotation in the 2D complex subspace spanned by (a_i, a_j)
    // preserves |a_i|^2 + |a_j|^2, hence the spherical constraint.
    //
    // We use a U(2) rotation parameterized by a small angle:
    //   (a_i', a_j') = (cos(theta)*a_i + e^{i*phi}*sin(theta)*a_j,
    //                   -e^{-i*phi}*sin(theta)*a_i + cos(theta)*a_j)

    float4 rnd = curand_uniform4(rng_state);
    double theta = delta * (rnd.x - 0.5);  // small angle
    double phi = 2.0 * M_PI * rnd.y;       // random phase

    double ct = cos(theta);
    double st = sin(theta);
    double cp = cos(phi);
    double sp = sin(phi);

    // e^{i*phi} = cp + i*sp
    // New a_i = ct * a_i + (cp + i*sp) * st * a_j
    cuDoubleComplex eiphi = make_cuDoubleComplex(cp, sp);
    cuDoubleComplex emiphi = make_cuDoubleComplex(cp, -sp);

    // a_i_new = ct * a_i + st * e^{iphi} * a_j
    cuDoubleComplex term1 = make_cuDoubleComplex(ct * cuCreal(a_i), ct * cuCimag(a_i));
    cuDoubleComplex rot_aj = cuCmul(eiphi, a_j);
    cuDoubleComplex term2 = make_cuDoubleComplex(st * cuCreal(rot_aj), st * cuCimag(rot_aj));
    *a_i_new = cuCadd(term1, term2);

    // a_j_new = -st * e^{-iphi} * a_i + ct * a_j
    cuDoubleComplex rot_ai = cuCmul(emiphi, a_i);
    cuDoubleComplex term3 = make_cuDoubleComplex(-st * cuCreal(rot_ai), -st * cuCimag(rot_ai));
    cuDoubleComplex term4 = make_cuDoubleComplex(ct * cuCreal(a_j), ct * cuCimag(a_j));
    *a_j_new = cuCadd(term3, term4);
}
