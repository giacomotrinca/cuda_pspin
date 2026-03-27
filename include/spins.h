#ifndef SPINS_H
#define SPINS_H

#include <cuComplex.h>
#include <curand_kernel.h>

// Initialize N complex spins uniformly on the hypersphere |a|^2 = N
// d_spins must be pre-allocated (N * sizeof(cuDoubleComplex))
void init_spins_uniform(cuDoubleComplex* d_spins, int N, unsigned long long seed);

// Propose a MC move: rotate a pair (i,j) on the hypersphere
// Returns the proposed new values for a_i, a_j
// The move is a rotation in the (i,j) plane by a random angle,
// preserving sum |a_i|^2 + |a_j|^2.
__device__ void propose_pair_rotation(
    cuDoubleComplex a_i, cuDoubleComplex a_j,
    double delta,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* a_i_new, cuDoubleComplex* a_j_new
);

#endif
