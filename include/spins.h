#ifndef SPINS_H
#define SPINS_H

#include <cuComplex.h>
#include <curand_kernel.h>

// Initialize N complex spins uniformly on the hypersphere |a|^2 = N
// d_spins must be pre-allocated (N * sizeof(cuDoubleComplex))
void init_spins_uniform(cuDoubleComplex* d_spins, int N, unsigned long long seed);

// Propose a MC move: independent phase rotation + modulus redistribution.
// Matches the legacy CreateProposedUpdates move. Preserves |a_i|^2 + |a_j|^2.
//
//   a_i' = a_i * e^{-i*phi1} * sqrt((r1+r2)/r1) * cos(alpha)
//   a_j' = a_j * e^{-i*phi2} * sqrt((r1+r2)/r2) * sin(alpha)
//
inline __device__ void propose_pair_rotation(
    cuDoubleComplex a_i, cuDoubleComplex a_j,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* a_i_new, cuDoubleComplex* a_j_new
) {
    double2 rnd1 = curand_uniform2_double(rng_state);
    double  rnd2 = curand_uniform_double(rng_state);

    double alpha = 2.0 * M_PI * rnd1.x;
    double phi1  = 2.0 * M_PI * rnd1.y;
    double phi2  = 2.0 * M_PI * rnd2;

    double ca, sa, cp1, sp1, cp2, sp2;
    sincos(alpha, &sa, &ca);
    sincos(phi1,  &sp1, &cp1);
    sincos(phi2,  &sp2, &cp2);

    double x1 = cuCreal(a_i), y1 = cuCimag(a_i);
    double x2 = cuCreal(a_j), y2 = cuCimag(a_j);

    double r1 = x1 * x1 + y1 * y1;
    double r2 = x2 * x2 + y2 * y2;

    // Phase rotation: a -> a * e^{-i*phi}
    double nx1 =  x1 * cp1 + y1 * sp1;
    double ny1 = -x1 * sp1 + y1 * cp1;
    double nx2 =  x2 * cp2 + y2 * sp2;
    double ny2 = -x2 * sp2 + y2 * cp2;

    // Modulus redistribution
    double factor1 = sqrt((r1 + r2) / r1) * ca;
    double factor2 = sqrt((r1 + r2) / r2) * sa;

    *a_i_new = make_cuDoubleComplex(nx1 * factor1, ny1 * factor1);
    *a_j_new = make_cuDoubleComplex(nx2 * factor2, ny2 * factor2);
}

#endif
