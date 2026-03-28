#ifndef SPINS_H
#define SPINS_H

#include <cuComplex.h>
#include <curand_kernel.h>

// Initialize N complex spins uniformly on the hypersphere |a|^2 = N
// d_spins must be pre-allocated (N * sizeof(cuDoubleComplex))
void init_spins_uniform(cuDoubleComplex* d_spins, int N, unsigned long long seed);

// Propose a MC move: rotate a pair (i,j) on the hypersphere.
// U(2) rotation in the (a_i, a_j) subspace, preserves |a_i|^2 + |a_j|^2.
inline __device__ void propose_pair_rotation(
    cuDoubleComplex a_i, cuDoubleComplex a_j,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* a_i_new, cuDoubleComplex* a_j_new
) {
    // Double-precision RNG + sincos for combined sin/cos
    double2 rnd = curand_uniform2_double(rng_state);
    double theta = 2.0 * M_PI * rnd.x;
    double phi   = 2.0 * M_PI * rnd.y;

    double ct, st, cp, sp;
    sincos(theta, &st, &ct);
    sincos(phi,   &sp, &cp);

    // Component-level U(2) rotation (avoids cuCmul overhead)
    double ai_re = cuCreal(a_i), ai_im = cuCimag(a_i);
    double aj_re = cuCreal(a_j), aj_im = cuCimag(a_j);

    // e^{i*phi} a_j
    double ep_aj_re = cp * aj_re - sp * aj_im;
    double ep_aj_im = sp * aj_re + cp * aj_im;
    // e^{-i*phi} a_i
    double em_ai_re =  cp * ai_re + sp * ai_im;
    double em_ai_im = -sp * ai_re + cp * ai_im;

    *a_i_new = make_cuDoubleComplex(ct * ai_re + st * ep_aj_re,
                                    ct * ai_im + st * ep_aj_im);
    *a_j_new = make_cuDoubleComplex(-st * em_ai_re + ct * aj_re,
                                    -st * em_ai_im + ct * aj_im);
}

#endif
