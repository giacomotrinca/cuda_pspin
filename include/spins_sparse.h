#ifndef SPINS_SPARSE_H
#define SPINS_SPARSE_H

#include <cuComplex.h>
#include <curand_kernel.h>

// Initialize N complex spins on the smoothed-cube surface: sum |a_k|^4 = N
// Each spin gets a random Gaussian direction, then all are rescaled so that
// sum |a_k|^4 = N  (fourth-moment normalization).
void init_spins_cube(cuDoubleComplex* d_spins, int N, unsigned long long seed);

// Propose a MC move preserving the smoothed-cube constraint: |a_i|^4 + |a_j|^4.
//
//   S4 = |a_i|^4 + |a_j|^4          (conserved quantity)
//
//   Parametrize with u = |a_j|^4 / S4  ∈ [0,1].  In this variable the
//   flat measure on the constraint surface is  dμ ∝ du dφ₁ dφ₂  (no
//   Jacobian needed), so a uniform proposal in (u, φ₁, φ₂) satisfies
//   detailed balance with acceptance  min(1, exp(-β ΔE)).
//
//   |a_i'|^2 = sqrt(S4) * sqrt(1 - u)
//   |a_j'|^2 = sqrt(S4) * sqrt(u)
//
//   => |a_i'|^4 + |a_j'|^4 = S4*(1-u) + S4*u = S4.  ✓
//
inline __device__ void propose_pair_rotation_cube(
    cuDoubleComplex a_i, cuDoubleComplex a_j,
    curandStatePhilox4_32_10_t* rng_state,
    cuDoubleComplex* a_i_new, cuDoubleComplex* a_j_new
) {
    double2 rnd1 = curand_uniform2_double(rng_state);
    double  rnd2 = curand_uniform_double(rng_state);

    double u    = rnd1.x;                  // u ∈ (0,1]  uniform
    double phi1 = rnd1.y * 2.0 * M_PI;    // [0, 2π)
    double phi2 = rnd2   * 2.0 * M_PI;    // [0, 2π)

    double x1 = cuCreal(a_i), y1 = cuCimag(a_i);
    double x2 = cuCreal(a_j), y2 = cuCimag(a_j);

    double r1sq = x1 * x1 + y1 * y1;   // |a_i|^2
    double r2sq = x2 * x2 + y2 * y2;   // |a_j|^2
    double S4   = r1sq * r1sq + r2sq * r2sq;  // |a_i|^4 + |a_j|^4
    double sqrtS4 = sqrt(S4);

    double cp1, sp1, cp2, sp2;
    sincos(phi1, &sp1, &cp1);
    sincos(phi2, &sp2, &cp2);

    // New moduli: |a_i'|^2 = sqrt(S4)*sqrt(1-u),  |a_j'|^2 = sqrt(S4)*sqrt(u)
    double r1_new = sqrt(sqrtS4 * sqrt(1.0 - u));   // |a_i'|
    double r2_new = sqrt(sqrtS4 * sqrt(u));          // |a_j'|

    *a_i_new = make_cuDoubleComplex(r1_new * cp1, r1_new * sp1);
    *a_j_new = make_cuDoubleComplex(r2_new * cp2, r2_new * sp2);
}

#endif
