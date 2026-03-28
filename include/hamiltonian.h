#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <cuComplex.h>

// Compute the full energy H = H2 + H4
// H2 = -sum_{i<j} g2_ij a_i a*_j + c.c.
// H4 = -sum_{i<j<k<l} g4_ijkl a_i a*_j a_k a*_l + c.c.
double compute_energy(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    int N
);

// Compute the change in energy when spins i,j are updated.
// This is the key kernel: it computes delta_E for the pair move.
// delta_E = E(new) - E(old), considering all terms that involve i or j.
double compute_delta_E_pair(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    int N,
    int i, int j,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    // Workspace for GPU reduction
    double* d_workspace
);

#endif
