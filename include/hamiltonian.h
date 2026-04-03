#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <cuComplex.h>
#include <cstdint>

// Compute the full energy H = H2 + H4
// H2 = -sum_{i<j} Re[ g2_ij a_i a*_j ]
// H4 = -sum_{i<j<k<l} Re[ g4_ijkl a_i a*_j a_k a*_l ]
// g4 has C(N,4) entries; g4_mask has C(N,4) uint8_t bitmasks (3 bits for channels)
double compute_energy(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    const uint8_t* d_g4_mask,
    int N
);

// Compute the change in energy when spins i,j are updated.
// delta_E = E(new) - E(old), considering all terms that involve i or j.
double compute_delta_E_pair(
    const cuDoubleComplex* d_spins,
    const cuDoubleComplex* d_g2,
    const cuDoubleComplex* d_g4,
    const uint8_t* d_g4_mask,
    int N,
    int i, int j,
    cuDoubleComplex a_i_new, cuDoubleComplex a_j_new,
    // Workspace for GPU reduction
    double* d_workspace
);

#endif
