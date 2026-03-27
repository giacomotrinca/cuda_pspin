# p-Spin Spherical Model (2+4) — CUDA Monte Carlo

Monte Carlo simulation of the **complex spherical p-spin glass** with mixed 2-body and 4-body interactions.

## Model

### Hamiltonian

$$
H = -\sum_{i<j} g^{(2)}_{ij}\, a_i\, a_j^* \;-\; \sum_{i<j<k<l} g^{(4)}_{ijkl}\, a_i\, a_j\, a_k^*\, a_l^* \;+\; \text{c.c.}
$$

where the $a_i \in \mathbb{C}$ are complex spins subject to the **spherical constraint**:

$$
\sum_{i=1}^{N} |a_i|^2 = N
$$

### Disorder

The couplings are i.i.d. complex Gaussian with zero mean and variance chosen to guarantee extensivity:

| Coupling | Variance |
|----------|----------|
| $g^{(2)}_{ij}$ | $J^2 \cdot 2! / (2N)$ |
| $g^{(4)}_{ijkl}$ | $J^2 \cdot 4! / (2N^3)$ |

### Monte Carlo move

Each MC step attempts updates on **all pairs** $(i,j)$ and **all quadruplets** $(i,j,k,l)$.
The move preserves the spherical constraint by construction (rotation on the hypersphere).

## Project Structure

```
├── include/          # Header files
│   ├── config.h      # Simulation parameters
│   ├── spins.h       # Spin configuration management
│   ├── disorder.h    # Coupling generation
│   ├── hamiltonian.h # Energy computation
│   └── mc.h          # Monte Carlo engine
├── src/              # Source and CUDA files
│   ├── main.cu       # Entry point
│   ├── spins.cu      # Spin initialization (uniform on hypersphere)
│   ├── disorder.cu   # Disorder generation on GPU
│   ├── hamiltonian.cu# Energy and ΔE kernels
│   └── mc.cu         # MC sweep logic
├── lib/              # External / utility libraries
├── bin/              # Compiled binaries (generated)
├── Makefile
└── README.md
```

## Requirements

- CUDA Toolkit 11.4+
- GPU: Tesla V100S (compute capability 7.0)
- GCC compatible with CUDA 11.4

## Build

```bash
make          # build
make clean    # clean build artifacts
```

## Run

```bash
./bin/pspin24 -N 64 -T 1.0 -J 1.0 -sweeps 10000 -seed 42
```
