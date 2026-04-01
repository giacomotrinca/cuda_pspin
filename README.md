# p-Spin 2+4 — CUDA Monte Carlo

GPU-accelerated Monte Carlo simulation of the **complex p-spin glass** with mixed 2-body and 4-body interactions. Includes plain Monte Carlo, simulated annealing, and parallel tempering, in two variants: the standard **spherical** model and a **sparse** variant with smoothed-cube constraint.

---

## Model

### Hamiltonian

The Hamiltonian combines a 2-body (quadratic) and a 4-body (quartic) interaction. For sorted indices $i<j<k<l$, the three interaction channels correspond to different conjugation patterns:

$$
H = H_2 + H_4
$$

$$
H_2 = -\sum_{i<j} \mathrm{Re}\!\bigl[\, g^{(2)}_{ij}\; a_i\, a_j^* \,\bigr]
$$

$$
H_4 = -\sum_{i<j<k<l} \sum_{\mathrm{ch}=0}^{2}\; \mathrm{Re}\!\bigl[\, g^{(4)}_{ijkl,\mathrm{ch}}\; a_i\, f_\mathrm{ch}(a_j)\, f_\mathrm{ch}(a_k)\, f_\mathrm{ch}(a_l) \,\bigr]
$$

where $a_i \in \mathbb{C}$ are complex amplitudes.

### Constraints

Two geometric constraints are available:

| Variant | Constraint | MC move preserves |
|---------|-----------|-------------------|
| **Spherical** (default) | $\displaystyle\sum_i \|a_i\|^2 = N$ | $\|a_i\|^2 + \|a_j\|^2$ via pair rotation on hypersphere |
| **Sparse** (cube) | $\displaystyle\sum_i \|a_i\|^4 = N$ | $\|a_i\|^4 + \|a_j\|^4$ via cube-constrained proposal |

### Disorder

The couplings are i.i.d. complex Gaussian with zero mean. After FMC filtering, they are rescaled to maintain extensivity:

$$
\sigma^{(p)} = J\,\sqrt{\frac{N}{n_\mathrm{surviving}^{(p)}}}
$$

### Frequency Matching Condition (FMC)

The FMC filter assigns a frequency $\omega_i$ to each mode and zeros out couplings that violate quasi-resonance:

| Mode | H2 condition | H4 condition (ch-dependent) |
|------|-------------|--------------------------|
| **FC** (`fmc=0`) | No filter | No filter |
| **Comb** (`fmc=1`) | $\omega_k = k$ → H2 fully killed | $\|\omega_i + \omega_j - \omega_k - \omega_l\| \le \gamma$ |
| **Uniform** (`fmc=2`) | $\|\omega_i - \omega_j\| \le \gamma$ | Same as comb |

### Sparse H4

In the sparse variant, after FMC filtering, only **N quartets** are sampled uniformly at random from the surviving set (partial Fisher-Yates shuffle). Couplings are rescaled accordingly. This reduces the H4 cost from $O(N^3)$ to $O(N)$ per MC step.

---

## Programs

| Target | Binary | Description |
|--------|--------|-------------|
| `make mc` | `bin/pspin24` | Plain Monte Carlo at fixed $T$ |
| `make sa` | `bin/simulated_annealing` | Simulated annealing (temperature ramp) |
| `make pt` | `bin/parallel_tempering` | Parallel tempering (spherical constraint) |
| `make pts` | `bin/parallel_tempering_sparse` | Parallel tempering (cube constraint, sparse H4) |
| `make analysis_mc` | `bin/analysis` | Analysis of MC output |
| `make analysis_sa` | `bin/analysis_sa` | Analysis of SA output |
| `make analysis_pt` | `bin/analysis_pt` | Analysis of PT output |
| `make bench` | `bin/benchmark` | Performance benchmark |

---

## Project structure

```
.
├── include/
│   ├── config.h              # SimConfig struct and CLI parser
│   ├── spins.h               # Spherical spin init + pair rotation proposal
│   ├── spins_sparse.h        # Cube spin init + cube-constrained proposal
│   ├── disorder.h            # Coupling generation, FMC filter, rescaling
│   ├── hamiltonian.h         # Energy computation (dense)
│   ├── mc.h                  # MCState (dense H2 + H4), sweep kernel
│   ├── mc_sparse.h           # MCStateSparse (dense H2, sparse H4)
│   └── box.h                 # Terminal box-drawing utilities
├── src/
│   ├── config.cu             # CLI argument parsing
│   ├── spins.cu              # Spherical init (uniform on S^{2N-1})
│   ├── spins_sparse.cu       # Cube init (project onto Σ|a|⁴=N)
│   ├── disorder.cu           # Generate g2/g4, FMC kernels, rescaling
│   ├── hamiltonian.cu        # Full energy / ΔE computation
│   ├── mc.cu                 # Dense MC sweep (3-type H4 enumeration, O(N³))
│   ├── mc_sparse.cu          # Sparse MC sweep (iterate N quartets, O(N))
│   ├── montecarlo.cu         # Main: plain MC
│   ├── simulated_annealing.cu# Main: simulated annealing
│   ├── parallel_tempering.cu # Main: parallel tempering (spherical)
│   ├── parallel_tempering_sparse.cu  # Main: parallel tempering (sparse)
│   ├── benchmark.cu          # Performance scaling tests
│   ├── analysis.cpp          # Analysis: MC
│   ├── analysis_sa.cpp       # Analysis: SA
│   ├── analysis_pt.cpp       # Analysis: PT
│   └── plot_benchmark.cpp    # Benchmark plot generator
├── parallel_tempering        # Wrapper script (spherical)
├── parallel_tempering_sparse # Wrapper script (sparse)
├── scan_samples_pt           # Multi-sample launcher for PT
├── scan_samples_sa           # Multi-sample launcher for SA
├── run_bench.sh              # Benchmark driver
├── Makefile
├── bin/                      # Compiled binaries (generated)
├── obj/                      # Object files (generated)
├── bench_data/               # Benchmark results and plots
└── data/                     # Simulation output (generated at runtime)
```

---

## Requirements

- **CUDA Toolkit** ≥ 11.4
- **GPU**: Tesla V100S (`sm_70`, default) or GTX 680 (`sm_30`, via `visnu=1`)
- **GCC** compatible with CUDA
- **gnuplot** (for benchmark plots, optional)

---

## Build

```bash
make pt          # spherical parallel tempering
make pts         # sparse parallel tempering (cube constraint)
make all         # all standard targets
make clean       # remove bin/ and obj/
```

Cross-compile for GTX 680:

```bash
make pt visnu=1
```

---

## Usage

### Wrapper scripts

The wrapper scripts translate `--key=value` style arguments into the binary's `-key value` format, display a summary banner, and invoke the correct binary.

```bash
# Single run — spherical PT
./parallel_tempering --size=32 --tmax=1.6 --tmin=0.2 --nt=20 \
    --iter=18 --nrep=4 --fmc=2 --seed=42

# Single run — sparse PT (cube constraint, N quartets)
./parallel_tempering_sparse --size=32 --tmax=1.6 --tmin=0.2 --nt=20 \
    --iter=18 --nrep=4 --fmc=2 --seed=42
```

### Multi-sample scans

Launch multiple independent disorder realisations:

```bash
# Spherical PT — 10 samples
./scan_samples_pt --size=32 --nsamples=10 --iter=20 --nrep=16

# Sparse PT — 10 samples
./scan_samples_pt --sparse --size=32 --nsamples=10 --iter=20 --nrep=16
```

Use `--help` on any script for the full option list.

### Direct binary invocation

```bash
./bin/parallel_tempering -N 32 -Tmax 1.6 -Tmin 0.2 -NT 20 \
    -iter 262144 -seed 42 -nrep 4 -pt_freq 64 -save_freq 64 \
    -fmc 2 -verbose 2
```

### Benchmarks

```bash
./run_bench.sh              # run GPU benchmark
./run_bench.sh --plot       # plot only (after copying bench_data/)
```

---

## Output

Each simulation creates a directory under `data/`:

```
data/PT_N32_NT20_NR4_S0/       # spherical PT, sample 0
data/PTS_N32_NT20_NR4_S0/      # sparse PT, sample 0
```

Contents:

| File | Description |
|------|-------------|
| `energy_accept.txt` | Per-sweep energy and acceptance ratio per temperature |
| `exchanges.txt` | Replica exchange statistics |
| `configs.bin` | Binary dump of spin configurations |
| `temperatures.txt` | Temperature schedule ($T_k$, $\beta_k$) |
| `frequencies.txt` | FMC frequencies $\omega_i$ (if FMC active) |
| `quartets.txt` | Sparse H4 quartet list (sparse variant only) |

---

## GPU kernel architecture

The MC sweep kernel runs **one CUDA block per replica** with N/2 pair proposals per sweep:

1. **Thread 0** picks a random pair $(i, j)$ and proposes a constraint-preserving move
2. **All threads** compute $\Delta E$ in parallel:
   - H2: thread-parallel sum over dense $g_2$ matrix (zeros skipped implicitly after FMC)
   - H4 spherical: 3-type enumeration of all quartets involving $i$ or $j$ — $O(N^3)$
   - H4 sparse: loop over $N$ stored quartets, check if $i$ or $j$ is involved — $O(N)$
3. **Warp-shuffle reduction** aggregates $\Delta E$
4. **Thread 0** applies Metropolis accept/reject
