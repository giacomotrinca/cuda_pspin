// bench_pt_scaling.cu — PT sweep-time scaling benchmark
//
// Measures average wall-clock time (seconds) per mc_sweep_pt call
// as a function of:
//   1) N     (number of spins),        fixed NT, NREP
//   2) NT    (temperature replicas),    fixed N, NREP
//   3) NREP  (real replicas per T),     fixed N, NT
//
// Both dense (spherical constraint, full H4) and sparse
// (smoothed-cube constraint, N random quartets) variants.
//
// Output: .dat files written to $TEST_OUTDIR/ (or current dir).
//   Dense:   sweep_time_vs_{N,NT,NREP}.dat
//   Sparse:  sweep_time_vs_{N,NT,NREP}_sparse.dat
//
// Usage: bench_pt_scaling [-warmup w] [-sweeps s] [-dev d]

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "config.h"
#include "mc.h"
#include "mc_sparse.h"
#include "disorder.h"
#include "box.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ---------- result struct ----------
struct TimingResult {
    int N, NT, NREP;
    int total_rep;
    int nsweeps;
    double sec_per_sweep;
};

// ---------- GPU memory estimate (bytes) ----------
static long long estimate_bytes(int N, int total_rep, bool sparse) {
    long long nq = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    long long bytes = 0;
    bytes += (long long)N * N * 16;                // g2
    if (sparse) {
        bytes += nq * (16 + 1);                    // temporary dense g4 during init
        bytes += (long long)N * (int)sizeof(SparseQuartet); // sparse table
    } else {
        bytes += nq * (16 + 1);                    // g4 + mask
    }
    bytes += (long long)total_rep * N * 16;        // spins
    bytes += (long long)total_rep * 64;            // rng states
    bytes += (long long)total_rep * 8 * 4;         // energies, accepted, proposed, betas
    return bytes;
}

static bool fits_in_gpu(int N, int total_rep, bool sparse) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return estimate_bytes(N, total_rep, sparse) < (long long)(free_mem * 0.85);
}

// ---------- Dense timing ----------
static TimingResult time_dense(int N, int NT, int NREP,
                               int warmup, int nsweeps, int dev) {
    int total = NT * NREP;
    SimConfig cfg = default_config();
    cfg.N    = N;
    cfg.nrep = total;
    cfg.T    = 1.0;
    cfg.J    = 1.0;
    cfg.seed = 12345;
    cfg.verbose = 0;
    cfg.dev  = dev;

    MCState state = mc_init(cfg);

    // temperature schedule  T ∈ [0.5, 2.0]
    std::vector<double> h_betas(total);
    for (int t = 0; t < NT; t++) {
        double T = 0.5 + (double)t / (NT > 1 ? NT - 1 : 1) * 1.5;
        for (int r = 0; r < NREP; r++)
            h_betas[t * NREP + r] = 1.0 / T;
    }
    mc_set_betas(state, h_betas.data());

    for (int i = 0; i < warmup; i++) mc_sweep_pt(state);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < nsweeps; i++) mc_sweep_pt(state);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    mc_free(state);

    TimingResult res;
    res.N = N;  res.NT = NT;  res.NREP = NREP;
    res.total_rep   = total;
    res.nsweeps     = nsweeps;
    res.sec_per_sweep = ms * 1e-3 / nsweeps;
    return res;
}

// ---------- Sparse timing ----------
static TimingResult time_sparse(int N, int NT, int NREP,
                                int warmup, int nsweeps, int dev) {
    int total = NT * NREP;
    SimConfig cfg = default_config();
    cfg.N    = N;
    cfg.nrep = total;
    cfg.T    = 1.0;
    cfg.J    = 1.0;
    cfg.seed = 12345;
    cfg.verbose = 0;
    cfg.dev  = dev;

    MCStateSparse state = mc_sparse_init(cfg);

    std::vector<double> h_betas(total);
    for (int t = 0; t < NT; t++) {
        double T = 0.5 + (double)t / (NT > 1 ? NT - 1 : 1) * 1.5;
        for (int r = 0; r < NREP; r++)
            h_betas[t * NREP + r] = 1.0 / T;
    }
    mc_sparse_set_betas(state, h_betas.data());

    for (int i = 0; i < warmup; i++) mc_sparse_sweep_pt(state);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < nsweeps; i++) mc_sparse_sweep_pt(state);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    mc_sparse_free(state);

    TimingResult res;
    res.N = N;  res.NT = NT;  res.NREP = NREP;
    res.total_rep   = total;
    res.nsweeps     = nsweeps;
    res.sec_per_sweep = ms * 1e-3 / nsweeps;
    return res;
}

// ---------- I/O ----------
static void print_hdr() {
    printf("%-6s %-6s %-6s %-6s %-14s\n",
           "N", "NT", "NREP", "total", "s/sweep");
    printf("----------------------------------------------\n");
}
static void print_row(const TimingResult& r) {
    printf("%-6d %-6d %-6d %-6d %-14.8e\n",
           r.N, r.NT, r.NREP, r.total_rep, r.sec_per_sweep);
}

static void write_dat(const char* path,
                      const std::vector<TimingResult>& results,
                      const char* scan_var) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "# %s  N  NT  NREP  total_rep  s_per_sweep\n", scan_var);
    for (auto& r : results) {
        int xval = 0;
        if (strcmp(scan_var, "N") == 0)    xval = r.N;
        else if (strcmp(scan_var, "NT") == 0) xval = r.NT;
        else                                  xval = r.NREP;
        fprintf(f, "%d\t%d\t%d\t%d\t%d\t%.8e\n",
                xval, r.N, r.NT, r.NREP, r.total_rep, r.sec_per_sweep);
    }
    fclose(f);
    printf("  Written %s\n", path);
}

// ---------- generic scan ----------
typedef TimingResult (*TimeFn)(int, int, int, int, int, int);

static std::vector<TimingResult>
run_scan(const char* label, TimeFn fn, bool sparse,
         const int* scan_vals, int n_scan,
         int fixed1, int fixed2,        // two fixed parameters
         int scan_id,                   // 0=N, 1=NT, 2=NREP
         int warmup, int nsweeps, int dev) {

    box_sec(label);
    print_hdr();
    std::vector<TimingResult> res;
    for (int i = 0; i < n_scan; i++) {
        int N, NT, NREP;
        if (scan_id == 0)      { N = scan_vals[i]; NT = fixed1; NREP = fixed2; }
        else if (scan_id == 1) { N = fixed1; NT = scan_vals[i]; NREP = fixed2; }
        else                   { N = fixed1; NT = fixed2; NREP = scan_vals[i]; }

        int total = NT * NREP;
        if (!fits_in_gpu(N, total, sparse)) {
            printf("%-6d %-6d %-6d %-6d  SKIP (memory)\n", N, NT, NREP, total);
            continue;
        }
        TimingResult r = fn(N, NT, NREP, warmup, nsweeps, dev);
        print_row(r);
        res.push_back(r);
    }
    return res;
}

// ================================================================
int main(int argc, char** argv) {
    int warmup  = 50;
    int nsweeps = 200;
    int dev     = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "-sweeps") == 0 && i + 1 < argc)
            nsweeps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-dev") == 0 && i + 1 < argc)
            dev = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [-warmup w] [-sweeps s] [-dev d]\n", argv[0]);
            return 1;
        }
    }

    if (dev >= 0) CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const char* outdir_env = getenv("TEST_OUTDIR");
    char outdir[512] = ".";
    if (outdir_env && outdir_env[0])
        snprintf(outdir, sizeof(outdir), "%s", outdir_env);

    box_top();
    box_title("  p-Spin 2+4 :: PT Sweep Time Scaling Bench       ");
    box_bot();
    printf("  GPU:    %s\n", prop.name);
    printf("  Warmup: %d sweeps\n", warmup);
    printf("  Timed:  %d sweeps\n\n", nsweeps);

    // scan arrays
    int Nvals[]    = {8, 10, 12, 14, 16, 18, 22, 26, 32, 40, 48, 64};
    int nN = sizeof(Nvals) / sizeof(Nvals[0]);

    int NTvals[]   = {1, 2, 4, 8, 16, 32, 64};
    int nNT = sizeof(NTvals) / sizeof(NTvals[0]);

    int NREPvals[] = {1, 2, 4, 8, 16, 32, 64};
    int nNR = sizeof(NREPvals) / sizeof(NREPvals[0]);

    char path[600];

    // ────────── DENSE ──────────
    {
        auto r = run_scan("DENSE: sweep time vs N  (NT=4, NREP=4)",
                          time_dense, false, Nvals, nN, 4, 4, 0,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_N.dat", outdir);
        write_dat(path, r, "N");
        printf("\n");
    }
    {
        auto r = run_scan("DENSE: sweep time vs NT (N=16, NREP=4)",
                          time_dense, false, NTvals, nNT, 16, 4, 1,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_NT.dat", outdir);
        write_dat(path, r, "NT");
        printf("\n");
    }
    {
        auto r = run_scan("DENSE: sweep time vs NREP (N=16, NT=4)",
                          time_dense, false, NREPvals, nNR, 16, 4, 2,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_NREP.dat", outdir);
        write_dat(path, r, "NREP");
        printf("\n");
    }

    // ────────── SPARSE ──────────
    {
        auto r = run_scan("SPARSE: sweep time vs N  (NT=4, NREP=4)",
                          time_sparse, false, Nvals, nN, 4, 4, 0,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_N_sparse.dat", outdir);
        write_dat(path, r, "N");
        printf("\n");
    }
    {
        auto r = run_scan("SPARSE: sweep time vs NT (N=16, NREP=4)",
                          time_sparse, false, NTvals, nNT, 16, 4, 1,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_NT_sparse.dat", outdir);
        write_dat(path, r, "NT");
        printf("\n");
    }
    {
        auto r = run_scan("SPARSE: sweep time vs NREP (N=16, NT=4)",
                          time_sparse, false, NREPvals, nNR, 16, 4, 2,
                          warmup, nsweeps, dev);
        snprintf(path, sizeof(path), "%s/sweep_time_vs_NREP_sparse.dat", outdir);
        write_dat(path, r, "NREP");
        printf("\n");
    }

    printf("PT scaling benchmark complete.\n");
    return 0;
}
