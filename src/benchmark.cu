// benchmark.cu — Performance benchmark for p-Spin 2+4 Monte Carlo
//
// Measures sweeps/second scaling with system size N and number of replicas nrep.
// Outputs TSV data to bench_data/ for plotting with plot_benchmark.cpp.
//
// Usage: benchmark [-warmup w] [-sweeps s] [-dev d]

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "config.h"
#include "mc.h"
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

// ---------- GPU memory estimate (bytes) ----------
static long long estimate_gpu_bytes(int N, int nrep) {
    long long np = (long long)N * (N - 1) / 2;
    long long nq = (long long)N * (N - 1) * (N - 2) * (N - 3) / 24;
    long long bytes = 0;
    bytes += (long long)N * N * 16;       // d_g2
    bytes += nq * 16;                     // d_g4  (cuDoubleComplex)
    bytes += nq * 1;                      // d_g4_mask (uint8_t)
    bytes += (long long)nrep * N * 16;    // d_spins
    bytes += (long long)nrep * 64;        // d_rng (Philox state)
    bytes += (long long)nrep * 8;         // d_energies
    bytes += (long long)nrep * 8 * 2;     // d_accepted + d_proposed
    bytes += (long long)nrep * 8;         // d_betas
    (void)np;
    return bytes;
}

struct BenchResult {
    int N;
    int nrep;
    int nsweeps;
    double time_ms;         // total GPU time for nsweeps
    double ms_per_sweep;
    double sweeps_per_sec;
    double spin_updates_per_ns;  // (nsweeps * N * n_pairs) / time_ns
    double mem_MB;
    double acceptance_rate;      // mean acceptance rate across replicas
    double total_throughput;     // nrep * sweeps_per_sec
};

// Run a single benchmark point
static BenchResult run_bench(int N, int nrep, int warmup, int nsweeps, int dev) {
    SimConfig cfg = default_config();
    cfg.N = N;
    cfg.nrep = nrep;
    cfg.T = 1.0;
    cfg.J = 1.0;
    cfg.seed = 12345;
    cfg.verbose = 0;
    cfg.dev = dev;

    MCState state = mc_init(cfg);

    // Set uniform betas
    std::vector<double> h_betas(nrep, 1.0 / cfg.T);
    mc_set_betas(state, h_betas.data());

    // Warmup
    for (int i = 0; i < warmup; i++)
        mc_sweep_pt(state);

    // Timed region with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nsweeps; i++)
        mc_sweep_pt(state);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Read acceptance counters
    std::vector<double> h_energies(nrep);
    std::vector<long long> h_accepted(nrep), h_proposed(nrep);
    mc_get_results(state, h_energies.data(), h_accepted.data(), h_proposed.data());

    double mean_acc = 0;
    for (int r = 0; r < nrep; r++) {
        if (h_proposed[r] > 0)
            mean_acc += (double)h_accepted[r] / h_proposed[r];
    }
    mean_acc /= nrep;

    mc_free(state);

    long long np = (long long)N * (N - 1) / 2;
    double time_ns = elapsed_ms * 1e6;

    BenchResult res;
    res.N = N;
    res.nrep = nrep;
    res.nsweeps = nsweeps;
    res.time_ms = elapsed_ms;
    res.ms_per_sweep = elapsed_ms / nsweeps;
    res.sweeps_per_sec = nsweeps / (elapsed_ms * 1e-3);
    res.spin_updates_per_ns = (double)nsweeps * nrep * np / time_ns;
    res.mem_MB = estimate_gpu_bytes(N, nrep) / (1024.0 * 1024.0);
    res.acceptance_rate = mean_acc;
    res.total_throughput = nrep * res.sweeps_per_sec;
    return res;
}

static void print_header() {
    printf("%-6s %-6s %-8s %-12s %-14s %-16s %-10s %-8s\n",
           "N", "nrep", "sweeps", "time(ms)", "ms/sweep", "sweeps/s", "mem(MB)", "acc");
    printf("--------------------------------------------------------------------------\n");
}

static void print_result(const BenchResult& r) {
    printf("%-6d %-6d %-8d %-12.2f %-14.4f %-16.1f %-10.2f %-8.4f\n",
           r.N, r.nrep, r.nsweeps, r.time_ms, r.ms_per_sweep,
           r.sweeps_per_sec, r.mem_MB, r.acceptance_rate);
}

static void write_tsv(const char* path, const std::vector<BenchResult>& results) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "N\tnrep\tnsweeps\ttime_ms\tms_per_sweep\tsweeps_per_sec\tspin_updates_per_ns\tmem_MB\tacceptance_rate\ttotal_throughput\n");
    for (auto& r : results) {
        fprintf(f, "%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.6f\t%.2f\t%.6f\t%.2f\n",
                r.N, r.nrep, r.nsweeps, r.time_ms, r.ms_per_sweep,
                r.sweeps_per_sec, r.spin_updates_per_ns, r.mem_MB,
                r.acceptance_rate, r.total_throughput);
    }
    fclose(f);
    printf("  Written %s\n", path);
}

int main(int argc, char** argv) {
    int warmup  = 100;
    int nsweeps = 500;
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

    // Print GPU info
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Write GPU name to file for plot_benchmark / run_bench.sh
    {
        FILE* fgpu = fopen("bench_data/gpu_name.txt", "w");
        if (fgpu) { fprintf(fgpu, "%s\n", prop.name); fclose(fgpu); }
    }

    box_top();
    box_title("    p-Spin 2+4 — Monte Carlo Benchmark        ");
    box_bot();
    printf("  GPU:      %s\n", prop.name);
    printf("  SMs:      %d\n", prop.multiProcessorCount);
    printf("  Mem:      %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Warmup:   %d sweeps\n", warmup);
    printf("  Timed:    %d sweeps\n", nsweeps);
    printf("\n");

    // ========== Scan 1: Scaling with N (fixed nrep=4) ==========
    {
        box_sec("Scaling with N (nrep=4)");
        print_header();

        int N_vals[] = {8, 10, 12, 14, 16, 18, 22, 26, 32, 40, 48, 64};
        int nN = sizeof(N_vals) / sizeof(N_vals[0]);
        int fixed_nrep = 4;

        std::vector<BenchResult> results;
        for (int i = 0; i < nN; i++) {
            // Check memory before running
            double est_MB = estimate_gpu_bytes(N_vals[i], fixed_nrep) / (1024.0 * 1024.0);
            double avail_MB = prop.totalGlobalMem / (1024.0 * 1024.0);
            if (est_MB > avail_MB * 0.9) {
                printf("%-6d %-6d (skipped: estimated %.0f MB > 90%% of %.0f MB)\n",
                       N_vals[i], fixed_nrep, est_MB, avail_MB);
                continue;
            }
            BenchResult r = run_bench(N_vals[i], fixed_nrep, warmup, nsweeps, dev);
            print_result(r);
            results.push_back(r);
        }
        write_tsv("bench_data/scaling_N.tsv", results);
        printf("\n");
    }

    // ========== Scan 2: Scaling with nrep (fixed N=18) ==========
    {
        box_sec("Scaling with nrep (N=18)");
        print_header();

        int nrep_vals[] = {1, 2, 4, 8, 16, 32, 64};
        int nR = sizeof(nrep_vals) / sizeof(nrep_vals[0]);
        int fixed_N = 18;

        std::vector<BenchResult> results;
        for (int i = 0; i < nR; i++) {
            double est_MB = estimate_gpu_bytes(fixed_N, nrep_vals[i]) / (1024.0 * 1024.0);
            double avail_MB = prop.totalGlobalMem / (1024.0 * 1024.0);
            if (est_MB > avail_MB * 0.9) {
                printf("%-6d %-6d (skipped: estimated %.0f MB > 90%% of %.0f MB)\n",
                       fixed_N, nrep_vals[i], est_MB, avail_MB);
                continue;
            }
            BenchResult r = run_bench(fixed_N, nrep_vals[i], warmup, nsweeps, dev);
            print_result(r);
            results.push_back(r);
        }
        write_tsv("bench_data/scaling_nrep.tsv", results);
        printf("\n");
    }

    // ========== Scan 3: Scaling with nrep (fixed N=32) ==========
    {
        box_sec("Scaling with nrep (N=32)");
        print_header();

        int nrep_vals[] = {1, 2, 4, 8, 16, 32, 64};
        int nR = sizeof(nrep_vals) / sizeof(nrep_vals[0]);
        int fixed_N = 32;

        std::vector<BenchResult> results;
        for (int i = 0; i < nR; i++) {
            double est_MB = estimate_gpu_bytes(fixed_N, nrep_vals[i]) / (1024.0 * 1024.0);
            double avail_MB = prop.totalGlobalMem / (1024.0 * 1024.0);
            if (est_MB > avail_MB * 0.9) {
                printf("%-6d %-6d (skipped: estimated %.0f MB > 90%% of %.0f MB)\n",
                       fixed_N, nrep_vals[i], est_MB, avail_MB);
                continue;
            }
            BenchResult r = run_bench(fixed_N, nrep_vals[i], warmup, nsweeps, dev);
            print_result(r);
            results.push_back(r);
        }
        write_tsv("bench_data/scaling_nrep_N32.tsv", results);
        printf("\n");
    }

    printf("Benchmark complete. Run bin/plot_benchmark to generate plots.\n");
    return 0;
}
