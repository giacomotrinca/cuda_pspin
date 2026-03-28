#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <ctime>

#include "config.h"
#include "mc.h"
#include "disorder.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    SimConfig cfg = parse_args(argc, argv);

    // Select GPU device
    if (cfg.dev >= 0) {
        CUDA_CHECK(cudaSetDevice(cfg.dev));
    }

    // Prepare output directory and file
    char datadir[128];
    if (cfg.label >= 0)
        snprintf(datadir, sizeof(datadir), "data/N%d_NR%d_S%d", cfg.N, cfg.nrep, cfg.label);
    else
        snprintf(datadir, sizeof(datadir), "data/N%d_NR%d", cfg.N, cfg.nrep);
    mkdir("data", 0755);
    mkdir(datadir, 0755);

    // Subdirectory for spin configurations
    char confdir[256];
    snprintf(confdir, sizeof(confdir), "%s/configs", datadir);
    mkdir(confdir, 0755);

    char datafile[256];
    snprintf(datafile, sizeof(datafile), "%s/energy_accept.txt", datadir);
    FILE* fout = fopen(datafile, "w");
    if (!fout) { fprintf(stderr, "Cannot open %s\n", datafile); return 2; }
    // Header
    fprintf(fout, "# iter");
    for (int r = 0; r < cfg.nrep; r++) fprintf(fout, "\tE%d/N\tacc%d", r, r);
    fprintf(fout, "\n");

    // Memory estimate (all GPU allocations)
    long long mem_g2    = (long long)cfg.N * cfg.N * sizeof(cuDoubleComplex);
    long long mem_g4    = n_quartets(cfg.N) * sizeof(cuDoubleComplex);
    long long mem_spins = (long long)cfg.nrep * cfg.N * sizeof(cuDoubleComplex);
    long long mem_rng   = (long long)cfg.nrep * 64;  // curandStatePhilox4_32_10_t = 64B
    long long mem_aux   = (long long)cfg.nrep * (sizeof(double) + 2 * sizeof(long long));
    long long mem_total = mem_g2 + mem_g4 + mem_spins + mem_rng + mem_aux;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (mem_total > (long long)free_mem * 0.9) {
        fprintf(stderr, "ERROR: Not enough GPU memory! Need %.1fMB, have %.1fMB free\n",
                mem_total / 1e6, free_mem / 1e6);
        return 1;
    }

    if (cfg.verbose) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("========================================\n");
        printf("p-Spin 2+4 Complex Spherical MC\n");
        printf("========================================\n");
        printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("N = %d, nrep = %d\n", cfg.N, cfg.nrep);
        printf("T = %.4f, J = %.4f\n", cfg.T, cfg.J);
        printf("MC iterations = %d, save_freq = %d\n", cfg.mc_iterations, cfg.save_freq);
        printf("Seed = %llu\n", (unsigned long long)cfg.seed);
        if (cfg.fmc_mode > 0) {
            const char* fmc_names[] = {"FC", "comb", "uniform"};
            printf("FMC: mode=%s, gamma=%.4f\n", fmc_names[cfg.fmc_mode], cfg.gamma);
        }
        printf("Pairs: %lld, Quartets: %lld\n", n_pairs(cfg.N), n_quartets(cfg.N));
        printf("Memory: g2=%.1fMB g4=%.1fMB spins=%.1fMB rng+aux=%.1fMB total=%.1fMB\n",
               mem_g2/1e6, mem_g4/1e6, mem_spins/1e6, (mem_rng+mem_aux)/1e6, mem_total/1e6);
        printf("GPU: %.0fMB free / %.0fMB total\n", free_mem/1e6, total_mem/1e6);
        printf("========================================\n");
    }

    // Initialize MC state (disorder + all replicas)
    MCState state = mc_init(cfg);

    // FMC stats and frequency file
    if (cfg.fmc_mode > 0) {
        const char* fmc_names[] = {"FC", "comb", "uniform"};
        printf("FMC: mode=%s gamma=%.4f  pairs=%lld/%lld  quartets=%lld/%lld\n",
               fmc_names[cfg.fmc_mode], cfg.gamma,
               state.n_pairs_active, n_pairs(cfg.N),
               state.n_quart_active, n_quartets(cfg.N));
        char freqfile[256];
        snprintf(freqfile, sizeof(freqfile), "%s/frequencies.txt", datadir);
        FILE* ff = fopen(freqfile, "w");
        if (ff) {
            fprintf(ff, "# i omega_i\n");
            for (int i = 0; i < cfg.N; i++)
                fprintf(ff, "%d\t%.12f\n", i, state.h_omega[i]);
            fclose(ff);
        }
    }

    // Allocate host buffers for results
    double* h_energies = new double[cfg.nrep];
    long long* h_accepted = new long long[cfg.nrep];
    long long* h_proposed = new long long[cfg.nrep];
    long long spin_count = (long long)cfg.nrep * cfg.N;
    cuDoubleComplex* h_spins = new cuDoubleComplex[spin_count];
    double* h_re_im = new double[2 * cfg.N]; // re/im interleaved for one replica

    // MC loop
    if (cfg.verbose) printf("Running %d MC iterations...\n", cfg.mc_iterations);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int s = 0; s < cfg.mc_iterations; s++) {
        mc_sweep(state, cfg);
        if ((s + 1) % cfg.save_freq == 0 || s == cfg.mc_iterations - 1) {
            mc_get_results(state, h_energies, h_accepted, h_proposed);
            // Write to file
            fprintf(fout, "%d", s + 1);
            for (int r = 0; r < cfg.nrep; r++) {
                double acc = (h_proposed[r] > 0)
                    ? (double)h_accepted[r] / h_proposed[r] : 0.0;
                fprintf(fout, "\t%.8f\t%.5f", h_energies[r] / cfg.N, acc);
            }
            fprintf(fout, "\n");
            fflush(fout);

            // Save spin configurations (binary: N doubles re, N doubles im per replica)
            mc_get_spins(state, h_spins);
            for (int r = 0; r < cfg.nrep; r++) {
                char conffile[512];
                snprintf(conffile, sizeof(conffile),
                         "%s/conf_r%d_iter%d.bin", confdir, r, s + 1);
                FILE* fc = fopen(conffile, "wb");
                if (fc) {
                    cuDoubleComplex* rep_spins = h_spins + (long long)r * cfg.N;
                    for (int i = 0; i < cfg.N; i++) {
                        h_re_im[2*i]     = cuCreal(rep_spins[i]);
                        h_re_im[2*i + 1] = cuCimag(rep_spins[i]);
                    }
                    fwrite(h_re_im, sizeof(double), 2 * cfg.N, fc);
                    fclose(fc);
                }
            }

            // Print to stdout
            if (cfg.verbose) {
                printf("  iter %d/%d:", s + 1, cfg.mc_iterations);
                for (int r = 0; r < cfg.nrep; r++) {
                    double acc = (h_proposed[r] > 0)
                        ? (double)h_accepted[r] / h_proposed[r] : 0.0;
                    printf("  [%d] E/N=%.4f acc=%.3f", r, h_energies[r] / cfg.N, acc);
                }
                printf("\n");
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
    double t_per_iter = elapsed / cfg.mc_iterations;
    printf("Time: %.3f s total, %.4f ms/iter (N=%d, nrep=%d)\n",
           elapsed, t_per_iter * 1e3, cfg.N, cfg.nrep);

    // Save timing to file
    char timefile[256];
    snprintf(timefile, sizeof(timefile), "%s/time_per_iter.txt", datadir);
    FILE* ftim = fopen(timefile, "w");
    if (ftim) {
        fprintf(ftim, "# N nrep mc_iterations total_time_s ms_per_iter\n");
        fprintf(ftim, "%d %d %d %.6f %.6f\n",
                cfg.N, cfg.nrep, cfg.mc_iterations, elapsed, t_per_iter * 1e3);
        fclose(ftim);
    }

    // Final results
    if (cfg.verbose) {
        mc_get_results(state, h_energies, h_accepted, h_proposed);
        printf("\n========================================\n");
        printf("RESULTS\n");
        printf("========================================\n");
        for (int r = 0; r < cfg.nrep; r++) {
            double acc = (h_proposed[r] > 0)
                ? (double)h_accepted[r] / h_proposed[r] : 0.0;
            printf("Replica %d: E/N = %.6f, acc = %.4f\n",
                   r, h_energies[r] / cfg.N, acc);
        }
        printf("========================================\n");
    }

    // Cleanup
    fclose(fout);
    delete[] h_energies;
    delete[] h_accepted;
    delete[] h_proposed;
    delete[] h_spins;
    delete[] h_re_im;
    mc_free(state);
    return 0;
}
