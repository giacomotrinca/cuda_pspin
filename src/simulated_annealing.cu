#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <ctime>
#include <sys/prctl.h>

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

// Usage: simulated_annealing -N n [-nrep n] -Tmax tmax -Tmin tmin -ntemp nt
//        -iter k [-J j] [-seed s] [-save_freq f] [-label l] [-dev d]
//        [-fmc 0|1|2] [-gamma g] [-verbose [0|1|2]]
//
// Linear schedule: T_k = Tmax - k*(Tmax-Tmin)/(ntemp-1),  k=0..ntemp-1
// Each temperature runs 2^iter MC sweeps.

int main(int argc, char** argv) {
    // --- Parse SA-specific args, then delegate the rest to parse_args ---
    double Tmax = -1.0, Tmin = -1.0;
    int ntemp = -1;

    // Pre-scan for SA args before parse_args sees them
    // We'll build a filtered argv for parse_args
    int new_argc = 0;
    char** new_argv = new char*[argc];
    new_argv[new_argc++] = argv[0];

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-Tmax") == 0 && i + 1 < argc) {
            Tmax = atof(argv[++i]);
        } else if (strcmp(argv[i], "-Tmin") == 0 && i + 1 < argc) {
            Tmin = atof(argv[++i]);
        } else if (strcmp(argv[i], "-ntemp") == 0 && i + 1 < argc) {
            ntemp = atoi(argv[++i]);
        } else {
            new_argv[new_argc++] = argv[i];
        }
    }

    if (Tmax < 0 || Tmin < 0 || ntemp < 1) {
        fprintf(stderr, "Simulated Annealing requires: -Tmax tmax -Tmin tmin -ntemp nt\n");
        fprintf(stderr, "Usage: %s -N n -Tmax tmax -Tmin tmin -ntemp nt -iter k "
                "[-nrep n] [-J j] [-seed s] [-save_freq f] [-label l] [-dev d] "
                "[-fmc 0|1|2] [-gamma g] [-verbose [0|1|2]]\n", argv[0]);
        delete[] new_argv;
        return 1;
    }
    if (Tmin > Tmax) {
        fprintf(stderr, "Error: Tmin (%.4f) > Tmax (%.4f)\n", Tmin, Tmax);
        delete[] new_argv;
        return 1;
    }

    // Parse remaining args (N, nrep, iter, seed, etc.)
    SimConfig cfg = parse_args(new_argc, new_argv);
    delete[] new_argv;

    int sweeps_per_temp = 1 << cfg.mc_iterations;  // 2^iter

    // Select GPU device
    if (cfg.dev >= 0) {
        CUDA_CHECK(cudaSetDevice(cfg.dev));
    }

    // Set process name for top/htop/btop
    {
        int lbl = (cfg.label >= 0) ? cfg.label : 0;
        char pname[16];
        snprintf(pname, sizeof(pname), "SA_N%d_NR%d_S%d", cfg.N, cfg.nrep, lbl);
        prctl(PR_SET_NAME, pname);
    }

    // Prepare output directory (always include sample label, default 0)
    int label = (cfg.label >= 0) ? cfg.label : 0;
    char datadir[256];
    snprintf(datadir, sizeof(datadir), "data/SA_N%d_NR%d_S%d", cfg.N, cfg.nrep, label);
    mkdir("data", 0755);
    mkdir(datadir, 0755);

    char confdir[256];
    snprintf(confdir, sizeof(confdir), "%s/configs", datadir);
    mkdir(confdir, 0755);

    // Open energy output file
    char datafile[256];
    snprintf(datafile, sizeof(datafile), "%s/energy_accept.txt", datadir);
    FILE* fout = fopen(datafile, "w");
    if (!fout) { fprintf(stderr, "Cannot open %s\n", datafile); return 2; }

    fprintf(fout, "# T\tsweep");
    for (int r = 0; r < cfg.nrep; r++) fprintf(fout, "\tE%d/N\tacc%d", r, r);
    fprintf(fout, "\n");

    // GPU memory check
    long long mem_g2    = (long long)cfg.N * cfg.N * sizeof(cuDoubleComplex);
    long long mem_g4    = n_g4_total(cfg.N) * sizeof(cuDoubleComplex);
    long long mem_spins = (long long)cfg.nrep * cfg.N * sizeof(cuDoubleComplex);
    long long mem_rng   = (long long)cfg.nrep * 64;
    long long mem_aux   = (long long)cfg.nrep * (sizeof(double) + 2 * sizeof(long long));
    long long mem_total = mem_g2 + mem_g4 + mem_spins + mem_rng + mem_aux;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (mem_total > (long long)free_mem * 0.9) {
        fprintf(stderr, "ERROR: Not enough GPU memory! Need %.1fMB, have %.1fMB free\n",
                mem_total / 1e6, free_mem / 1e6);
        return 1;
    }

    // Print header
    printf("\n");
    box_top();
    box_title("        p-Spin 2+4 :: Simulated Annealing         ");
    box_bot();
    if (cfg.verbose >= 2) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("  %-22s %s (compute %d.%d)\n", "GPU", prop.name, prop.major, prop.minor);
    }
    if (cfg.verbose >= 1) {
        printf("  %-22s %d\n", "N", cfg.N);
        printf("  %-22s %d\n", "nrep", cfg.nrep);
        printf("  %-22s %.6f\n", "Tmax", Tmax);
        printf("  %-22s %.6f\n", "Tmin", Tmin);
        printf("  %-22s %d\n", "ntemp", ntemp);
        printf("  %-22s 2^%d = %d\n", "sweeps/T", cfg.mc_iterations, sweeps_per_temp);
        printf("  %-22s %d\n", "save_freq", cfg.save_freq);
        printf("  %-22s %llu\n", "seed", (unsigned long long)cfg.seed);
        if (cfg.fmc_mode > 0) {
            const char* fmc_names[] = {"FC", "comb", "uniform"};
            printf("  %-22s %s (gamma=%.6f)\n", "FMC", fmc_names[cfg.fmc_mode], cfg.gamma);
        }
        printf("  %-22s %lld\n", "pairs", n_pairs(cfg.N));
        printf("  %-22s %lld\n", "quartets", n_quartets(cfg.N));
        printf("  %-22s %.1f MB (%.0f MB free / %.0f MB)\n", "memory",
               mem_total/1e6, free_mem/1e6, total_mem/1e6);
        printf("\n");
    }

    // Initialize MC state at Tmax
    cfg.T = Tmax;
    MCState state = mc_init(cfg);

    // FMC stats
    if (cfg.fmc_mode > 0) {
        const char* fmc_names[] = {"FC", "comb", "uniform"};
        printf("  %-22s %s (gamma=%.6f)  pairs=%lld/%lld  quartets=%lld/%lld\n",
               "FMC active", fmc_names[cfg.fmc_mode], cfg.gamma,
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

    // Host buffers
    double* h_energies = new double[cfg.nrep];
    long long* h_accepted = new long long[cfg.nrep];
    long long* h_proposed = new long long[cfg.nrep];
    long long spin_count = (long long)cfg.nrep * cfg.N;
    cuDoubleComplex* h_spins = new cuDoubleComplex[spin_count];
    double* h_re_im = new double[2 * cfg.N];

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Geometric cooling ratio: T_k = Tmax * A^k, A = (Tmin/Tmax)^(1/(ntemp-1))
    double A = (ntemp > 1) ? pow(Tmin / Tmax, 1.0 / (ntemp - 1)) : 1.0;
    printf("  %-22s %.8f\n", "schedule (A)", A);

    // Annealing loop
    box_sec("Running");
    for (int step = 0; step < ntemp; step++) {
        // Geometric schedule
        double T = Tmax * pow(A, step);
        cfg.T = T;

        // Reset acceptance counters for this temperature
        CUDA_CHECK(cudaMemset(state.d_accepted, 0, cfg.nrep * sizeof(long long)));
        CUDA_CHECK(cudaMemset(state.d_proposed, 0, cfg.nrep * sizeof(long long)));

        // Run sweeps, saving every save_freq
        if (cfg.verbose >= 2) {
            printf("  -- T = %.6f  [step %d/%d] --\n", T, step + 1, ntemp);
            printf("  %6s", "sweep");
            for (int r = 0; r < cfg.nrep; r++)
                printf("     E%d/N       acc%d   ", r, r);
            printf("\n");
        }

        for (int s = 0; s < sweeps_per_temp; s++) {
            mc_sweep(state, cfg);

            if ((s + 1) % cfg.save_freq == 0 || s == sweeps_per_temp - 1) {
                mc_get_results(state, h_energies, h_accepted, h_proposed);

                // Write time series to file
                fprintf(fout, "%.8f\t%d", T, s + 1);
                for (int r = 0; r < cfg.nrep; r++) {
                    double acc = (h_proposed[r] > 0)
                        ? (double)h_accepted[r] / h_proposed[r] : 0.0;
                    fprintf(fout, "\t%.8f\t%.5f", h_energies[r] / cfg.N, acc);
                }
                fprintf(fout, "\n");
                fflush(fout);

                // Verbose level 2: print sweep progress
                if (cfg.verbose >= 2) {
                    printf("  %6d", s + 1);
                    for (int r = 0; r < cfg.nrep; r++) {
                        double acc = (h_proposed[r] > 0)
                            ? (double)h_accepted[r] / h_proposed[r] : 0.0;
                        printf("  % .3e  % .3e", h_energies[r] / cfg.N, acc);
                    }
                    printf("\n");
                }

                // Reset acceptance counters for next window
                CUDA_CHECK(cudaMemset(state.d_accepted, 0, cfg.nrep * sizeof(long long)));
                CUDA_CHECK(cudaMemset(state.d_proposed, 0, cfg.nrep * sizeof(long long)));
            }
        }

        // Save spin configuration at this temperature only from second half
        if (step >= ntemp / 2) {
            mc_get_spins(state, h_spins);
            for (int r = 0; r < cfg.nrep; r++) {
                char conffile[512];
                snprintf(conffile, sizeof(conffile),
                         "%s/conf_r%d_T%.6f.bin", confdir, r, T);
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
        }

        // Verbose level 1: compact temperature progress
        if (cfg.verbose == 1) {
            printf("  [%d/%d] T=%.6f", step + 1, ntemp, T);
            for (int r = 0; r < cfg.nrep; r++)
                printf("  E/N=%.4f", h_energies[r] / cfg.N);
            printf("\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
    printf("\n"); box_sec("Summary");
    printf("  %-22s %.3f s\n", "total time", elapsed);
    printf("  %-22s %d x %d = %lld\n", "total sweeps",
           ntemp, sweeps_per_temp, (long long)ntemp * sweeps_per_temp);
    printf("\n");

    // Save timing
    char timefile[256];
    snprintf(timefile, sizeof(timefile), "%s/time.txt", datadir);
    FILE* ftim = fopen(timefile, "w");
    if (ftim) {
        fprintf(ftim, "# N nrep ntemp sweeps_per_temp Tmax Tmin total_time_s\n");
        fprintf(ftim, "%d %d %d %d %.6f %.6f %.6f\n",
                cfg.N, cfg.nrep, ntemp, sweeps_per_temp, Tmax, Tmin, elapsed);
        fclose(ftim);
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
