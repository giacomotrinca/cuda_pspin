#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <ctime>
#include <cstdlib>
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

// Usage: parallel_tempering -N n -Tmax t -Tmin t -NT nt -pt_freq p -iter k
//        [-nrep n] [-J j] [-seed s] [-save_freq f] [-label l] [-dev d]
//        [-fmc 0|1|2] [-gamma g] [-verbose [0|1|2]]
//
// NT temperatures (linear schedule), each replicated nrep times.
// Every pt_freq sweeps, propose NT-1 adjacent replica exchanges.
// Total GPU replicas = NT * nrep.

int main(int argc, char** argv) {
    // --- Parse PT-specific args, filter the rest for parse_args ---
    double Tmax = -1.0, Tmin = -1.0;
    int NT = -1, pt_freq = -1;
    int log_temp = 0;

    int new_argc = 0;
    char** new_argv = new char*[argc];
    new_argv[new_argc++] = argv[0];

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-Tmax") == 0 && i + 1 < argc)
            Tmax = atof(argv[++i]);
        else if (strcmp(argv[i], "-Tmin") == 0 && i + 1 < argc)
            Tmin = atof(argv[++i]);
        else if (strcmp(argv[i], "-NT") == 0 && i + 1 < argc)
            NT = atoi(argv[++i]);
        else if (strcmp(argv[i], "-pt_freq") == 0 && i + 1 < argc)
            pt_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "-log_temp") == 0)
            log_temp = 1;
        else
            new_argv[new_argc++] = argv[i];
    }

    if (Tmax < 0 || Tmin < 0 || NT < 2 || pt_freq < 1) {
        fprintf(stderr, "Parallel Tempering requires: -Tmax t -Tmin t -NT nt -pt_freq p\n");
        fprintf(stderr, "Usage: %s -N n -Tmax t -Tmin t -NT nt -pt_freq p -iter k "
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

    SimConfig cfg = parse_args(new_argc, new_argv);
    delete[] new_argv;

    int nrep_per_temp = cfg.nrep;   // independent copies per temperature
    int total_replicas = NT * nrep_per_temp;

    // Override nrep for MCState: manages all replicas
    cfg.nrep = total_replicas;
    cfg.T = Tmax;  // initial T (overridden by per-replica betas)

    if (cfg.dev >= 0) CUDA_CHECK(cudaSetDevice(cfg.dev));

    // Set process name for top/htop/btop
    {
        int lbl = (cfg.label >= 0) ? cfg.label : 0;
        char pname[16];
        snprintf(pname, sizeof(pname), "PT_N%d_NR%d_S%d", cfg.N, nrep_per_temp, lbl);
        prctl(PR_SET_NAME, pname);
    }

    // --- Temperature schedule ---
    double* T_sched    = new double[NT];
    double* beta_sched = new double[NT];
    if (log_temp) {
        // Geometric: T_0 = Tmax, T[NT-1] = Tmin, T_k = A * T_{k-1}
        double A = (NT > 1) ? pow(Tmin / Tmax, 1.0 / (NT - 1)) : 1.0;
        for (int t = 0; t < NT; t++) {
            T_sched[t]    = Tmax * pow(A, t);
            beta_sched[t] = 1.0 / T_sched[t];
        }
    } else {
        // Linear: T_k = Tmax - k * dT
        double dT = (NT > 1) ? (Tmax - Tmin) / (NT - 1) : 0.0;
        for (int t = 0; t < NT; t++) {
            T_sched[t]    = Tmax - t * dT;
            beta_sched[t] = 1.0 / T_sched[t];
        }
    }

    // --- Output directories ---
    int label = (cfg.label >= 0) ? cfg.label : 0;
    char datadir[256];
    snprintf(datadir, sizeof(datadir), "data/PT_N%d_NT%d_NR%d_S%d",
             cfg.N, NT, nrep_per_temp, label);
    mkdir("data", 0755);
    mkdir(datadir, 0755);

    // --- Open output files ---
    char efile[256], xfile[256];
    snprintf(efile, sizeof(efile), "%s/energy_accept.txt", datadir);
    snprintf(xfile, sizeof(xfile), "%s/exchanges.txt", datadir);

    FILE* fout = fopen(efile, "w");
    FILE* fex  = fopen(xfile, "w");
    if (!fout || !fex) { fprintf(stderr, "Cannot open output files\n"); return 2; }

    // Configs: single binary file (header + appended snapshots)
    char confbin[256];
    snprintf(confbin, sizeof(confbin), "%s/configs.bin", datadir);
    FILE* fconf = fopen(confbin, "wb");
    if (!fconf) { fprintf(stderr, "Cannot open configs.bin\n"); return 2; }
    // Header: N, NT, nrep_per_temp
    {
        int hdr[3] = { cfg.N, NT, nrep_per_temp };
        fwrite(hdr, sizeof(int), 3, fconf);
    }

    // Headers
    fprintf(fout, "# sweep\tTidx\tT");
    for (int r = 0; r < nrep_per_temp; r++)
        fprintf(fout, "\tE%d/N\tacc%d", r, r);
    fprintf(fout, "\n");

    fprintf(fex, "# sweep\tTidx\tT_high\tT_low\tn_acc\tn_prop\trate\n");

    // --- GPU memory check ---
    long long mem_g2    = (long long)cfg.N * cfg.N * sizeof(cuDoubleComplex);
    long long mem_g4    = n_quartets(cfg.N) * (sizeof(cuDoubleComplex) + sizeof(uint8_t));
    long long mem_spins = (long long)total_replicas * cfg.N * sizeof(cuDoubleComplex);
    long long mem_rng   = (long long)total_replicas * 64;
    long long mem_aux   = (long long)total_replicas * (sizeof(double) * 2 + 2 * sizeof(long long));
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
    box_title("         p-Spin 2+4 :: Parallel Tempering         ");
    box_bot();
    if (cfg.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.dev >= 0 ? cfg.dev : 0));
        printf("  %-22s %s (compute %d.%d)\n", "GPU", prop.name, prop.major, prop.minor);
        printf("  %-22s %d\n", "N", cfg.N);
        printf("  %-22s %d\n", "NT", NT);
        printf("  %-22s %d\n", "nrep/T", nrep_per_temp);
        printf("  %-22s %d\n", "total replicas", total_replicas);
        printf("  %-22s %.6f\n", "Tmax", Tmax);
        printf("  %-22s %.6f\n", "Tmin", Tmin);
        if (log_temp) {
            double A = (NT > 1) ? pow(Tmin / Tmax, 1.0 / (NT - 1)) : 1.0;
            printf("  %-22s geometric (A=%.8f)\n", "T schedule", A);
        } else {
            double dT = (NT > 1) ? (Tmax - Tmin) / (NT - 1) : 0.0;
            printf("  %-22s linear (dT=%.8f)\n", "T schedule", dT);
        }
        printf("  %-22s %d\n", "MC sweeps", cfg.mc_iterations);
        printf("  %-22s %d\n", "pt_freq", pt_freq);
        printf("  %-22s %d\n", "save_freq", cfg.save_freq);
        printf("  %-22s %llu\n", "seed", (unsigned long long)cfg.seed);
        printf("  %-22s %.4f\n", "J", cfg.J);
        printf("  %-22s %.4f\n", "J0", cfg.J0);
        printf("  %-22s %.4f\n", "alpha", cfg.alpha);
        printf("  %-22s %.4f\n", "alpha0", cfg.alpha0);
        printf("  %-22s %.4f  (J2=(1-a)*J)\n", "J2", (1.0 - cfg.alpha) * cfg.J);
        printf("  %-22s %.4f  (J4=a*J)\n", "J4", cfg.alpha * cfg.J);
        if (cfg.fmc_mode > 0) {
            const char* fmc_names[] = {"FC", "comb", "uniform"};
            printf("  %-22s %s (gamma=%.6f)\n", "FMC", fmc_names[cfg.fmc_mode], cfg.gamma);
        }
        printf("  %-22s %lld\n", "pairs", n_pairs(cfg.N));
        printf("  %-22s %lld\n", "quartets", n_quartets(cfg.N));
        printf("  %-22s %.1f MB (%.0f MB free / %.0f MB)\n", "memory",
               mem_total/1e6, free_mem/1e6, total_mem/1e6);
        printf("\n");
        printf("  %-22s %d temperatures in [%.6f, %.6f]\n\n",
               "schedule", NT, T_sched[NT - 1], T_sched[0]);
    }

    // --- Initialize MCState (all replicas, shared disorder) ---
    MCState state = mc_init(cfg);

    // FMC stats and frequency file
    if (cfg.fmc_mode > 0) {
        const char* fmc_names[] = {"FC", "comb", "uniform"};
        printf("  %-22s %s (gamma=%.6f)  pairs=%lld/%lld  quartets=%lld/%lld\n",
               "FMC active", fmc_names[cfg.fmc_mode], cfg.gamma,
               state.n_pairs_active, n_pairs(cfg.N),
               state.n_quart_active, n_g4_total(cfg.N));
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

    // Save temperature schedule
    {
        char tfile[256];
        snprintf(tfile, sizeof(tfile), "%s/temperatures.txt", datadir);
        FILE* ft = fopen(tfile, "w");
        if (ft) {
            fprintf(ft, "# Tidx T beta\n");
            for (int t = 0; t < NT; t++)
                fprintf(ft, "%d\t%.12f\t%.12f\n", t, T_sched[t], beta_sched[t]);
            fclose(ft);
        }
    }

    // --- Set initial per-replica betas ---
    double* h_betas = new double[total_replicas];
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < nrep_per_temp; r++)
            h_betas[t * nrep_per_temp + r] = beta_sched[t];
    mc_set_betas(state, h_betas);

    // --- Permutation: perm[logical_idx] = physical replica index ---
    // logical_idx = t * nrep_per_temp + r
    // Initially: identity mapping
    int* perm = new int[total_replicas];
    for (int i = 0; i < total_replicas; i++) perm[i] = i;

    // --- Host buffers ---
    double* h_energies     = new double[total_replicas];
    long long* h_accepted  = new long long[total_replicas];
    long long* h_proposed  = new long long[total_replicas];
    long long spin_count   = (long long)total_replicas * cfg.N;
    cuDoubleComplex* h_spins = new cuDoubleComplex[spin_count];
    double* h_re_im = new double[2 * cfg.N];

    // Exchange counters (cumulative and per-save window)
    long long* ex_acc_total  = new long long[NT - 1]();
    long long* ex_prop_total = new long long[NT - 1]();
    long long* ex_acc_win    = new long long[NT - 1]();
    long long* ex_prop_win   = new long long[NT - 1]();

    // Reproducible RNG for exchange acceptance
    srand48((long)cfg.seed + 5000);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    if (cfg.verbose >= 1) box_sec("Running");

    // Main MC loop
    for (int s = 0; s < cfg.mc_iterations; s++) {
        mc_sweep_pt(state);

        // --- Replica exchange ---
        if ((s + 1) % pt_freq == 0) {
            mc_get_results(state, h_energies, h_accepted, h_proposed);

            // Sweep from highest T (t=0) to lowest-1 (t=NT-2)
            for (int t = 0; t < NT - 1; t++) {
                for (int r = 0; r < nrep_per_temp; r++) {
                    int phys_a = perm[t       * nrep_per_temp + r]; // at T_t (higher T)
                    int phys_b = perm[(t + 1) * nrep_per_temp + r]; // at T_{t+1} (lower T)

                    // Delta = (beta_t - beta_{t+1})(E_a - E_b)
                    // beta_t < beta_{t+1}, so beta_t - beta_{t+1} < 0
                    double delta = (beta_sched[t] - beta_sched[t + 1])
                                 * (h_energies[phys_a] - h_energies[phys_b]);

                    ex_prop_total[t]++;
                    ex_prop_win[t]++;

                    bool accept = (delta >= 0.0);
                    if (!accept) accept = (drand48() < exp(delta));

                    if (accept) {
                        ex_acc_total[t]++;
                        ex_acc_win[t]++;
                        // Swap permutation entries
                        int tmp = perm[t * nrep_per_temp + r];
                        perm[t * nrep_per_temp + r] = perm[(t + 1) * nrep_per_temp + r];
                        perm[(t + 1) * nrep_per_temp + r] = tmp;
                    }
                }
            }

            // Rebuild betas from current permutation
            for (int t = 0; t < NT; t++)
                for (int r = 0; r < nrep_per_temp; r++)
                    h_betas[perm[t * nrep_per_temp + r]] = beta_sched[t];
            mc_set_betas(state, h_betas);
        }

        // --- Save data ---
        if ((s + 1) % cfg.save_freq == 0 || s == cfg.mc_iterations - 1) {
            mc_get_results(state, h_energies, h_accepted, h_proposed);

            // Energy & acceptance per temperature
            for (int t = 0; t < NT; t++) {
                fprintf(fout, "%d\t%d\t%.8f", s + 1, t, T_sched[t]);
                for (int r = 0; r < nrep_per_temp; r++) {
                    int phys = perm[t * nrep_per_temp + r];
                    double acc = (h_proposed[phys] > 0)
                        ? (double)h_accepted[phys] / h_proposed[phys] : 0.0;
                    fprintf(fout, "\t%.8f\t%.5f", h_energies[phys] / cfg.N, acc);
                }
                fprintf(fout, "\n");
            }
            fflush(fout);

            // Exchange rates (window since last save)
            for (int t = 0; t < NT - 1; t++) {
                double rate = (ex_prop_win[t] > 0)
                    ? (double)ex_acc_win[t] / ex_prop_win[t] : 0.0;
                fprintf(fex, "%d\t%d\t%.8f\t%.8f\t%lld\t%lld\t%.5f\n",
                        s + 1, t, T_sched[t], T_sched[t + 1],
                        ex_acc_win[t], ex_prop_win[t], rate);
                ex_acc_win[t] = 0;
                ex_prop_win[t] = 0;
            }
            fflush(fex);

            // Reset MC acceptance counters
            CUDA_CHECK(cudaMemset(state.d_accepted, 0, total_replicas * sizeof(long long)));
            CUDA_CHECK(cudaMemset(state.d_proposed, 0, total_replicas * sizeof(long long)));

            // Save spin configurations only in second half of simulation
            if (s >= cfg.mc_iterations / 2) {
                mc_get_spins(state, h_spins);
                int sweep = s + 1;
                fwrite(&sweep, sizeof(int), 1, fconf);
                for (int t = 0; t < NT; t++) {
                    for (int r = 0; r < nrep_per_temp; r++) {
                        int phys = perm[t * nrep_per_temp + r];
                        cuDoubleComplex* rep = h_spins + (long long)phys * cfg.N;
                        for (int i = 0; i < cfg.N; i++) {
                            h_re_im[2*i]     = cuCreal(rep[i]);
                            h_re_im[2*i + 1] = cuCimag(rep[i]);
                        }
                        fwrite(h_re_im, sizeof(double), 2 * cfg.N, fconf);
                    }
                }
                fflush(fconf);
            }

            // Verbose output
            if (cfg.verbose >= 2) {
                // Row format: sweep  E/N[t] ex[t] mc[t]  for cold, mid, hot
                int t_cold = NT - 1;
                int t_mid  = NT / 2;
                int t_hot  = 0;

                int phys_cold = perm[t_cold * nrep_per_temp];
                int phys_mid  = perm[t_mid  * nrep_per_temp];
                int phys_hot  = perm[t_hot  * nrep_per_temp];

                double ex_cold = (t_cold > 0 && ex_prop_total[t_cold - 1] > 0)
                    ? (double)ex_acc_total[t_cold - 1] / ex_prop_total[t_cold - 1] : 0.0;
                double ex_mid  = (t_mid < NT - 1 && ex_prop_total[t_mid] > 0)
                    ? (double)ex_acc_total[t_mid] / ex_prop_total[t_mid] : 0.0;
                double ex_hot  = (t_hot < NT - 1 && ex_prop_total[t_hot] > 0)
                    ? (double)ex_acc_total[t_hot] / ex_prop_total[t_hot] : 0.0;

                // Mean MC acceptance over real replicas at each temperature
                auto mean_mc_acc = [&](int tidx) {
                    double sum = 0;
                    for (int r = 0; r < nrep_per_temp; r++) {
                        int ph = perm[tidx * nrep_per_temp + r];
                        sum += (h_proposed[ph] > 0)
                            ? (double)h_accepted[ph] / h_proposed[ph] : 0.0;
                    }
                    return sum / nrep_per_temp;
                };
                double mc_cold = mean_mc_acc(t_cold);
                double mc_mid  = mean_mc_acc(t_mid);
                double mc_hot  = mean_mc_acc(t_hot);

                struct timespec t_now;
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                double elapsed_now = (t_now.tv_sec - t_start.tv_sec)
                                   + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
                double avg_s_per_it = elapsed_now / (s + 1);

                printf("  %6d  E/N[%d]=% .3e ex=%.3f mc=%.3f  E/N[%d]=% .3e ex=%.3f mc=%.3f  E/N[%d]=% .3e ex=%.3f mc=%.3f  [%.2e s/it]\n",
                       s + 1,
                       t_cold, h_energies[phys_cold] / cfg.N, ex_cold, mc_cold,
                       t_mid,  h_energies[phys_mid]  / cfg.N, ex_mid,  mc_mid,
                       t_hot,  h_energies[phys_hot]  / cfg.N, ex_hot,  mc_hot,
                       avg_s_per_it);
            } else if (cfg.verbose == 1) {
                // Compact: one line with energy at coldest T
                int phys_cold = perm[(NT - 1) * nrep_per_temp];
                int phys_hot  = perm[0];
                printf("  sweep %d/%d:  E_hot/N=% .3e  E_cold/N=% .3e\n",
                       s + 1, cfg.mc_iterations,
                       h_energies[phys_hot] / cfg.N,
                       h_energies[phys_cold] / cfg.N);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    // Final summary
    printf("\n"); box_sec("Summary");
    printf("  %-22s %.3f s\n", "total time", elapsed);
    printf("  %-22s %.4f ms\n", "time/sweep", elapsed / cfg.mc_iterations * 1e3);
    printf("  %-22s %d\n", "sweeps", cfg.mc_iterations);
    printf("  %-22s %d\n", "total replicas", total_replicas);
    printf("\n"); box_sec("Exchange Rates");
    for (int t = 0; t < NT - 1; t++) {
        double rate = (ex_prop_total[t] > 0)
            ? (double)ex_acc_total[t] / ex_prop_total[t] : 0.0;
        printf("  T[%d]-T[%d]  (%.4f-%.4f)  %lld/%lld = %.4f\n",
               t, t + 1, T_sched[t], T_sched[t + 1],
               ex_acc_total[t], ex_prop_total[t], rate);
    }
    printf("\n");

    // Save timing
    char timefile[256];
    snprintf(timefile, sizeof(timefile), "%s/time.txt", datadir);
    FILE* ftim = fopen(timefile, "w");
    if (ftim) {
        fprintf(ftim, "# N NT nrep_per_temp total_replicas mc_iterations pt_freq save_freq Tmax Tmin total_time_s ms_per_sweep\n");
        fprintf(ftim, "%d %d %d %d %d %d %d %.6f %.6f %.6f %.6f\n",
                cfg.N, NT, nrep_per_temp, total_replicas,
                cfg.mc_iterations, pt_freq, cfg.save_freq,
                Tmax, Tmin, elapsed, elapsed / cfg.mc_iterations * 1e3);
        fclose(ftim);
    }

    // Cleanup
    fclose(fout);
    fclose(fex);
    fclose(fconf);
    delete[] T_sched;
    delete[] beta_sched;
    delete[] h_betas;
    delete[] perm;
    delete[] h_energies;
    delete[] h_accepted;
    delete[] h_proposed;
    delete[] h_spins;
    delete[] h_re_im;
    delete[] ex_acc_total;
    delete[] ex_prop_total;
    delete[] ex_acc_win;
    delete[] ex_prop_win;
    mc_free(state);
    return 0;
}
