// Smoothed Cube visualization test
//
// For N complex spins on the smoothed-cube surface: sum_k |a_k|^4 = N
//
// In dimension N=2 the constraint |a_1|^4 + |a_2|^4 = 2 is a "squircle"
// (superellipse with exponent 4).  This program:
//
//   1) Generates many independent random configurations on the smoothed cube
//      and plots (Re a_1, Re a_2) → the filled squircle
//   2) Does free MC sweeps (pair rotations, no Hamiltonian → always accepted)
//      and records the trajectory to demonstrate the constraint is preserved.
//
// Output goes to smoothed_cube/ directory.
//
// Usage: smoothed_cube [-N 2] [-iter 10] [-seed 42] [-nsamples 10000]

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/stat.h>

#include "spins_sparse.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Normalize nsamples independent configs to smoothed cube: sum|a_k|^4 = N
// One thread per sample (fine for small N).
// ============================================================================
__global__ void normalize_many_cube_kernel(cuDoubleComplex* all_spins,
                                           int N, int nsamples) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nsamples) return;
    cuDoubleComplex* sp = all_spins + (long long)s * N;
    double sum_r4 = 0.0;
    for (int i = 0; i < N; i++) {
        double re = cuCreal(sp[i]), im = cuCimag(sp[i]);
        double r2 = re * re + im * im;
        sum_r4 += r2 * r2;
    }
    double scale = pow((double)N / sum_r4, 0.25);
    for (int i = 0; i < N; i++)
        sp[i] = make_cuDoubleComplex(cuCreal(sp[i]) * scale,
                                     cuCimag(sp[i]) * scale);
}

// ============================================================================
// Free MC walk on the smoothed cube (no Hamiltonian → always accept).
// Single thread, N/2 pair proposals per sweep.
// Saves configurations at regular intervals.
// ============================================================================
__global__ void free_mc_kernel(
    cuDoubleComplex* spins, int N,
    int n_sweeps, int save_freq,
    unsigned long long seed,
    cuDoubleComplex* trajectory, int* d_n_saved
) {
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, 0, 0, &rng);

    int saved = 0;
    int n_steps = N / 2;
    if (n_steps < 1) n_steps = 1;

    // Save initial state
    for (int k = 0; k < N; k++)
        trajectory[saved * N + k] = spins[k];
    saved++;

    for (int sw = 0; sw < n_sweeps; sw++) {
        for (int p = 0; p < n_steps; p++) {
            int i0, j0;
            if (N == 2) {
                i0 = 0; j0 = 1;
            } else {
                i0 = (int)(curand_uniform_double(&rng) * N);
                if (i0 >= N) i0 = N - 1;
                j0 = (int)(curand_uniform_double(&rng) * (N - 1));
                if (j0 >= N - 1) j0 = N - 2;
                if (j0 >= i0) j0++;
            }
            cuDoubleComplex ai_new, aj_new;
            propose_pair_rotation_cube(spins[i0], spins[j0], &rng,
                                       &ai_new, &aj_new);
            // No Hamiltonian → always accept
            spins[i0] = ai_new;
            spins[j0] = aj_new;
        }
        if ((sw + 1) % save_freq == 0) {
            for (int k = 0; k < N; k++)
                trajectory[saved * N + k] = spins[k];
            saved++;
        }
    }
    *d_n_saved = saved;
}

// ============================================================================
// Analytic boundary curves (host)
// ============================================================================

// Squircle: |x|^4 + |y|^4 = N  (full boundary, all four quadrants)
static void write_squircle(const char* fname, int N, int npts) {
    FILE* f = fopen(fname, "w");
    if (!f) { perror(fname); return; }
    double s = pow((double)N, 0.25);
    fprintf(f, "# Squircle boundary: |x|^4 + |y|^4 = %d\n", N);
    for (int i = 0; i <= npts; i++) {
        double t = 2.0 * M_PI * i / npts;
        double ct = cos(t), st = sin(t);
        double x = (ct >= 0 ? 1.0 : -1.0) * s * sqrt(fabs(ct));
        double y = (st >= 0 ? 1.0 : -1.0) * s * sqrt(fabs(st));
        fprintf(f, "%.8f %.8f\n", x, y);
    }
    fclose(f);
}

// Superellipse: r1^4 + r2^4 = N  (first quadrant, r1,r2 >= 0)
static void write_superellipse(const char* fname, int N, int npts) {
    FILE* f = fopen(fname, "w");
    if (!f) { perror(fname); return; }
    double s = pow((double)N, 0.25);
    fprintf(f, "# Superellipse: r1^4 + r2^4 = %d  (first quadrant)\n", N);
    for (int i = 0; i <= npts; i++) {
        double t = 0.5 * M_PI * i / npts;
        double r1 = s * sqrt(cos(t));
        double r2 = s * sqrt(sin(t));
        fprintf(f, "%.8f %.8f\n", r1, r2);
    }
    fclose(f);
}

// ============================================================================
// Generate gnuplot scripts and run them
// ============================================================================
static void generate_plots(int N, int sweeps) {
    double s = pow((double)N, 0.25);
    double margin = s * 1.15;

    // 1) Scatter of Re(a1) vs Re(a2) — filled squircle
    {
        FILE* f = fopen("smoothed_cube/init_scatter.gp", "w");
        fprintf(f,
            "set terminal pngcairo enhanced size 900,900 font 'Helvetica,14'\n"
            "set output 'smoothed_cube/init_scatter.png'\n"
            "set size square\n"
            "set xlabel 'Re(a_1)' font 'Helvetica-Oblique,16'\n"
            "set ylabel 'Re(a_2)' font 'Helvetica-Oblique,16'\n"
            "set title 'Smoothed Cube: random samples  (N=%d)' "
            "font 'Helvetica-Bold,16'\n"
            "set xrange [%.4f:%.4f]\n"
            "set yrange [%.4f:%.4f]\n"
            "set grid ls 0 lc rgb '#CCCCCC' lw 1 dt 2\n"
            "plot 'smoothed_cube/init_scatter.dat' u 1:2 w dots "
            "lc rgb '#4393c3' notitle, \\\n"
            "     'smoothed_cube/squircle.dat' u 1:2 w lines lw 2.5 "
            "lc rgb '#d6604d' title '|x|^4+|y|^4=%d'\n",
            N, -margin, margin, -margin, margin, N);
        fclose(f);
        (void)system("gnuplot smoothed_cube/init_scatter.gp");
        printf("  Written smoothed_cube/init_scatter.png\n");
    }

    // 2) Scatter of moduli — points collapse on the superellipse curve
    {
        FILE* f = fopen("smoothed_cube/moduli_scatter.gp", "w");
        fprintf(f,
            "set terminal pngcairo enhanced size 900,900 font 'Helvetica,14'\n"
            "set output 'smoothed_cube/moduli_scatter.png'\n"
            "set size square\n"
            "set xlabel '|a_1|' font 'Helvetica-Oblique,16'\n"
            "set ylabel '|a_2|' font 'Helvetica-Oblique,16'\n"
            "set title 'Smoothed Cube: moduli  (N=%d)' "
            "font 'Helvetica-Bold,16'\n"
            "set xrange [0:%.4f]\n"
            "set yrange [0:%.4f]\n"
            "set grid ls 0 lc rgb '#CCCCCC' lw 1 dt 2\n"
            "plot 'smoothed_cube/moduli.dat' u 1:2 w dots "
            "lc rgb '#66c2a5' notitle, \\\n"
            "     'smoothed_cube/superellipse.dat' u 1:2 w lines lw 2.5 "
            "lc rgb '#d6604d' title 'r_1^4+r_2^4=%d'\n",
            N, margin, margin, N);
        fclose(f);
        (void)system("gnuplot smoothed_cube/moduli_scatter.gp");
        printf("  Written smoothed_cube/moduli_scatter.png\n");
    }

    // 3) MC trajectory — 2D scatter colored by sweep
    {
        FILE* f = fopen("smoothed_cube/mc_scatter.gp", "w");
        fprintf(f,
            "set terminal pngcairo enhanced size 900,900 font 'Helvetica,14'\n"
            "set output 'smoothed_cube/mc_scatter.png'\n"
            "set size square\n"
            "set xlabel 'Re(a_1)' font 'Helvetica-Oblique,16'\n"
            "set ylabel 'Re(a_2)' font 'Helvetica-Oblique,16'\n"
            "set title 'MC trajectory on Smoothed Cube  (N=%d)' "
            "font 'Helvetica-Bold,16'\n"
            "set xrange [%.4f:%.4f]\n"
            "set yrange [%.4f:%.4f]\n"
            "set palette defined (0 '#2166ac', 0.5 '#f4a582', 1 '#b2182b')\n"
            "set cblabel 'sweep'\n"
            "set grid ls 0 lc rgb '#CCCCCC' lw 1 dt 2\n"
            "plot 'smoothed_cube/squircle.dat' u 1:2 w lines lw 2 "
            "lc rgb '#d6604d' title '|x|^4+|y|^4=%d', \\\n"
            "     'smoothed_cube/mc_trajectory.dat' u 2:3:1 w points pt 7 "
            "ps 0.4 lc palette z title 'MC path'\n",
            N, -margin, margin, -margin, margin, N);
        fclose(f);
        (void)system("gnuplot smoothed_cube/mc_scatter.gp");
        printf("  Written smoothed_cube/mc_scatter.png\n");
    }

    // 4) MC trajectory — 3D splot (Re(a1), Re(a2), sweep)
    {
        FILE* f = fopen("smoothed_cube/mc_splot.gp", "w");
        fprintf(f,
            "set terminal pngcairo enhanced size 1200,900 font 'Helvetica,14'\n"
            "set output 'smoothed_cube/mc_splot.png'\n"
            "set xlabel 'Re(a_1)' font 'Helvetica-Oblique,14'\n"
            "set ylabel 'Re(a_2)' font 'Helvetica-Oblique,14'\n"
            "set zlabel 'sweep' font 'Helvetica-Oblique,14'\n"
            "set title 'MC trajectory on Smoothed Cube  (N=%d, %d sweeps)' "
            "font 'Helvetica-Bold,16'\n"
            "set palette defined (0 '#2166ac', 0.5 '#f4a582', 1 '#b2182b')\n"
            "set cblabel 'sweep'\n"
            "set grid\n"
            "set ticslevel 0\n"
            "splot 'smoothed_cube/squircle.dat' u 1:2:(0) w lines lw 1.5 "
            "lc rgb '#cccccc' title '|x|^4+|y|^4=%d  (z=0)', \\\n"
            "      'smoothed_cube/mc_trajectory.dat' u 2:3:1 every 1 "
            "w linespoints pt 7 ps 0.3 lw 0.5 lc palette z title 'MC path'\n",
            N, sweeps, N);
        fclose(f);
        (void)system("gnuplot smoothed_cube/mc_splot.gp");
        printf("  Written smoothed_cube/mc_splot.png\n");
    }

    // 5) Constraint check: S4 vs sweep
    {
        FILE* f = fopen("smoothed_cube/constraint_check.gp", "w");
        fprintf(f,
            "set terminal pngcairo enhanced size 900,500 font 'Helvetica,14'\n"
            "set output 'smoothed_cube/constraint_check.png'\n"
            "set xlabel 'sweep' font 'Helvetica-Oblique,14'\n"
            "set ylabel 'S_4 - N' font 'Helvetica-Oblique,14'\n"
            "set title 'Constraint residual:  {/Symbol S}|a_k|^4 - N  "
            "(N=%d)' font 'Helvetica-Bold,16'\n"
            "set grid ls 0 lc rgb '#CCCCCC' lw 1 dt 2\n"
            "set format y '%%.2e'\n"
            "plot 'smoothed_cube/mc_trajectory.dat' u 1:($6 - %d) w lines "
            "lw 1.5 lc rgb '#2166ac' title 'S_4 - %d', \\\n"
            "     0 w lines lw 2 lc rgb '#d6604d' dt 2 title '0'\n",
            N, N, N);
        fclose(f);
        (void)system("gnuplot smoothed_cube/constraint_check.gp");
        printf("  Written smoothed_cube/constraint_check.png\n");
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    int N = 2;
    int iter = 10;
    unsigned long long seed = 42;
    int nsamples = 10000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc)
            iter = atoi(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
            seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-nsamples") == 0 && i + 1 < argc)
            nsamples = atoi(argv[++i]);
        else {
            fprintf(stderr,
                "Usage: %s [-N 2] [-iter 10] [-seed 42] [-nsamples 10000]\n",
                argv[0]);
            exit(1);
        }
    }

    if (N < 2) {
        fprintf(stderr, "Error: N must be >= 2.\n");
        exit(1);
    }

    int sweeps = 1 << iter;
    int save_freq = (sweeps > 2048) ? sweeps / 1024 : 1;

    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║    Smoothed Cube Visualization Test          ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    printf("  N        = %d\n", N);
    printf("  sweeps   = %d  (2^%d)\n", sweeps, iter);
    printf("  seed     = %llu\n", seed);
    printf("  nsamples = %d\n", nsamples);
    printf("  save_freq= %d  (~%d trajectory points)\n\n",
           save_freq, sweeps / save_freq + 1);

    mkdir("smoothed_cube", 0755);

    // =================================================================
    // 1) Generate nsamples independent configs on the smoothed cube
    // =================================================================
    printf("─ Generating %d independent configurations ...\n", nsamples);

    long long total = (long long)nsamples * N;
    cuDoubleComplex* d_all;
    CUDA_CHECK(cudaMalloc(&d_all, total * sizeof(cuDoubleComplex)));

    // Fill with Gaussians
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    long long count = 2 * total;
    if (count % 2 != 0) count++;
    CURAND_CHECK(curandGenerateNormalDouble(gen, (double*)d_all, count, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // Normalize each config to sum|a_k|^4 = N
    int bs = 256;
    int gs = (nsamples + bs - 1) / bs;
    normalize_many_cube_kernel<<<gs, bs>>>(d_all, N, nsamples);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuDoubleComplex* h_all = new cuDoubleComplex[total];
    CUDA_CHECK(cudaMemcpy(h_all, d_all, total * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_all));

    // Write scatter data
    {
        FILE* f = fopen("smoothed_cube/init_scatter.dat", "w");
        fprintf(f, "# Re(a_1)  Re(a_2)\n");
        for (int s = 0; s < nsamples; s++) {
            double x1 = h_all[(long long)s * N + 0].x;
            double x2 = h_all[(long long)s * N + 1].x;
            fprintf(f, "%.8f %.8f\n", x1, x2);
        }
        fclose(f);
        printf("  Written smoothed_cube/init_scatter.dat\n");
    }
    {
        FILE* f = fopen("smoothed_cube/moduli.dat", "w");
        fprintf(f, "# |a_1|  |a_2|\n");
        for (int s = 0; s < nsamples; s++) {
            cuDoubleComplex a1 = h_all[(long long)s * N + 0];
            cuDoubleComplex a2 = h_all[(long long)s * N + 1];
            double r1 = sqrt(a1.x * a1.x + a1.y * a1.y);
            double r2 = sqrt(a2.x * a2.x + a2.y * a2.y);
            fprintf(f, "%.8f %.8f\n", r1, r2);
        }
        fclose(f);
        printf("  Written smoothed_cube/moduli.dat\n");
    }
    delete[] h_all;

    // Analytic boundary curves
    write_squircle("smoothed_cube/squircle.dat", N, 1000);
    printf("  Written smoothed_cube/squircle.dat\n");
    write_superellipse("smoothed_cube/superellipse.dat", N, 500);
    printf("  Written smoothed_cube/superellipse.dat\n");

    // =================================================================
    // 2) MC trajectory (free walk, no Hamiltonian)
    // =================================================================
    printf("\n─ Running %d MC sweeps (free walk, no Hamiltonian) ...\n", sweeps);

    cuDoubleComplex* d_spins;
    CUDA_CHECK(cudaMalloc(&d_spins, N * sizeof(cuDoubleComplex)));
    init_spins_cube(d_spins, N, seed + 9999);

    int max_saved = sweeps / save_freq + 2;
    cuDoubleComplex* d_traj;
    CUDA_CHECK(cudaMalloc(&d_traj,
                          (long long)max_saved * N * sizeof(cuDoubleComplex)));
    int* d_nsaved;
    CUDA_CHECK(cudaMalloc(&d_nsaved, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nsaved, 0, sizeof(int)));

    free_mc_kernel<<<1, 1>>>(d_spins, N, sweeps, save_freq,
                             seed + 12345, d_traj, d_nsaved);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_nsaved;
    CUDA_CHECK(cudaMemcpy(&h_nsaved, d_nsaved, sizeof(int),
                          cudaMemcpyDeviceToHost));

    cuDoubleComplex* h_traj = new cuDoubleComplex[(long long)h_nsaved * N];
    CUDA_CHECK(cudaMemcpy(h_traj, d_traj,
                          (long long)h_nsaved * N * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    {
        FILE* f = fopen("smoothed_cube/mc_trajectory.dat", "w");
        fprintf(f, "# sweep  Re(a1)  Re(a2)  |a1|  |a2|  S4\n");
        for (int i = 0; i < h_nsaved; i++) {
            cuDoubleComplex a1 = h_traj[(long long)i * N + 0];
            cuDoubleComplex a2 = h_traj[(long long)i * N + 1];
            double r1 = sqrt(a1.x * a1.x + a1.y * a1.y);
            double r2 = sqrt(a2.x * a2.x + a2.y * a2.y);
            double S4 = 0.0;
            for (int k = 0; k < N; k++) {
                cuDoubleComplex ak = h_traj[(long long)i * N + k];
                double rk = sqrt(ak.x * ak.x + ak.y * ak.y);
                S4 += rk * rk * rk * rk;
            }
            int sweep_idx = (i == 0) ? 0 : i * save_freq;
            fprintf(f, "%d %.8f %.8f %.8f %.8f %.15f\n",
                    sweep_idx, a1.x, a2.x, r1, r2, S4);
        }
        fclose(f);
        printf("  Written smoothed_cube/mc_trajectory.dat (%d points)\n",
               h_nsaved);
    }

    delete[] h_traj;
    CUDA_CHECK(cudaFree(d_spins));
    CUDA_CHECK(cudaFree(d_traj));
    CUDA_CHECK(cudaFree(d_nsaved));

    // =================================================================
    // 3) Generate plots
    // =================================================================
    printf("\n─ Generating plots ...\n");
    generate_plots(N, sweeps);

    printf("\nDone.  See smoothed_cube/*.png\n\n");
    return 0;
}
