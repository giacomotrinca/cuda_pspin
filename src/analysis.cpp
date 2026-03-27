#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

struct DataPoint {
    int iter;
    std::vector<double> energy;  // E/N per replica
    std::vector<double> acc;     // acceptance per replica
};

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s -N <N> [-nrep <nrep>] [-label <l>] [-datadir <path>]\n", prog);
    exit(1);
}

int main(int argc, char** argv) {
    int N = 0;
    int nrep = 1;
    int label = -1;
    std::string datadir_override;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-label") == 0 && i + 1 < argc)
            label = atoi(argv[++i]);
        else if (strcmp(argv[i], "-datadir") == 0 && i + 1 < argc)
            datadir_override = argv[++i];
        else usage(argv[0]);
    }
    if (N < 4) usage(argv[0]);

    // Data directory
    char datadir[256];
    if (!datadir_override.empty())
        snprintf(datadir, sizeof(datadir), "%s", datadir_override.c_str());
    else if (label >= 0)
        snprintf(datadir, sizeof(datadir), "data/N%d_NR%d_S%d", N, nrep, label);
    else
        snprintf(datadir, sizeof(datadir), "data/N%d_NR%d", N, nrep);

    // Read energy_accept.txt
    char infile[512];
    snprintf(infile, sizeof(infile), "%s/energy_accept.txt", datadir);
    FILE* fin = fopen(infile, "r");
    if (!fin) {
        fprintf(stderr, "Cannot open %s\n", infile);
        return 1;
    }

    std::vector<DataPoint> data;
    char line[4096];
    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        DataPoint dp;
        dp.energy.resize(nrep);
        dp.acc.resize(nrep);

        char* tok = strtok(line, " \t\n");
        if (!tok) continue;
        dp.iter = atoi(tok);

        bool ok = true;
        for (int r = 0; r < nrep; r++) {
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            dp.energy[r] = atof(tok);
            tok = strtok(nullptr, " \t\n");
            if (!tok) { ok = false; break; }
            dp.acc[r] = atof(tok);
        }
        if (!ok) continue;
        data.push_back(dp);
    }
    fclose(fin);

    int M = (int)data.size();
    if (M == 0) {
        fprintf(stderr, "No data points found in %s\n", infile);
        return 1;
    }

    printf("Read %d data points from %s (%d replicas)\n\n", M, infile, nrep);

    // ================================================================
    // Block averaging with doubling block sizes.
    // Last block = last half of data,  second-to-last = previous quarter, etc.
    // Blocks (from most recent to oldest):
    //   [M/2, M), [M/4, M/2), [M/8, M/4), ... , [0, ...)
    // ================================================================

    // Build block boundaries from end to start
    struct Block { int start; int end; }; // [start, end)
    std::vector<Block> blocks;
    int pos = M;
    int bsize = M / 2;
    if (bsize < 1) bsize = 1;
    while (pos > 0) {
        int bstart = pos - bsize;
        if (bstart < 0) bstart = 0;
        blocks.push_back({bstart, pos});
        pos = bstart;
        bsize /= 2;
        if (bsize < 1) bsize = 1;
    }
    // Reverse so blocks go from earliest to latest
    std::reverse(blocks.begin(), blocks.end());

    int nblocks = (int)blocks.size();

    // Output file
    char outfile[512];
    snprintf(outfile, sizeof(outfile), "%s/block_energy.txt", datadir);
    FILE* fout = fopen(outfile, "w");
    if (!fout) {
        fprintf(stderr, "Cannot open %s for writing\n", outfile);
        return 1;
    }

    fprintf(fout, "# Block averaging of E/N (doubling block sizes, last block = last half)\n");
    fprintf(fout, "# Errors: jackknife\n");
    fprintf(fout, "# Columns: 1:block  2:iter_start  3:iter_end  4:n_samples");
    int col = 5;
    for (int r = 0; r < nrep; r++) {
        fprintf(fout, "  %d:<E%d/N>  %d:err%d", col, r, col+1, r);
        col += 2;
    }
    fprintf(fout, "\n");

    printf("Block   iter_range          samples");
    for (int r = 0; r < nrep; r++) printf("    <E%d/N>      err%d   ", r, r);
    printf("\n");
    printf("--------------------------------------------------------------");
    for (int r = 0; r < nrep; r++) printf("------------------------");
    printf("\n");

    for (int b = 0; b < nblocks; b++) {
        int s = blocks[b].start;
        int e = blocks[b].end;
        int n = e - s;

        fprintf(fout, "%d\t%d\t%d\t%d", b, data[s].iter, data[e-1].iter, n);
        printf("%-6d  [%7d, %7d]  %6d ", b, data[s].iter, data[e-1].iter, n);

        for (int r = 0; r < nrep; r++) {
            // Full-sample mean
            double sum_full = 0.0;
            for (int i = s; i < e; i++)
                sum_full += data[i].energy[r];
            double mean_full = sum_full / n;

            // Jackknife error
            double jk_sum2 = 0.0;
            for (int j = s; j < e; j++) {
                double jk_mean = (sum_full - data[j].energy[r]) / (n - 1);
                double diff = jk_mean - mean_full;
                jk_sum2 += diff * diff;
            }
            double jk_err = sqrt((double)(n - 1) / n * jk_sum2);

            fprintf(fout, "\t%.8f\t%.8f", mean_full, jk_err);
            printf("  %10.6f  %10.6f", mean_full, jk_err);
        }

        fprintf(fout, "\n");
        printf("\n");
    }

    fclose(fout);
    printf("\nBlock averages written to %s\n", outfile);

    return 0;
}
