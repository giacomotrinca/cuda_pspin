#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

SimConfig parse_args(int argc, char** argv) {
    SimConfig cfg = default_config();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            cfg.N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrep") == 0 && i + 1 < argc)
            cfg.nrep = atoi(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc)
            cfg.T = atof(argv[++i]);
        else if (strcmp(argv[i], "-J") == 0 && i + 1 < argc)
            cfg.J = atof(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc)
            cfg.mc_iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
            cfg.seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-save_freq") == 0 && i + 1 < argc)
            cfg.save_freq = atoi(argv[++i]);
        else if (strcmp(argv[i], "-label") == 0 && i + 1 < argc)
            cfg.label = atoi(argv[++i]);
        else if (strcmp(argv[i], "-dev") == 0 && i + 1 < argc)
            cfg.dev = atoi(argv[++i]);
        else if (strcmp(argv[i], "-verbose") == 0)
            cfg.verbose = 1;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-N n] [-nrep n] [-T temp] [-J coupling] "
                    "[-iter n] [-seed s] [-save_freq k] [-label l] [-dev d] [-verbose]\n",
                    argv[0]);
            exit(1);
        }
    }

    if (cfg.N < 4) {
        fprintf(stderr, "Error: N must be >= 4 for 4-body interactions.\n");
        exit(1);
    }

    return cfg;
}
