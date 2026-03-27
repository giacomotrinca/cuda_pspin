#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

SimConfig parse_args(int argc, char** argv) {
    SimConfig cfg = default_config();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            cfg.N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc)
            cfg.T = atof(argv[++i]);
        else if (strcmp(argv[i], "-J") == 0 && i + 1 < argc)
            cfg.J = atof(argv[++i]);
        else if (strcmp(argv[i], "-sweeps") == 0 && i + 1 < argc)
            cfg.n_sweeps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-therm") == 0 && i + 1 < argc)
            cfg.n_therm = atoi(argv[++i]);
        else if (strcmp(argv[i], "-measure") == 0 && i + 1 < argc)
            cfg.measure_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
            cfg.seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-delta") == 0 && i + 1 < argc)
            cfg.delta = atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-N n] [-T temp] [-J coupling] "
                    "[-sweeps n] [-therm n] [-measure n] [-seed s] [-delta d]\n",
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
