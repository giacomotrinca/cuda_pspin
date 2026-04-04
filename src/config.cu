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
        else if (strcmp(argv[i], "-J0") == 0 && i + 1 < argc)
            cfg.J0 = atof(argv[++i]);
        else if (strcmp(argv[i], "-alpha") == 0 && i + 1 < argc)
            cfg.alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "-alpha0") == 0 && i + 1 < argc)
            cfg.alpha0 = atof(argv[++i]);
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
        else if (strcmp(argv[i], "-fmc") == 0 && i + 1 < argc)
            cfg.fmc_mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "-gamma") == 0 && i + 1 < argc)
            cfg.gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "-verbose") == 0) {
            // Accept optional integer level: -verbose [0|1|2]
            if (i + 1 < argc && argv[i+1][0] >= '0' && argv[i+1][0] <= '9')
                cfg.verbose = atoi(argv[++i]);
            else
                cfg.verbose = 1;
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-N n] [-nrep n] [-T temp] [-J coupling] "
                    "[-J0 mean] [-alpha a] [-alpha0 a0] "
                    "[-iter n] [-seed s] [-save_freq k] [-label l] [-dev d] "
                    "[-fmc 0|1|2] [-gamma g] [-verbose [0|1|2]]\n",
                    argv[0]);
            exit(1);
        }
    }

    if (cfg.N < 4) {
        fprintf(stderr, "Error: N must be >= 4 for 4-body interactions.\n");
        exit(1);
    }

    // Set gamma automatically based on FMC mode (if not given via -gamma)
    if (cfg.fmc_mode > 0 && cfg.gamma <= 0.0) {
        if (cfg.fmc_mode == 1)
            cfg.gamma = 0.0;           // comb: exact frequency matching
        else if (cfg.fmc_mode == 2)
            cfg.gamma = 1.0 / (2.0 * (cfg.N - 1));  // uniform: half spacing
    }

    return cfg;
}
