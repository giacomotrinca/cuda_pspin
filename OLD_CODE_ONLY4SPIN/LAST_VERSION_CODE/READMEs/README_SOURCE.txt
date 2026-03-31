The main is SMrandomTetrads.cu and is contained in the directory SOURCE 
together with all the other libraries and files connected to it.

>>> Temperatures generation

>>> Graph generation
    The graph of interactions is generated with the function InitGraphStructure() 
    defined in the header SMrandomTetrads_CPU_GPU_initializations.h. In turn,
    InitGraphStructure() calls
    > generateTetradsHashing_giacomo() defined in tetrads.cpp 
      to generate the fully connected graph;
    > generate4pletsCycles_discreteWs_wPlaqs() defined in generate4plets.cpp 
      to decimate the graph leaving only the 4plets that satisfy the FMC.
NB tetrads.cpp contains functions to generate the fully connected grphs
   and generate4plets.cpp contains functions for the dilution.


>>> Creation of replicas [...]


>>> MonteCarlo and Parallel Tempering
    > The function trajectory_MonteCarlo() performs the local MC dynamics for each of the NREPLICAS
    > The function exchange_replicas_Parallel_Tempering() performs the swaps among the adjacent replicas NPT
    Both functions are defined in SMrandomTetrads_CPU_GPU_initializations.h.
>>> Output
    The output files are printed by the functions:
    > print_replica() 
    > print_configuration()
    both defined in SMrandomTetrads_CPU_GPU_initializations.h.

Every ind_iter=0,...,NITERATIONS the function trajectory_MonteCarlo() is called for each of the NREPLICAS:
NSTEP of Monte Carlo (i_montecarlo=0,...,NSTEP) are performed. Then the function print_replica() 
generates the output files parallel_tempering#.dat (one for each of the NREPLICAS) and their résumé on stdout. 
Then the function exchange_replicas_Parallel_Tempering() propose a swap among adjacent replicas NPT 
that is performed or not dependig on the overlap of energy distributions.
Then if(ind_iter>NITER_MIN_PRINT && ind_iter%NITER_PRINT_CONF==0) the function print_configuration() generates
the output file config_nrep%d_iter_%d.dat saving the configurations: the first conditions fixes a minimun 
iteration to start saving the configurations (the first ones are surely far from equilibrium); the second one
fixes an interval of iterations after which saving a configuration.
