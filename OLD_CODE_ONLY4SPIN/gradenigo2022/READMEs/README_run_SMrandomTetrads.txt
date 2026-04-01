SMrandomTetrads takes in input (from command line) 
        >>> 5 integer random numbers (seeds of random sequences used in the simulation)  
	>>> 2 real numbers for T_min and T_max
	>>> 1 integer number (0,1,2,3) for the device (one of the 4 GPU's installed on the machine)
For example:
./SMrandomTetrads 3834672 2325183 3656267 9637761 303401 0.5 1.4 1

Thr first five numbers are the seeds of the random sequences: by reusing them, the same graph
of interactions (realization of disorder) and the same initial conditions are created and 
each step of the simulation is repeated. The following two numbers are the temperatures. 
The last one is the device [cfr README_machines].

To automate the compilation and the execution of SMrandomTetrads.cu there is a shell script called 
run_samples.sh. The script run_samples.sh generates a directory for each sample, copies the SOURCE
directory inside and runs the simulation. Moreover it generates another shell script called
run_analysis.sh that will be used to run the analysis source. 

NB: the seeds for each sample are saved in the shell script run_analysis.sh and in launch_sample$i.txt
NB: to see for how much time a simulation has run check last operation on launch_sample$i.txt (START) and on 
    parallel_tempering#.dat (END).

--------------------------------

Principal global parameters:
	>>> Size				size of the system (N)
	>>> PLAQ_NUMBER				pre-fixed # of plaquettes (a power of 2)


	>>> TWO_REPLICA         		ifdef only two replicas are simulated
	>>> NREPLICAS				# of replicas at the same temperature (to compute overlaps)
        >>> NTJUMPS             		# jumps of temperature
	>>> NPT	(NTJUMPS+1)			# of replicas at different temperatures (for Parallel Tempering) 
	>>> NSTEP               		# of steps after which 1 iteration of MC_PT is done 
	>>> NITERATIONS (2^x_steps/NSTEP)   	# of iterations of the MC trajectory 
	>>> NITER_MIN_PRINT (2^y_steps/NSTEP)	minimum # of iterations after which the configurations are printed
                                                (must be at least 1/4 of NITERATIONS)
	>>> NITER_PRINT_CONF    		when iter%NITER_PRINT_CONF==0 configurations are printed
						(take a multiple of 2, like 8)

	>>> REPLICA EXCHANGE			ifdef allows PT
	>>> I_WANT_GAIN         		1 gain active, 0 gain off
	>>> _GainMax_	 			maximum value of the gain
	>>> EQUISPACEDTS			1 equispaced temperatures between T_min and T_max, 
						0 if temperatures around T_c
	>>> USE_REDUCE_UNROLL                   (to be used only if N>76)

	               
-----------------------------	

Outputs of SMrandomTetrads:
	>>> stdout
	>>> parallel_tempering#.dat
	>>> config_nrep%d_iter_%d.dat

>>>In the standard output it are printed:
   > infos about the graph of interactions
   > infos about the allocation of memory
   > some lines of the files parallel_tempering#.dat

>>>parallel_tempering#.dat is a list of NREPLICAS files containing some checks about PT. 
   In particular: each file contains NITERATIONS line and 1+3*NPT columns:
   the 1st colums is the step of MC, i.e. ind_iter*NSTEP; the 3*NPT columns contain
   > temperature 
   > acceptation rate  
   > energy/N  
    
>>>config_nrep%d_iter_%d.dat are lists of files containing the configurations saved during the dynamics.
   In particular: there is one list of files for each of the NREPLICAS, correspondig to nrep%d,
   one file per saved configuration, corrispondig to the iteration iter_%d.
   Each file is made up of NPT blocks of N rows (one per spin); there are 4 columns:
   1st: temperature
   2nd: identity char of the replica 
   3rd: spin_x (modulus?) 
   4th: spin_y (phase?)
   

