// -*- c++ -*-

using namespace std;         // Using the standard library namespace.

// -----------STANDARD LIBRARIES ----- ////// 
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h> // needed for "atoi()" etc.
#include <stdio.h> 
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
// #include <curand_helper.h>
// #include <cuda_runtime.h>

// ----------- THRUST LIBRARIES -------- /////  
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
// #include <thrust/execution_policy.h>

// N (spin) ---> Nplaq (plaquettes)
// 32            2^11 = 2048
// 64            2^14 = 16384 
// 128           2^17 = 131072
// 256           2^20 = 1048576
 
//preferences of the simulation:
///////////////////////////////////////////////////////////////// 
#define Size 64
#define PLAQ_NUMBER 16384


//#define BLOCK_SIZE 512 

// #define SINGLE_REPLICA_OUTPUT

#define REPLICA_EXCHANGE 
#define NCHAR_IDENTITY 8
#define twopi 6.283185307179586
#define I_WANT_GAIN 0  // 1 gain active, 0 gain off // to change gain form see frequencyGeneration.cpp
#define _GainMax_ 1.e-10   // maximum value of the gain (only if I_WANT_GAIN is 1)
#define I_WANT_PT 0  // if 0 no PT, if 1 PT active  
#define EQUISPACEDTS 1 
#define NBINS 60
#define NBINS_LINK 120

// #define USE_REDUCE_UNROLL // to be used ONLY IF N > 64 

// #define ONE_REPLICA   // With this setting it initializes 4 replicas but only
#define NREPLICAS 4   // 1 among these 4 is evolved with stochastic MC dynamics (Parallel Tempering)
#define CYCLE_UPDATE_ENERGY
#define NSTEP 50
#define NITER_PRINT_CONF 10
#define NITER_MIN_PRINT 10000
#define NITERATIONS 20000 
#define PRINT_FREQUENCY 100 
#define NTJUMPS 31
#define NTIMES_PROFILING 10
#define FREQ_ENABLE 2  // if 0 all the frequencies are equal, if 2 equispaced frequencies
#define GAMMA 1 // linewidth in discrete units of wsindices - for FREQ_ENABLE 1 change nf (it is sigma = 4 nf, so that the effective linewidth is proportional to GAMMA/nf )
//#define FULLY 1 // =1 initial tetrads are all N^4. 
#define NPT (NTJUMPS+1) // N-REPLICHE PARALLEL TEMPERING 
#define TSAMP 32 // 256 // 1024 // 256 // 256 // 256 // 64 // 128 // 256  // 64 // 256 // 128 // 32 // 16 // 128 // 256   // can be any number (just compare with TSHUFF and TPT)
#define NBEQUIL 16384 // 32768 // 65536 // 32768 // 65536 // 262144 // 131072 // 65536 // 32768 // 2048 // 4096 // 8192 // 65536 // 524288 // 262144 // 32768   // must be a power of 2
#define NBMEAS 16384 //16384 // 8192 // 32768 // 65536 // 262144 // 131072 // 65536 // 32768 // 2048 // 4096 // 8192 // 65536 // 524288 // 262144 // 32768  // can be any number
#define TSHUFF 2  // if TSHUFF >= tsamp, no reshuffling is done
#define TPT 2  // (TSAMP - 2)   // for parallel tempering
#define PRINT_CONFS 2 // 2  // if 0 does not print confs, if 2 print from the beginning, if 1 just in measures
#define NB_PRINTED_CONFS 2048 // 1024 // 8192 //20000 // // the max value is (NBMEAS+NBEQUIL) for  PRINT_CONFS==2 ; otherwise the max is NBMEAS if PRINT_CONFS==1 
#define REDUCE 2
#define UNROLLING_DE 1
#define N_THREADS_1_BLOCK 128
#define N_DE_REDUCED  (PLAQ_NUMBER/(4*N_THREADS_1_BLOCK*N_THREADS_1_BLOCK)) 

#define NR 1  // number of replicas 
#define DISORDER 1
#define _Jvalue_ 0.     // coefficient of the average of the 4-body interaction for the disordered case
#define _sJvalue_ 1.    // coefficient of the sigma of the 4-body interaction for the disordered case

#define ALL_MEASURES 1    // if 0 it measures only the energy - to be implemented (the xs must be copied on the GPU....)
#define measure_M2 0   // if 1 M^2 is measures, otherwise Mx AND My  ( then there is one additional column for each temperature
#define BINARY 1  // 0 if overlapIFO print formatted, 1 for write in binary

//#define SIGNALCOMP 1  // measures of signal - comment to remove
#define epsilonSM 1.0 // vincolo sferico di ogni spin (cf. tesi di P. Rotondo, p. 66)
#define _Tref_ 0.40726429   // reference temperature for the gain  - note as multiply _GainMax_ by K is the same that divide _Tref_ by K**2

#define MCBS 32    //  must be MCBS <= sfib
#define LBS 32     // 64   


#define sfib    861
#define rfib   1279
#define rpsfib 2140


#define myexp exp
#define mysincos sincos

//#define mysqrt  __fsqrt_rn

#define mysqrt sqrt
typedef double  spin_t;
typedef double  spin_t2; // spin_t2 is of higher quality (random numbers)
/////////////////////////////////////////////////////////////////


// defines for random generator
#define ONEOVTWOSQRTWO 0.353553390593274
#define Arand 1664525
#define Crand 1013904223
#define INVMrand 2.328306437080797e-10
#define arand 0.000387552
#define crand 0.236067973
#define MULTrand2 4.6566128752457969e-10
#define Mrand 4294967296
#define ranLCG(I) ((Arand*I + Crand)%Mrand)


//------------------------------------------------//////
//---------------- MACROS CUDA ------------------ ////// 
// ---------------------------------------------- //////

#define CHECK(call)							\
  {									\
    const cudaError_t error = call;					\
    if (error != cudaSuccess)						\
      {									\
        fprintf(stderr, "# Error: %s:%d, ", __FILE__, __LINE__);	\
        fprintf(stderr, "# code: %d, reason: %s\n", error,		\
                cudaGetErrorString(error));				\
        exit(1);							\
      }									\
  }


#define CUDA_CALL( call )                \
  {					 \
    cudaError_t result = call;						\
    if ( cudaSuccess != result )					\
      std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl; \
  }

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__); exit(1); }} while(0)

#include "SMrandomTetrads_structures.h" // giac 09-2017 
#include "hashing.cpp"
#include "graph.cpp"
#include "generate4plets.cpp"  // here is used a structure defined in "SMrandomTetrads_structures.h"

#include <gsl_randist.h>
#include <gsl_rng.h>

#include "tetrads.cpp"
#include "generateQuadsFully.cpp"
#include "parallelMCstep.cu"
#include "functionsSM_CPU.cpp"
#include "SMrandomTetrads_CPU_GPU_initializations.h"

int main(int argc, char ** argv)
{
  
  ////////////////////////// ////////////////////////// //////////////////////////
  
#if EQUISPACEDTS
  if(argc < 6){
    printf("\nERROR: usage SMrandomTetrads.out seed seed2 Tmin Tmax GPU_index \n\n");
    exit(1);
  }
#else
  if(argc < 5){
    printf("\nERROR: usage SMrandomTetrads.out seed seed2 Tc \n\n");
    exit(1);
  }
#endif
  
  if(I_WANT_GAIN != 0 && I_WANT_PT != 0){
    printf("\nERROR: don't use PT with gain !!!\n\n");
    exit(4);
  }
  
  FILE * myfile2 = fopen("about.txt","w");
  
  int seed_int  = atoi(argv[1]);
  int seed0 = atoi(argv[2]);
  int seed1 = atoi(argv[3]);
  int seed2 = atoi(argv[4]);
  int seed3 = atoi(argv[5]);


#if EQUISPACEDTS
  double Tmin = atof(argv[6]);
  double Tmax = atof(argv[7]);
  int device = atoi(argv[8]);
#else
  double Tc = atof(argv[6]);
  int device = atoi(argv[7]);
#endif

  int N = Size;
  
  
  int nPT = NPT;
  double * beta = (double *)calloc(nPT,sizeof(double));
  double * temp = (double *)calloc(nPT,sizeof(double));

  double dT=(Tmax-Tmin)/NTJUMPS;

  //for(int i=0;i<NPT;i++){
  //temp[i]=Tmax-(double)i*dT;
  //}
  
  double beta_max = 1./Tmin; // N = 128 ==> Tmin = 0.775
  double beta_min = 1./Tmax; // N = 128 ==> Tmax = 1.300

  double delta_beta = (beta_max-beta_min)/NTJUMPS;
  
  for(int i=0;i<NPT;i++){
    temp[i]=1./(beta_min+(double)i*delta_beta);
  }
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  // TEMPERATURE INDEX INCREASES AS THE TEMPERATURE IS LOWERED //
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  printf("# N_DE_REDUCED = %d \n", N_DE_REDUCED);
  
#include "tempsGeneration.cpp"
  
  ////////////////////////////////////////////////////
  /// --- CUDA UTILITIES -----------------------//////
  ////////////////////////////////////////////////////
  /// --- LEARN THE NUMBER OF THREADS PER CORE -----//
  ////////////////////////////////////////////////////

  
  int dev; //driverVersion=0, runtimeVersion=0; 
  dev=device;
  
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\n# Device %d: \"%s\"\n", dev, deviceProp.name);
  //printf("Maximum number of threads per multiprocessor: %d\n\n", deviceProp.maxThreadsPerMultiProcessor);
  int n_threads=deviceProp.maxThreadsPerBlock;

  printf("# Maximum sizes of x dimension of a grid: %12.8e\n", (double)deviceProp.maxGridSize[0]);
  
  printf("# Maximum number of threads per block: %d\n\n", N_THREADS_1_BLOCK);

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  // ///////////////////////////////////////////////////

  int Nplaqs = PLAQ_NUMBER;

  Plaqs_type * placchette; // definisce il puntatore   
  placchette = (Plaqs_type *)calloc(Nplaqs,sizeof(Plaqs_type)); // alloca le placchette
  
  double * ws = new double [N];
  int * wsindices = new int [N];
  double * gain = new double [N];
  
  ///////////////// ABOUT FREQUENCIES TO PUT ON THE GRAPH ////////////////////////// 
  
#include "frequencyGeneration.cpp"
  
#if I_WANT_GAIN
  {
    FILE *fileg = fopen("final_gain.dat","w");
    for(int i=0;i<N;i++){
      fprintf(fileg,"%d %f \n", i, gain[i]);
    }
    fclose(fileg);
  }
#endif
  
  ///////////////////////////////////////////////////////////////////////////////// 

  ////// ----- SI FA UNA VOLTA SOLA --- UGUALE PER TUTTI ----- ////// 

  InitGraphStructure(seed_int,N,placchette,wsindices);

  ///////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////
  // 2.2 INITIALIZATION of INTERACTIONS on HOST ////////////
  //////////////////////////////////////////////////////////
  
  Int_type * inter;
  inter = (Int_type *)calloc(1,sizeof(Int_type));
  
  init_Interactions_host(Nplaqs,inter,placchette);
  
  //printf("TUTTO OK con le INTERAZIONI \n");
  
  /////////////////// ---- DISTRUGGI LE PLACCHETTE ----- ///// 
  free(placchette); //////////////////////////////////////////
  ////////////////////////////////////////////////////////////

  //    ///////////////////////////////////////////////////////////// 
  //  /// A. PUT "INTERACTIONS" ON THE DEVICE /////////////////////
  //  /////////////////////////////////////////////////////////////
  
  Int_type * d_inter;
  cudaMalloc((Int_type **) &d_inter,sizeof(Int_type));
  
  init_Inter_device(Nplaqs,inter,d_inter);
  
  printf("# Ho copiato le placchette e accoppiamenti sul device \n");

  FILE * fout_inter = fopen("interactions_file_reconstruction.dat","w");
  for(int i=0; i<Nplaqs; i++)
    fprintf(fout_inter,"%d %d %d %d %12.8e \n",inter->spin_index[i*4],inter->spin_index[i*4+1],inter->spin_index[i*4+2],inter->spin_index[i*4+3],inter->J[i]);
  fclose(fout_inter);

  ///////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////////////////////
  // -- ALLOCAZIONE CONTATORI ED ENERGIA RUNNING TIME ------ ////////
  ///////////////////////////////////////////////////////////////////

  Clock_type * clock0, * clock1, * clock2, * clock3;
  Clock_type * d_clock0, * d_clock1, * d_clock2, * d_clock3;

  clock0 = (Clock_type *)calloc(1,sizeof(Clock_type));
#ifndef ONE_REPLICA
  clock1 = (Clock_type *)calloc(1,sizeof(Clock_type));
  clock2 = (Clock_type *)calloc(1,sizeof(Clock_type));
  clock3 = (Clock_type *)calloc(1,sizeof(Clock_type));
#endif

  cudaMalloc((Clock_type **) &d_clock0,sizeof(Clock_type));
#ifndef ONE_REPLICA
  cudaMalloc((Clock_type **) &d_clock1,sizeof(Clock_type));
  cudaMalloc((Clock_type **) &d_clock2,sizeof(Clock_type));
  cudaMalloc((Clock_type **) &d_clock3,sizeof(Clock_type));
#endif

  Initialize_counters(clock0,d_clock0);
#ifndef ONE_REPLICA
  Initialize_counters(clock1,d_clock1);
  Initialize_counters(clock2,d_clock2);
  Initialize_counters(clock3,d_clock3);
#endif

  /////////////////////////////////////////////////////////////////
  /// CREAZIONE DELLE REPLICHE ////////////////////////////////////
  /////////////////////////////////////////////////////////////////

  Conf_type ** sys0, ** sys1, ** sys2, ** sys3;
  Conf_type ** d_sys0, ** d_sys1, ** d_sys2, ** d_sys3; 
  
  sys0 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#ifndef ONE_REPLICA
  sys1 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  sys2 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  sys3 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#endif

  d_sys0 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#ifndef ONE_REPLICA
  d_sys1 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  d_sys2 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  d_sys3 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#endif

  int Ncoppie = N/2;
  
  MC_type ** d_mc_step0, ** d_mc_step1, ** d_mc_step2, ** d_mc_step3;
  MC_type ** mc_step0, ** mc_step1, ** mc_step2, ** mc_step3; 

  d_mc_step0 = (MC_type **)calloc(NPT,sizeof(MC_type *));
#ifndef ONE_REPLICA
  d_mc_step1 = (MC_type **)calloc(NPT,sizeof(MC_type *));
  d_mc_step2 = (MC_type **)calloc(NPT,sizeof(MC_type *));
  d_mc_step3 = (MC_type **)calloc(NPT,sizeof(MC_type *));
#endif

  mc_step0 = (MC_type **)calloc(NPT,sizeof(MC_type *));  
#ifndef ONE_REPLICA
  mc_step1 = (MC_type **)calloc(NPT,sizeof(MC_type *));  
  mc_step2 = (MC_type **)calloc(NPT,sizeof(MC_type *));  
  mc_step3 = (MC_type **)calloc(NPT,sizeof(MC_type *));    
#endif

  crea_replica(seed0,N,Nplaqs,Ncoppie,temp,sys0,d_sys0,mc_step0,d_mc_step0);
  
#ifndef ONE_REPLICA
  crea_replica(seed1,N,Nplaqs,Ncoppie,temp,sys1,d_sys1,mc_step1,d_mc_step1);
  crea_replica(seed2,N,Nplaqs,Ncoppie,temp,sys2,d_sys2,mc_step2,d_mc_step2);
  crea_replica(seed3,N,Nplaqs,Ncoppie,temp,sys3,d_sys3,mc_step3,d_mc_step3);
#endif

  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys0[i],inter);
    printf("# REPLICA n° = 0 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }
  
  printf("\n\n");

#ifndef ONE_REPLICA
  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys1[i],inter);
    printf("# REPLICA n° = 1 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }
  
  printf("\n\n");
  
  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys2[i],inter);
    printf("# REPLICA n° = 2 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }

  printf("\n\n");
  
  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys3[i],inter);
    printf("# REPLICA n° = 3 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }
#endif


  //////////////////////////////////////////////////////////////////
  ///// GINO PAOLI IMPERO DELLA TRISTEZZA //////////////////////////
  //////////////////////////////////////////////////////////////////

  //exit(0);

  //  /////////////////////////////////////////////////////////////
  //  /// INITIALIZE PLAQUETTES ENERGY ////////////////////////////
  //  /////////////////////////////////////////////////////////////
  
  double * point_pl_ene0, * point_pl_ene1, * point_pl_ene2, * point_pl_ene3;
  
  printf("# \n# \n# \n# \n");

  initialize_energy_plaquettes(0,d_inter,sys0,d_sys0,clock0,d_clock0);
#ifndef ONE_REPLICA
  initialize_energy_plaquettes(1,d_inter,sys1,d_sys1,clock0,d_clock1);
  initialize_energy_plaquettes(2,d_inter,sys2,d_sys2,clock0,d_clock2);
  initialize_energy_plaquettes(3,d_inter,sys3,d_sys3,clock0,d_clock3);
#endif
  
  printf("# \n# \n# \n# \n");
  
  /////////////////////////////////////////////////////////////
  ///// MONTE CARLO UTILITIES ALLOCATION //////////////////////
  /////////////////////////////////////////////////////////////
  
  printf("\n# PRIMA DI INIZIALIZZARE VARIABILI MONTE CARLO \n\n");
  
  /////////////////////////////////////////////////////////////////////////////
  //////////////  ALLOCA I PUNTATORI ANCHE SUL DEVICE /////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  printf("\n# FINE PROVE PRELIMINARI \n\n");

  //exit(0);

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  printf("# \n# \n# \n# \n");
  
  for(int i=0;i<NPT;i++){
    printf("# REPLICA 0 - PT n° = %d : identification sys = %s \n",i,sys0[i]->identity);
#ifndef ONE_REPLICA
    printf("# REPLICA 1 - PT n° = %d : identification sys = %s \n",i,sys1[i]->identity);
    printf("# REPLICA 2 - PT n° = %d : identification sys = %s \n",i,sys2[i]->identity);
    printf("# REPLICA 3 - PT n° = %d : identification sys = %s \n",i,sys3[i]->identity);
#endif
    }



  ////////////////////////////////////////////////////////////////////////
  // -- FINE ALLOCAZIONE CONTATORI ED ENERGIA RUNNING TIME ------ ////////
  ////////////////////////////////////////////////////////////////////////  

  //////////////////////////////////////////////////////////////////////////
  //// FAI LA SIMULAZIONE MONTE CARLO DI UNA REPLICA ///////////////////////
  //////////////////////////////////////////////////////////////////////////

  printf("# Inizializzo Orologio \n");

  //open_rng(seed+2);
  
  for(int i_jump=0; i_jump<NTJUMPS; i_jump++){

    clock0->n_attemp_exchange[i_jump] = 0;
#ifndef ONE_REPLICA
    clock1->n_attemp_exchange[i_jump] = 0;
    clock2->n_attemp_exchange[i_jump] = 0;
    clock3->n_attemp_exchange[i_jump] = 0;
#endif

    clock0->acc_rate_exchange[i_jump] = 0;
#ifndef ONE_REPLICA
    clock1->acc_rate_exchange[i_jump] = 0;
    clock2->acc_rate_exchange[i_jump] = 0;
    clock3->acc_rate_exchange[i_jump] = 0;
#endif

  }


  printf("\n\n#Comincio la dinamica Monte Carlo \n");
  
  double time_MC_sweep=0;
  int random_seed;
  int random_seed2; 

  printf("\n\n\n\n",Nplaqs);
  printf("NPLAQS (used)= %d \n",Nplaqs);
  printf("\n\n\n\n",Nplaqs);

  printf("# nstep temp[1] acc[0-1] ene[1] temp[9] acc[8-9] ene[9] temp[17] acc[16-17] ene[17]  temp[25] acc[24-25] ene[25] temp[29] acc[28-29] ene[29] \n\n");

  double * point_pl_ene;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  //  for(int ind_iter=NITER_MIN_PRINT+10; ind_iter<NITERATIONS; ind_iter+=10){

  int k_spin;
  
  int ntot_overlaps=(int)(NITERATIONS-NITER_MIN_PRINT)/NITER_PRINT_CONF-1;
  double overlaps[NPT][6*ntot_overlaps];
  double link_overlaps[NPT][6*ntot_overlaps];
  
  printf("ntot_overlaps = %d \n",6*ntot_overlaps);

  int icount[NPT];
  for(int i=0;i<NPT;i++) icount[i]=0;

  for(int ind_iter=NITER_MIN_PRINT+NITER_PRINT_CONF; ind_iter<NITERATIONS; ind_iter+=NITER_PRINT_CONF){

  //for(int ind_iter=NITER_MIN_PRINT+NITER_PRINT_CONF; ind_iter<NITER_MIN_PRINT+2*NITER_PRINT_CONF; ind_iter+=NITER_PRINT_CONF){

    char nome_file0[60],nome_file1[60],nome_file2[60],nome_file3[60];
    
    sprintf(nome_file0,"config_nrep0_iter_%d.dat",ind_iter);
    sprintf(nome_file1,"config_nrep1_iter_%d.dat",ind_iter);
    sprintf(nome_file2,"config_nrep2_iter_%d.dat",ind_iter);
    sprintf(nome_file3,"config_nrep3_iter_%d.dat",ind_iter);

    
    ////////////////////////////////////////////////////////
    /////// LEGGI CONF REPLICA 0 ///////////////////////////
    ////////////////////////////////////////////////////////

    read_configuration(temp,N,sys0,nome_file0);
    read_configuration(temp,N,sys1,nome_file1);
    read_configuration(temp,N,sys2,nome_file2);
    read_configuration(temp,N,sys3,nome_file3);

    
    for(int itemp=0;itemp<NPT;itemp++){

      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys0,sys1);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys0,sys1,inter);
      icount[itemp]++;
      
      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys0,sys2);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys0,sys2,inter);
      icount[itemp]++;
      
      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys0,sys3);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys0,sys3,inter);
      icount[itemp]++;
      
      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys1,sys2);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys1,sys2,inter);
      icount[itemp]++;
      
      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys1,sys3);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys1,sys3,inter);
      icount[itemp]++;
      
      overlaps[itemp][icount[itemp]] = compute_overlap(N,itemp,sys2,sys3);
      link_overlaps[itemp][icount[itemp]] = compute_link_overlap(Nplaqs,itemp,sys2,sys3,inter);
      icount[itemp]++;
    
    }

  }

  int imax_overlap[NPT];
  for(int i=0;i<NPT;i++) imax_overlap[i]=icount[i];

  FILE * fout_over;
  char nomeout_overlaps[60];

  ///////////////////////////////////////////////////////////
  // PRINT FILE WITH PHASOR OVERLAP /////////////////////////
  ///////////////////////////////////////////////////////////
  
  for(int itemp=0;itemp<NPT;itemp++){

    sprintf(nomeout_overlaps,"overlaps_T_%d.dat",itemp);

    if((fout_over=fopen(nomeout_overlaps,"w"))==NULL){
      printf("impossibile aprire %s \n ",nomeout_overlaps);
      exit(1);
    }else{
      printf("ho aperto per scrittura %s \n",nomeout_overlaps);
    }
    
    for(int i=0;i<imax_overlap[itemp];i++)
      fprintf(fout_over,"%12.8e %12.8e \n",overlaps[itemp][i],link_overlaps[itemp][i]);

    fclose(fout_over);
    
  }
  //////////////////////////////////////////////////////////

  double q_min_spin=-1;
  double q_max_spin=1;

  double q_min_link=-0.05;
  double q_max_link=1;
  
  double dq_isto_spin=(q_max_spin-q_min_spin)/NBINS;
  double dq_isto_link=(q_max_link-q_min_link)/NBINS_LINK;

  double istodata[NPT][NBINS];
  double istodata_link[NPT][NBINS_LINK];

  for(int itemp=0;itemp<NPT;itemp++){
    
    for(int i=0;i<NBINS;i++) istodata[itemp][i]=0;
    for(int i=0;i<NBINS_LINK;i++) istodata_link[itemp][i]=0;
    
    for(int i=0;i<imax_overlap[itemp];i++){
      
      int ibin; 

      ibin = (int)floor((overlaps[itemp][i]-q_min_spin)/dq_isto_spin);
      istodata[itemp][ibin]+=1./imax_overlap[itemp];
      
      ibin = (int)floor((link_overlaps[itemp][i]-q_min_link)/dq_isto_link);
      istodata_link[itemp][ibin]+=1./imax_overlap[itemp];
      
    }
  
    FILE * fisto;
    char nomeisto[60];
    
    // PRINT ISTOGRAM OF SPIN OVERLAP 
    
    sprintf(nomeisto,"overlap_spin_isto_T%d.dat",itemp);
    
    if((fisto=fopen(nomeisto,"w"))==NULL){
      printf("impossibile aprire %s \n ",nomeisto);
      exit(1);
    }else{
      printf("ho aperto per scrittura %s \n",nomeisto);
    }
    
    for(int i=0;i<NBINS;i++)
      fprintf(fisto,"%g %12.8e \n",q_min_spin+(i+0.5)*dq_isto_spin,istodata[itemp][i]);
    
    fclose(fisto);
    
    
    // PRINT ISTOGRAM OF LINK OVERLAP 
    
    sprintf(nomeisto,"overlap_link_isto_T%d.dat",itemp);
    
    if((fisto=fopen(nomeisto,"w"))==NULL){
      printf("impossibile aprire %s \n ",nomeisto);
      exit(1);
    }else{
      printf("ho aperto per scrittura %s \n",nomeisto);
    }
    
    for(int i=0;i<NBINS_LINK;i++)
      fprintf(fisto,"%g %12.8e \n",q_min_link+(i+0.5)*dq_isto_link,istodata_link[itemp][i]);
    
    fclose(fisto);
  
  
  }   //// END CYCLE OVER TEMPERATURES

  
  fclose(myfile2);

  //////////////////////////
  
  //////////////////////////
  delete [] ws;
  delete [] wsindices;
  delete [] gain;

  //////////////////////////
 
  //// --- LIBERA NUOVI VETTORI ------- ////////////// 

  cudaDeviceReset();

   
  return 0;

  
}


