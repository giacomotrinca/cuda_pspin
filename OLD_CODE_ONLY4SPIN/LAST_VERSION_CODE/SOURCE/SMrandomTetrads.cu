// -*- c++ -*-

using namespace std;   //Using the standard library namespace.

// -----------STANDARD LIBRARIES ----- ////// 
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>  //Needed for "atoi()" etc.
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

//////////////////////////////////////////////////////////////////
// Preferences of the simulations:
// N (spin) ---> Nplaq (plaquettes) ---
// 18            2^8  = 256
// 32            2^11 = 2048             
// 48            2^13 = 8192             
// 62            2^14 = 16384            
// 76            2^15 = 32768            
// 96            2^16 = 65536            
// 120           2^17 = 131072           
// 150           2^18 = 262144           
// 190           2^19 = 524288           
// 236           2^20 = 1048576          

///////////////////////////////////////////////////////////////// 
//Parameters to be checked before run:

#define Size 120
#define PLAQ_NUMBER 131072                //Power of two closest to the real number of plaquettes after dilution
#define TWO_REPLICA                       //With this  this settings it performs the simulations with only two replicas of the system with the same disorder (real-replicas)
#define NREPLICAS 4   
#define NITERATIONS (2*1048576/NSTEP)     //2^19, 2^20 or 2^21 for thermalization depending on the size (Number of PTsteps = MCsteps/NSTEP)
#define NSTEP 64                          //Number of MCsteps after which a PTstep is made
#define NITER_PRINT_CONF 4                //Number of PTsteps after which a config is printed
#define NITER_MIN_PRINT (2*262144/NSTEP)  //2^18 or 2^19 (anyway 4 times less than NITERATIONS)
#define NTJUMPS 35
#define NPT (NTJUMPS+1)                   //Number of replicas at different T (PT-replicas)
#define USE_REDUCE_UNROLL                 //To be used ONLY IF N > 76


/////////////////////////////////////////////////////////////////
//Other parameters:
//#define BLOCK_SIZE 512 
// #define SINGLE_REPLICA_OUTPUT
#define REPLICA_EXCHANGE 
#define NCHAR_IDENTITY 8
#define twopi 6.283185307179586
#define I_WANT_GAIN 0  // 1 gain active, 0 gain off // to change gain form see frequencyGeneration.cpp
#define _GainMax_ 1.e-10   // maximum value of the gain (only if I_WANT_GAIN is 1)
#define I_WANT_PT 0  // if 0 no PT, if 1 PT active  
#define EQUISPACEDTS 1 

#define CYCLE_UPDATE_ENERGY
#define PRINT_FREQUENCY 64
#define NTIMES_PROFILING 10
#define FREQ_ENABLE 2  // if 0 all the frequencies are equal, if 2 equispaced frequencies
#define GAMMA 1 // linewidth in discrete units of wsindices - for FREQ_ENABLE 1 change nf (it is sigma = 4 nf, so that the effective linewidth is proportional to GAMMA/nf )
//#define FULLY 1 // =1 initial tetrads are all N^4. 
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

//////////////////////////////////////////////////////////////////////
//Defines for random generator
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


////////////////////////////////////////////////////////////////////////
//////////////////// --- MAIN ---///////////////////////////////////////
///////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
  
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

  ///////////////////////////////////////////////////////////////////////////////
  /////////// 1. CHOOSE EQUISPACED TEMPERATURES IN T OR BETA  //////////////////
  //////////////////////////////////////////////////////////////////////////////
  
  int nPT = NPT;
  //  double * beta = (double *)calloc(nPT,sizeof(double));
  double * temp = (double *)calloc(nPT,sizeof(double));

  double dT=(Tmax-Tmin)/nPT;

  for(int i=0;i<NPT;i++){
  temp[i]=Tmax-(double)i*dT;
  }
  
  //double beta_max = 1./Tmin; // N = 128 ==> Tmin = 0.775
  //double beta_min = 1./Tmax; // N = 128 ==> Tmax = 1.300

  //double delta_beta = (beta_max-beta_min)/NTJUMPS;
  
  //for(int i=0;i<NPT;i++){
  // temp[i]=1./(beta_min+(double)i*delta_beta);
  //}
  

  //------------------------------------------------------------//
  // TEMPERATURE INDEX INCREASES AS THE TEMPERATURE IS LOWERED //
  //-----------------------------------------------------------//
      

  printf("# N_DE_REDUCED = %d \n", N_DE_REDUCED);
  
  //#include "tempsGeneration.cpp"
  
  ////////////////////////////////////////////////////
  //////// ---------- CUDA UTILITIES ----------////////
  ////////////////////////////////////////////////////
  /////  2. LEARN THE NUMBER OF THREADS PER CORE /////
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
  // ////////////////////////////////////////////////

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
  ////// 3. INIZIALIZATION OF THE INTERACTION NETWORK ////////////////////////////
  //////////////////////////////////////////////////////////////////////////////// 

  InitGraphStructure(seed_int,N,placchette,wsindices);

  
  //////////////////////////////////////////////////////////
  /////  4. INITIALIZATION of INTERACTIONS on HOST //////////
  //////////////////////////////////////////////////////////
  
  Int_type * inter;
  inter = (Int_type *)calloc(1,sizeof(Int_type));

  inter->J = (double *)calloc(Nplaqs,sizeof(double));
  inter->spin_index = (int *)calloc(4*Nplaqs,sizeof(int));
  inter->Nplaqs = Nplaqs;
  
  init_Interactions_host(Nplaqs,inter,placchette);
  
  //printf("TUTTO OK con le INTERAZIONI \n");

  /////////////////////////////////////////////////////////////
  /////////////////// ---- DISTRUGGI LE PLACCHETTE ----- ///// 
  free(placchette); //////////////////////////////////////////
  ////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////// 
  ////// 5. PUT "INTERACTIONS" ON THE DEVICE /////////////////////
  ///////////////////////////////////////////////////////////////
  
  Int_type * d_inter;
  cudaMalloc((Int_type **) &d_inter,sizeof(Int_type));
  
  init_Inter_device(Nplaqs,inter,d_inter);
  
  printf("# Ho copiato le placchette e accoppiamenti sul device \n");

  FILE * fout_inter = fopen("interactions_file.dat","w");
  for(int i=0; i<Nplaqs; i++)
    fprintf(fout_inter,"%d %d %d %d %12.8e \n",inter->spin_index[i*4],inter->spin_index[i*4+1],inter->spin_index[i*4+2],inter->spin_index[i*4+3],inter->J[i]);
  fclose(fout_inter);

  ///////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////////////////////
  // -- 6. ALLOCAZIONE CONTATORI ED ENERGIA RUNNING TIME ------ /////
  ///////////////////////////////////////////////////////////////////

  Clock_type * clock0, * clock1, * clock2, * clock3;
  Clock_type * d_clock0, * d_clock1, * d_clock2, * d_clock3;

  clock0 = (Clock_type *)calloc(1,sizeof(Clock_type));
  clock1 = (Clock_type *)calloc(1,sizeof(Clock_type));
#ifndef TWO_REPLICA
  clock2 = (Clock_type *)calloc(1,sizeof(Clock_type));
  clock3 = (Clock_type *)calloc(1,sizeof(Clock_type));
#endif

  cudaMalloc((Clock_type **) &d_clock0,sizeof(Clock_type));
  cudaMalloc((Clock_type **) &d_clock1,sizeof(Clock_type));
#ifndef TWO_REPLICA
  cudaMalloc((Clock_type **) &d_clock2,sizeof(Clock_type));
  cudaMalloc((Clock_type **) &d_clock3,sizeof(Clock_type));
#endif

  Initialize_counters(clock0,d_clock0);
  Initialize_counters(clock1,d_clock1);
#ifndef TWO_REPLICA
  Initialize_counters(clock2,d_clock2);
  Initialize_counters(clock3,d_clock3);
#endif

  /////////////////////////////////////////////////////////////////
  /// 7. CREAZIONE DELLE REPLICHE /////////////////////////////////
  /////////////////////////////////////////////////////////////////

  Conf_type ** sys0, ** sys1, ** sys2, ** sys3;
  Conf_type ** d_sys0, ** d_sys1, ** d_sys2, ** d_sys3; 
  
  sys0 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  sys1 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#ifndef TWO_REPLICA
  sys2 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  sys3 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#endif

  d_sys0 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  d_sys1 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#ifndef TWO_REPLICA
  d_sys2 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
  d_sys3 = (Conf_type **)calloc(NPT,sizeof(Conf_type *));
#endif

  int Ncoppie = N/2;
  
  MC_type ** d_mc_step0, ** d_mc_step1, ** d_mc_step2, ** d_mc_step3;
  MC_type ** mc_step0, ** mc_step1, ** mc_step2, ** mc_step3; 

  d_mc_step0 = (MC_type **)calloc(NPT,sizeof(MC_type *));
  d_mc_step1 = (MC_type **)calloc(NPT,sizeof(MC_type *));
#ifndef TWO_REPLICA
  d_mc_step2 = (MC_type **)calloc(NPT,sizeof(MC_type *));
  d_mc_step3 = (MC_type **)calloc(NPT,sizeof(MC_type *));
#endif

  mc_step0 = (MC_type **)calloc(NPT,sizeof(MC_type *));  
  mc_step1 = (MC_type **)calloc(NPT,sizeof(MC_type *));
#ifndef TWO_REPLICA  
  mc_step2 = (MC_type **)calloc(NPT,sizeof(MC_type *));  
  mc_step3 = (MC_type **)calloc(NPT,sizeof(MC_type *));    
#endif

  crea_replica(seed0,N,Nplaqs,Ncoppie,temp,sys0,d_sys0,mc_step0,d_mc_step0);
  crea_replica(seed1,N,Nplaqs,Ncoppie,temp,sys1,d_sys1,mc_step1,d_mc_step1);
#ifndef TWO_REPLICA
  crea_replica(seed2,N,Nplaqs,Ncoppie,temp,sys2,d_sys2,mc_step2,d_mc_step2);
  crea_replica(seed3,N,Nplaqs,Ncoppie,temp,sys3,d_sys3,mc_step3,d_mc_step3);
#endif

  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys0[i],inter);
    printf("# REPLICA n° = 0 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }
  
  printf("\n\n");


  for(int i=0;i<NPT;i++){
    double ene_tot = energyPT_disorder1replica_plaqs(sys1[i],inter);
    printf("# REPLICA n° = 1 PT n° = %d : ene iniziale HOST = %g \n",i,ene_tot);
  }
  
  printf("\n\n");

#ifndef TWO_REPLICA
  
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



  ///////////////////////////////////////////////////////////////
  ///// 8. INITIALIZE PLAQUETTES ENERGY /////////////////////////
  ///////////////////////////////////////////////////////////////
  
  double * point_pl_ene0, * point_pl_ene1, * point_pl_ene2, * point_pl_ene3;
  
  printf("# \n# \n# \n# \n");

  initialize_energy_plaquettes(0,d_inter,sys0,d_sys0,clock0,d_clock0);
  initialize_energy_plaquettes(1,d_inter,sys1,d_sys1,clock0,d_clock1);
#ifndef TWO_REPLICA
  initialize_energy_plaquettes(2,d_inter,sys2,d_sys2,clock0,d_clock2);
  initialize_energy_plaquettes(3,d_inter,sys3,d_sys3,clock0,d_clock3);
#endif
  
  printf("# \n# \n# \n# \n");
  
  /////////////////////////////////////////////////////////////
  /////9.  MONTE CARLO UTILITIES ALLOCATION ///////////////////
  ////////////////////////////////////////////////////////////
  
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
    printf("# REPLICA 1 - PT n° = %d : identification sys = %s \n",i,sys1[i]->identity);
#ifndef TWO_REPLICA
    printf("# REPLICA 2 - PT n° = %d : identification sys = %s \n",i,sys2[i]->identity);
    printf("# REPLICA 3 - PT n° = %d : identification sys = %s \n",i,sys3[i]->identity);
#endif
    }

  ////////////////////////////////////////////////////////////////////////
  // -- FINE ALLOCAZIONE CONTATORI ED ENERGIA RUNNING TIME ------ ////////
  ////////////////////////////////////////////////////////////////////////  
  
  //########################################################################################################################################
  //########################################################################################################################################

  ////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////
  //  // // START of MONTE CARLO DYNAMICS //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
 

  //////////////////////////////////////////////////////////////////////////
  ////  CUDA RANDOM GENERATOR INITIALIZATION////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  // curandGenerator_t gen;
  
  // curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  // curandSetPseudoRandomGeneratorSeed(gen,12345);

 

  printf("# Inizializzo Orologio \n");

  //open_rng(seed+2);
  
  for(int i_jump=0; i_jump<NTJUMPS; i_jump++){

    clock0->n_attemp_exchange[i_jump] = 0;
    clock1->n_attemp_exchange[i_jump] = 0;
#ifndef TWO_REPLICA
    clock2->n_attemp_exchange[i_jump] = 0;
    clock3->n_attemp_exchange[i_jump] = 0;
#endif

    clock0->acc_rate_exchange[i_jump] = 0;
    clock1->acc_rate_exchange[i_jump] = 0;
#ifndef TWO_REPLICA
    clock2->acc_rate_exchange[i_jump] = 0;
    clock3->acc_rate_exchange[i_jump] = 0;
#endif

  }


  printf("\n\n#Comincio la dinamica Monte Carlo \n");
  
  FILE *fout = fopen("output.dat","w");
  
  /*  FILE *fout_parallel0 = fopen("parallel_tempering0.dat","w");
  fclose(fout_parallel0);
  
  FILE *fout_parallel1 = fopen("parallel_tempering1.dat","w");
  fclose(fout_parallel1);

#ifndef TWO_REPLICA  
  FILE *fout_parallel2 = fopen("parallel_tempering2.dat","w");
  fclose(fout_parallel2);
  
  FILE *fout_parallel3 = fopen("parallel_tempering3.dat","w");
  fclose(fout_parallel3);
#endif */

  FILE *fout_parallel0 = fopen("parallel_tempering0.dat","w");
  fprintf(fout_parallel0, "#1:MC_STEP \t");
  for (int i=1; i<NPT; i++){
    fprintf(fout_parallel0, "%d:T[%d] \t %d:RATE[%d-%d] \t %d:ENE[%d] \t", 3*i-1, i, 3*i, i-1, i, 3*i+1, i);
  }
  fprintf(fout_parallel0, "\n");
  fclose(fout_parallel0);

  FILE *fout_parallel1 = fopen("parallel_tempering1.dat","w");
  fprintf(fout_parallel1, "#1:MC_STEP \t");
  for (int i=1; i<NPT; i++){
    fprintf(fout_parallel1, "%d:T[%d] \t %d:RATE[%d-%d] \t %d:ENE[%d] \t", 3*i-1, i, 3*i, i-1, i, 3*i+1, i);
  }
  fprintf(fout_parallel1, "\n");
  fclose(fout_parallel1);

#ifndef TWO_REPLICA
  FILE *fout_parallel2 = fopen("parallel_tempering2.dat","w");
  fprintf(fout_parallel2, "#1:MC_STEP \t");
  for (int i=1; i<NPT; i++){
    fprintf(fout_parallel2, "%d:T[%d] \t %d:RATE[%d-%d] \t %d:ENE[%d] \t", 3*i-1, i, 3*i, i-1, i, 3*i+1, i);
  }
  fprintf(fout_parallel2, "\n");
  fclose(fout_parallel2);

  FILE *fout_parallel3 = fopen("parallel_tempering3.dat","w");
  fprintf(fout_parallel3, "#1:MC_STEP \t");
  for (int i=1; i<NPT; i++){
    fprintf(fout_parallel3, "%d:T[%d] \t %d:RATE[%d-%d] \t %d:ENE[%d] \t", 3*i-1, i, 3*i, i-1, i, 3*i+1, i);
  }
  fprintf(fout_parallel3, "\n");
  fclose(fout_parallel3);
  #endif  
  
  //fclose(fout);
  
  double time_MC_sweep=0;
  int random_seed;
  int random_seed2; 

  printf("\n\n\n\n",Nplaqs);
  printf("NPLAQS (used)= %d \n",Nplaqs);
  printf("\n\n\n\n",Nplaqs);

  //  printf("# nstep temp[1] acc[0-1] ene[1] temp[9] acc[8-9] ene[9] temp[17] acc[16-17] ene[17]  temp[25] acc[24-25] ene[25] temp[29] acc[28-29] ene[29] \n\n");

  int s=NPT/2;
  int s1=s+1;
  int t=NPT/4;
  int t1=t+1;
  int q=NPT/4*3;
  int q1=q+1;
  int r=NPT-2;
  int r1=r+1;
  printf("# nstep temp[1] acc[0-1] ene[1] temp[%d] acc[%d-%d] ene[%d] temp[%d] acc[%d-%d] ene[%d]  temp[%d] acc[%d-%d] ene[%d] temp[%d] acc[%d-%d] ene[%d] \n\n", s1, s, s1, s1, t1, t, t1, t1, q1, q, q1, q1, r1, r, r1, r1); 

  
  double * point_pl_ene;


   /////////////////////////////////////////////////////////////////////////////////////////
  //// 10. FAI LA SIMULAZIONE MONTE CARLO DEL NUMERO DI REPLICHE SCELTO ///////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  
  for(int ind_iter=0; ind_iter<NITERATIONS; ind_iter++){
    
    trajectory_MonteCarlo(fout,ind_iter,
			  &time_MC_sweep,NPT,
			  seed0+ind_iter,
			  N,
			  Nplaqs,
			  d_clock0,clock0,
			  d_sys0,sys0,
			  d_inter,inter,
			  d_mc_step0,mc_step0);


    trajectory_MonteCarlo(fout,ind_iter,
			  &time_MC_sweep,NPT,
			  seed1+ind_iter,
			  N,
			  Nplaqs,
			  d_clock1,clock1,
			  d_sys1,sys1,
			  d_inter,inter,
			  d_mc_step1,mc_step1);

#ifndef TWO_REPLICA    
    trajectory_MonteCarlo(fout,ind_iter,
			  &time_MC_sweep,NPT,
			  seed2+ind_iter,
			  N,
			  Nplaqs,
			  d_clock2,clock2,
			  d_sys2,sys2,
			  d_inter,inter,
			  d_mc_step2,mc_step2);
    
    trajectory_MonteCarlo(fout,ind_iter,
			  &time_MC_sweep,NPT,
			  seed3+ind_iter,
			  N,
			  Nplaqs,
			  d_clock3,clock3,
			  d_sys3,sys3,
			  d_inter,inter,
			  d_mc_step3,mc_step3);
#endif
    
    double * point_double;
    
    exchange_replicas_Parallel_Tempering(seed0+ind_iter,temp,clock0,d_clock0,sys0,d_sys0,mc_step0,d_mc_step0);
    exchange_replicas_Parallel_Tempering(seed1+ind_iter,temp,clock1,d_clock1,sys1,d_sys1,mc_step1,d_mc_step1);
#ifndef TWO_REPLICA
    exchange_replicas_Parallel_Tempering(seed2+ind_iter,temp,clock2,d_clock2,sys2,d_sys2,mc_step2,d_mc_step2);
    exchange_replicas_Parallel_Tempering(seed3+ind_iter,temp,clock3,d_clock3,sys3,d_sys3,mc_step3,d_mc_step3);
#endif

    fout_parallel0 = fopen("parallel_tempering0.dat","a");
    fout_parallel1 = fopen("parallel_tempering1.dat","a");
#ifndef TWO_REPLICA
    fout_parallel2 = fopen("parallel_tempering2.dat","a");
    fout_parallel3 = fopen("parallel_tempering3.dat","a");
#endif
    
    print_replica(ind_iter,temp,N,clock0,fout_parallel0);
    print_replica(ind_iter,temp,N,clock1,fout_parallel1);
#ifndef TWO_REPLICA    
    print_replica(ind_iter,temp,N,clock2,fout_parallel2);
    print_replica(ind_iter,temp,N,clock3,fout_parallel3);
#endif

    if(ind_iter>NITER_MIN_PRINT && ind_iter%NITER_PRINT_CONF==0){

      print_configuration(0,temp,ind_iter,N,inter,sys0,d_sys0);
      print_configuration(1,temp,ind_iter,N,inter,sys1,d_sys1);
#ifndef TWO_REPLICA
      print_configuration(2,temp,ind_iter,N,inter,sys2,d_sys2);
      print_configuration(3,temp,ind_iter,N,inter,sys3,d_sys3);
#endif
    }
    
  }


  ////// FINAL OPERATIONS  //////////////////////////////////////////////
	   
  
  // curandDestroyGenerator(gen);
  
    
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


