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
//#include <curand_kernel.h>
//#include <cuda_runtime.h>

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
//#include <thrust/execution_policy.h>

//preferences of the simulation:
///////////////////////////////////////////////////////////////// 
#define Size 128 // 64 // 128
//#define BLOCK_SIZE 512 

#define twopi 6.283185307179586
#define I_WANT_GAIN 0  // 1 gain active, 0 gain off // to change gain form see frequencyGeneration.cpp
#define _GainMax_ 1.e-10   // maximum value of the gain (only if I_WANT_GAIN is 1)
#define I_WANT_PT 0  // if 0 no PT, if 1 PT active  
#define EQUISPACEDTS 1 
#define PLAQ_NUMBER 131072 // 16384  // 2^14

#define NSTEP 20000
#define NTJUMPS 1
#define NTIMES_PROFILING 10
#define FREQ_ENABLE 2  // if 0 all the frequencies are equal, if 2 equispaced frequencies
#define GAMMA 1 // linewidth in discrete units of wsindices - for FREQ_ENABLE 1 change nf (it is sigma = 4 nf, so that the effective linewidth is proportional to GAMMA/nf )
//#define FULLY 1 // =1 initial tetrads are all N^4. 
#define NPT 1
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
#define N_DE_REDUCED  (PLAQ_NUMBER/(4*N_THREADS_1_BLOCK*N_THREADS_1_BLOCK)) //(PLAQ_NUMBER/(N_THREADS_1_BLOCK*8)) //(PLAQ_NUMBER/(N_THREADS_1_BLOCK*8*8))  // PAY ATTENTION THAT IS MUST BE " PLAQ_NUMBER = 2**n N_THREADS_1_BLOCK*8*8" for some n

#define NR 1  // number of replicas 
#define DISORDER 1
#define _Jvalue_ 0.     // coefficient of the average of the 4-body interaction for the disordered case
#define _sJvalue_ 1.    // coefficient of the sigma of the 4-body interaction for the disordered case

#define ALL_MEASURES 1    // if 0 it measures only the energy - to be implemented (the xs must be copied on the GPU....)
#define measure_M2 0   // if 1 M^2 is measures, otherwise Mx AND My  ( then there is one additional column for each temperature
#define BINARY 1  // 0 if overlapIFO print formatted, 1 for write in binary

//#define SIGNALCOMP 1  // measures of signal - comment to remove
#define epsilonSM 1. // vincolo sferico di ogni spin (cf. tesi di P. Rotondo, p. 66)
#define _Tref_ 0.40726429   // reference temperature for the gain  - note as multiply _GainMax_ by K is the same that divide _Tref_ by K**2

#define MCBS 32    //  must be MCBS <= sfib
#define LBS 32     // 64   


/* from Brent, Proc. Fifth Australian Supercomputer Conference (1992).
#define sfib 97
#define rfib 127
#define sfib 175
#define rfib 258
#define sfib 353
#define rfib 521
#define sfib 334
#define rfib 607
#define sfib 861
#define rfib 1279
#define sfib 1252
#define rfib 2281
#define sfib 2641
#define rfib 3217
#define sfib 3004
#define rfib 4423*/

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
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);		\
        fprintf(stderr, "code: %d, reason: %s\n", error,		\
                cudaGetErrorString(error));				\
        exit(1);							\
      }									\
  }



// #include "/home/fabrizio/physics/graph/codes/hashing.cpp"
// #include "/home/fabrizio/physics/graph/codes/graph.cpp"
// #include "/home/fabrizio/physics/graph/minimalCycleDetection/4-pletDetection/codes/generate4plets.cpp"  
// #include "/home/fabrizio/physics/graph/minimalCycleDetection/4-pletDetection/codes/graphOf4plets.cpp"

#include "SMrandomTetrads_structures.h" // giac 09-2017 
#include "hashing.cpp"
#include "graph.cpp"
#include "generate4plets.cpp"  // here is used a structure defined in "SMrandomTetrads_structures.h"
#include "graphOf4plets.cpp"

//#include <gsl/gsl_randist.h>
//#include <gsl/gsl_rng.h>

#include <gsl_randist.h>
#include <gsl_rng.h>


#include "SMrandomTetrads_GPU_functions_spacchetta.cu"
#include "tetrads.cpp"
#include "generateQuadsFully.cpp"
#include "parallelMCstep.cu"
#include "kernels_disorder.cpp"
#include "SMrandomTetrads_CPU_functions.h"


int main(int argc, char ** argv)
{
  ////////////////////////// ////////////////////////// //////////////////////////

#if EQUISPACEDTS
  if(argc < 5){
    printf("\nERROR: usage SMrandomTetrads.out seed seed2 Tmin Tmax \n\n");
    exit(1);
  }
#else
  if(argc < 4){
    printf("\nERROR: usage SMrandomTetrads.out seed seed2 Tc \n\n");
    exit(1);
  }
#endif

  if(I_WANT_GAIN != 0 && I_WANT_PT != 0){
    printf("\nERROR: don't use PT with gain !!!\n\n");
    exit(4);
  }

  FILE * myfile2 = fopen("about.txt","w");
  
  int seed  = atoi(argv[1]);
  int seed2 = atoi(argv[2]);
#if EQUISPACEDTS
  double Tmin = atof(argv[3]);
  double Tmax = atof(argv[4]);
#else
  double Tc = atof(argv[3]);
#endif
  int N = Size;
  
  int nPT = NPT;
  double * beta = (double *) malloc(nPT*sizeof(double));
  double * temp = (double *)calloc(NTJUMPS,sizeof(double));

  double dT=(Tmax-Tmin)/NTJUMPS;

  for(int i=0;i<NTJUMPS;i++)
    temp[i]=Tmin+(double)i*dT;

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  
  
  printf(" N_DE_REDUCED = %d \n", N_DE_REDUCED);

#include "tempsGeneration.cpp"

  ////////////////////////////////////////////////////
  /// --- CUDA UTILITIES -----------------------//////
  ////////////////////////////////////////////////////
  /// --- LEARN THE NUMBER OF THREADS PER CORE -----//
  ////////////////////////////////////////////////////

  int dev; //, driverVersion=0, runtimeVersion=0; 
  dev=0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\n#Device %d: \"%s\"\n", dev, deviceProp.name);
  //printf("Maximum number of threads per multiprocessor: %d\n\n", deviceProp.maxThreadsPerMultiProcessor);
  int n_threads=deviceProp.maxThreadsPerBlock;

  printf("#Maximum sizes of x dimension of a grid: %12.8e\n", (double)deviceProp.maxGridSize[0]);
  
  printf("#Maximum number of threads per block: %d\n\n", N_THREADS_1_BLOCK);

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
  /////////// ATTEMPTS TO USE STRUCTURES ////////////////////////////
  ///////////////////////////////////////////////////////////////////

  // int nrepliche=10;
  // int nspin=128;
  // double * spin_pointer;
  // Replica_type ** rep;
  // Replica_type ** dev_rep;
  

  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////// STRUCTURES INITIALIZATION TRIALS ////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////

  // dim3 blocco (nspin);
  // dim3 griglia (1);

  // rep = (Replica_type **)calloc(nrepliche,sizeof(Replica_type *));
  
  // for(int i=0;i<nrepliche;i++){
  //   rep[i] = (Replica_type *)calloc(1,sizeof(Replica_type));
  //   rep[i]->spin = (double *)calloc(nspin,sizeof(double));
  // }
  
  // Replica_type ** dev_rep_array;
  
  // dev_rep_array=(Replica_type **)calloc(nrepliche,sizeof(Replica_type));

  // for(int i=0;i<nrepliche;i++){
  
  //   Replica_type * dev_rep; 
  
  //   cudaMalloc((Replica_type **) &dev_rep,sizeof(Replica_type));
    
  //   double * temporary; 
    
  //   cudaMalloc((double **) &temporary,nspin*sizeof(double));
    
  //   cudaMemcpy(&(dev_rep->spin),&temporary,sizeof(double *),cudaMemcpyHostToDevice);
  //   cudaMemcpy(temporary,rep[0]->spin,nspin*sizeof(double),cudaMemcpyHostToDevice);

  //   dev_rep_array[i]=dev_rep;
    
  //   InitializeSpinsOrdered <<< griglia, blocco >>> (dev_rep_array[i]);
    
  //   cudaDeviceSynchronize();
    
  // }
  
  
  // double * spin_array;
  // double * spin_array_pointer;

  // spin_array = (double *)calloc(nspin,sizeof(double));

  // cudaMemcpy(&(spin_array_pointer),&(dev_rep_array[2]->spin),sizeof(double *),cudaMemcpyDeviceToHost);
  // cudaMemcpy(spin_array,spin_array_pointer,nspin*sizeof(double),cudaMemcpyDeviceToHost);
  
  // exit(0);
  
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////

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

  InitGraphStructure(seed,N,placchette,wsindices);
  
  /////////////////////////////////////////////////////
  // 2. INITIALIZATION OF SPINS ON HOST ///////////////
  ////////////////////////// //////////////////////////
  ////////////////////////// //////////////////////////

  Conf_type * sys;

  sys = (Conf_type *)calloc(1,sizeof(Conf_type));
  sys->xs = (spin_t *)calloc(N,sizeof(spin_t));
  sys->ys = (spin_t *)calloc(N,sizeof(spin_t));

  uniformInit(sys->xs,sqrt(epsilonSM)/sqrt(2.),N,seed);   
  uniformInit(sys->ys,sqrt(epsilonSM)/sqrt(2.),N,seed+1);

  // spin_t * dev_xs, * dev_ys; 
  //////////////////////////////////////////////////////////
  // 2.2 INITIALIZATION of INTERACTIONS on HOST ////////////
  //////////////////////////////////////////////////////////

  Int_type * inter;
  
  inter = (Int_type *)calloc(1,sizeof(Int_type));
  inter->J = (double *)calloc(Nplaqs,sizeof(double));
  inter->spin_index = (int *)calloc(Nplaqs,sizeof(int));

  for(int np=0; np<Nplaqs; np++){
    
    inter->J[np]=placchette[np].J;

    for(int ispin=0; ispin<4; ispin++) inter->spin_index[4*np+ispin]=placchette[np].spin_index[ispin];

  }
  
  ////////////////////////////////////////////////////////// 
  /// A. PUT ""CONFIGURATION"" ON THE DEVICE ///////////////////
  //////////////////////////////////////////////////////////

  Conf_type * d_sys;
 
  cudaMalloc((Conf_type **) &d_sys,sizeof(Conf_type));
  
  spin_t *point_xs, *point_ys;
  
  cudaMalloc((spin_t **) &(point_xs),N*sizeof(spin_t));
  cudaMalloc((spin_t **) &(point_ys),N*sizeof(spin_t));
  
  cudaMemcpy(&(d_sys->xs),&(point_ys),sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->ys),&(point_ys),sizeof(spin_t *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_xs,sys->xs,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ys,sys->ys,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  
  
  double *point_ene, *point_ene_new, *point_de;
  double *point_de_block, *point_de_reduced;
  double *zeri_ene,*zeri_ene_block,*zeri_ene_reduced;

  zeri_ene = (double *)calloc(Nplaqs,sizeof(double));
  zeri_ene_block = (double *)calloc(Nplaqs/N_THREADS_1_BLOCK,sizeof(double));
  zeri_ene_reduced = (double *)calloc(N_DE_REDUCED,sizeof(double));

  cudaMalloc((double **) &point_ene,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_ene_new,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_de,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_de_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double));
  cudaMalloc((double **) &point_de_reduced,N_DE_REDUCED*sizeof(double));

  cudaMemcpy(&(d_sys->pl_ene),&(point_ene),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_ene_new),&(point_ene_new),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_de),&(point_de),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_de_block),&(point_de_block),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_de_reduced),&(point_de_reduced),sizeof(double *),cudaMemcpyHostToDevice);

  cudaMemcpy(point_ene,zeri_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ene_new,zeri_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_de,zeri_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_de_block,zeri_ene_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_de_reduced,zeri_ene_reduced,N_DE_REDUCED*sizeof(double),cudaMemcpyHostToDevice);
  
  ///////////////////////////////////////////////////////////// 
  /// A. PUT "INTERACTIONS" ON THE DEVICE /////////////////////
  /////////////////////////////////////////////////////////////

  Int_type * d_inter;

  int *point_spin_index;
  double *point_J;

  cudaMalloc((int **) &(point_spin_index),4*Nplaqs*sizeof(int));
  cudaMalloc((double **) &(point_J),Nplaqs*sizeof(double));
  
  cudaMemcpy(&(d_inter->spin_index),&point_spin_index,sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_inter->J),&point_J,sizeof(double *),cudaMemcpyHostToDevice);

  cudaMemcpy(point_spin_index,inter->spin_index,4*Nplaqs*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(point_J,inter->J,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  
  printf("#Nplaqs = %d \n",Nplaqs);
  
  //////////    //initializing the J's (order, gaussian and \pm 1): //////////////
  
  // ----------- DEVICE DATA STRUCTURES -------------// 

  // int * pl_spin_index;
  // double * pl_ene;
  // double * pl_ene_new;
  // double * pl_J;
  // double * pl_de;
  // double * pl_de_block;
  // double * pl_de_reduced;
  
  // int * dev_pl_spin_index;
  // double * dev_pl_ene;
  // double * dev_pl_ene_new;
  // double * dev_pl_J;
  // double * dev_pl_de;
  // double *  dev_pl_de_block; 

  
  // double * dev_pl_de_reduced; /// ATTENZIONE !!!!!!!!!!!! 
  
  //copy_plaquettes_host_to_device(Nplaqs,d_placchette,placchette);
  
  //pl_spin_index = (int *)calloc(4*Nplaqs,sizeof(int));
  
  // pl_ene = (double *)calloc(Nplaqs,sizeof(double));
  // pl_ene_new = (double *)calloc(Nplaqs,sizeof(double));
  // pl_de = (double *)calloc(Nplaqs,sizeof(double));
  
  // pl_de_block = (double *)calloc(Nplaqs/N_THREADS_1_BLOCK,sizeof(double));
  
  // pl_J = (double *)calloc(Nplaqs,sizeof(double));
  
  // for(int i=0;i<Nplaqs;i++){
  //   pl_J[i]=placchette[i].J;
    
  //   for(int j=0;j<4;j++) 
  //     pl_spin_index[4*i+j]=placchette[i].spin_index[j]; 
  // }
  
  // cudaMalloc((int **) &dev_pl_spin_index,4*Nplaqs*sizeof(int));
  // cudaMalloc((double **) &dev_pl_ene,Nplaqs*sizeof(double));
  // cudaMalloc((double **) &dev_pl_ene_new,Nplaqs*sizeof(double));
  // cudaMalloc((double **) &dev_pl_de,Nplaqs*sizeof(double));
  // cudaMalloc((double **) &dev_pl_de_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double));
  // cudaMalloc((double **) &dev_pl_J,Nplaqs*sizeof(double));

  // cudaMemcpy(dev_pl_spin_index,pl_spin_index,4*Nplaqs*sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_pl_ene,pl_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_pl_ene_new,pl_ene_new,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_pl_de,pl_de,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_pl_de_block,pl_de_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_pl_J,pl_J,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);


  printf("Ho copiato le placchette e accoppiamenti sul device \n");

     
  // 4. CONFRONTO MISURA INIZIALE ENERGIE STRUTTURA DATI Giacomo E STRUTTURA DATI Fabrizio  
  // (2017)


  ///// --- DA CORREGGERE !!!!!!!!!!!!!!! ATTENZIONE ATTENZIONE ATTENZIONE ---------------/////////////
  
  double ene_tot=0;
  ene_tot = energyPT_disorder1replica_plaqs(sys,inter);
  printf("#ene iniziale NUOVA = %g \n",ene_tot);
  
  ///// --- DA CORREGGERE !!!!!!!!!!!!!!! ATTENZIONE ATTENZIONE ATTENZIONE ---------------/////////////
  
 

  // COPIO GLI SPINS SULLE GPU PER FARE LA DINAMICA MONTE CARLO! 

  //cudaMemGetInfo(&free,&total);

  //printf("Free memory (BEFORE) = %12.8e \nTotal memory (BEFORE) = %12.8e \n",(double)free,(double)total);

  // cudaMalloc((spin_t **) &dev_xs,N*sizeof(spin_t));
  // cudaMalloc((spin_t **) &dev_ys,N*sizeof(spin_t));

  // cudaMemcpy(dev_xs,xs,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_ys,ys,N*sizeof(spin_t),cudaMemcpyHostToDevice);

    
  /////////////////////////////////////////////////////////////////////////////
  /// VEDI COSA HA SCRITTO NELLE ENERGIE DU DEVICE ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  total_energy_parallel(d_sys,d_inter);
  thrust::device_ptr<double> d_ene_ptr (dev_pl_ene);
  double ene_tot_plaqs=thrust::reduce(d_ene_ptr,d_ene_ptr+Nplaqs);
  printf("ene_tot_plaqs (GPU) = %g \n",ene_tot_plaqs);

  // 4. MC CYCLE :
  ////////////////////////// ////////////////////////// //////////////////////////
  
  // int tsamp=TSAMP;
  // int nbEquil=NBEQUIL;
  // int nbMeas=NBMEAS;

  //printf("#Anticipated exit for TEST purposes \n");
  //exit(0);

  double * de_TOT;
  double zero=0;
  
  cudaMalloc((double **) &de_TOT,sizeof(double));
  

  ////////////////////////// ////////////////////////// //////////////////////////
  ////////////////////////// ////////////////////////// //////////////////////////
  double * nrg  = (double *) calloc(1,sizeof(double));
  double * betas  = (double *) calloc(1,sizeof(double));
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  
  (*nrg) = ene_tot_plaqs;
  (*betas) = 1./Tmin;

  // init RNG
  ///////////////////////////////////////////////////////////////////
  open_rng(seed+2);
  ///////////////////////////////////////////////////////////////////
  
  // -- tests prima di cominciare la dinamica monte carlo

  int acc_rate=0;
  int n_attemp=0; 

  double * prof_time;
  prof_time = (double *)calloc(NTIMES_PROFILING,sizeof(double));


  ///// --------------------------------------------------------/////
  ///////////////////////////////////////////////////////////////////
  ///// ---- ALLOCAZIONE VETTORI PER MOSSA MONTE CARLO SU GPU -----// 
  ///////////////////////////////////////////////////////////////////


  /////////////////////////////////////////////////////////////
  ///// MONTE CARLO UTILITIES ALLOCATION //////////////////////
  /////////////////////////////////////////////////////////////
  
  //int * dev_ind_1, * dev_ind_2;

  int Ncoppie = N/2;

  curandGenerator_t gen;
  // create pseudo-random number generator // 
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,12345);
  
  MC_type * mc_step;
  monte_carlo=(MC_step *)calloc(1,sizeof(MC_step));

  spin_t zeri_spin[N];
  for(int i=0; i<N; i++) zeri_spin[i]=0;

  double zeri_coppie[Ncoppie];
  for(int i=0; i<Ncoppie; i++) zeri_coppie[i]=0;

  double zero_singolo=0;
  
  MC_type * d_mc_step;
  cudaMalloc((MC_step **) &d_mc_step,sizeof(MC_step));

  spin_t * point_nx1, * point_nx2, * point_ny1, * point_ny2;
  float * point_alpha_rand, * point_phi1_rand, * point_phi2_rand;
  int * point_Ncoppie, * point_icoppia, * point_coppie, * point_Nplaqs;

  int * coppie;
  coppie = (int *)calloc(N,sizeof(int));
  
  cudaMalloc((spin_t **) &point_nx1,(N/2)*sizeof(spin_t));
  cudaMalloc((spin_t **) &point_nx2,(N/2)*sizeof(spin_t));
  
  cudaMalloc((spin_t **) &point_ny1,(N/2)*sizeof(spin_t));
  cudaMalloc((spin_t **) &point_ny2,(N/2)*sizeof(spin_t));

  cudaMemcpy(&(d_mc_step->nx1),&point_nx1,sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->nx2),&point_nx2,sizeof(spin_t *),cudaMemcpyHostToDevice);

  cudaMemcpy(&(d_mc_step->ny1),&point_ny1,sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->ny2),&point_ny2,sizeof(spin_t *),cudaMemcpyHostToDevice);

  cudaMemcpy(point_nx1,zeri_spin,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_nx2,zeri_spin,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  	              
  cudaMemcpy(point_ny1,zeri_spin,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ny2,zeri_spin,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  cudaMalloc((float **) &point_alpha_rand,Ncoppie*sizeof(float));
  cudaMalloc((float **) &point_phi1_rand,Ncoppie*sizeof(float));
  cudaMalloc((float **) &point_phi2_rand,Ncoppie*sizeof(float));
  
  cudaMemcpy(&(d_mc_step->alpha_rand),&point_alpha_rand,sizeof(float *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->phi1_rand),&point_phi1_rand,sizeof(float *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->phi2_rand),&point_phi2_rand,sizeof(float *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_alpha_rand,zeri_coppie,Ncoppie*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(point_phi1_rand ,zeri_coppie,Ncoppie*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(point_phi2_rand ,zeri_coppie,Ncoppie*sizeof(float),cudaMemcpyHostToDevice);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc((int **) &point_Ncoppie,sizeof(int));
  cudaMalloc((int **) &point_Nplaqs,sizeof(int));
  cudaMalloc((int **) &point_icoppia,sizeof(int));
  
  cudaMemcpy(&(d_mc_step->Ncoppie),&point_Ncoppie,sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->Nplaqs), &point_Nplaqs,sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->icoppia),&point_icoppia,sizeof(int *),cudaMemcpyHostToDevice);
  

  cudaMemcpy(point_Ncoppie,zero_singolo,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(point_Nplaqs,&Nplaqs,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(point_icoppia,zero_singolo,sizeof(int),cudaMemcpyHostToDevice);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
    
  cudaMalloc((int **) &point_coppie,N*sizeof(int));
  cudaMemcpy(&(d_mc_step->coppie),&point_coppie,sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(point_coppie,zero_coppie,N*sizeof(int),cudaMemcpyHostToDevice);

  
  //////////////////////////////////////////////////////////////////////////////
  ////// YOU ARRIVED UP TO HERE !!! ------- //////////////////////////////////// 
  ////// ---------------------------------------------------------- ////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// ---- PLEASE RECHECK EVERYTHING DOWN THERE!!!! ------------ ////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  
  // -------- Allocazione memoria array di coppie (ovvero tutte le possibili coppie di spin che possono interagire) 
  //###############################################################################################################
  
    
  for(int i=0; i<N; i++) coppie[i]=i;
  
  //for(int i=0; i<N; i++) printf("coppie[%d] = %d \n",i,coppie[i]);
  
  for(int i=N; i>1; i--){
    int i_rand  = (int)floor((double)i*rand_double());
    int temp    = coppie[i-1];
    coppie[i-1] = coppie[i_rand];
    coppie[i_rand]=temp;
  }
  
    ///////////////////////////////////////////////////////////////////
  // MONTE CARLO DYNAMICS ///////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////

  double time_MC_sweep=0,time_MC_sweep_START=0;

  printf("\n\n#Comincio la dinamica Monte Carlo \n");

  FILE *fout = fopen("output.dat","w");

  fprintf(fout," \n\n");
  fprintf(fout,"#T   STEP   ENE   MX   MY   ACC   TIME_0...(gen rand)    TIME_1...(create update)   TIME_2...(de[iplaq])     TIME_3...(sum de)    TIME_4...(acc update)   TIME_SWEEP \n\n");

  fclose(fout);

  printf(" \n\n");
  printf("#T   STEP   ENE   MX   MY   ACC   TIME_0...(gen rand)     TIME_1...(create update)   TIME_2...(de[iplaq])    TIME_3...(sum de)    TIME_4...(acc update)   TIME_SWEEP \n\n");

  
  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  
  dim3 block (N_THREADS_1_BLOCK);
    
  dim3 grid ((Nplaqs+block.x-1)/block.x);
  

  
  printf("N_THREADS_1_BLOCK = %d \n",N_THREADS_1_BLOCK);

  
  for(int ntemp=NTJUMPS-1;ntemp>-1;ntemp--){
    
    double temperatura=temp[ntemp];
    
    for(int i_montecarlo=0;i_montecarlo<NSTEP;i_montecarlo++){
      
      // // printf(" MC STEP = %d \n",i_montecarlo);

      // // ---------- RESHUFFLE ORDER of COPPIE ----------------- // 
      
      time_MC_sweep_START = cpuSecond();

      for(int i=N; i>1; i--){
      	int i_rand  = (int)floor((double)i*rand_double());
      	int temp    = coppie[i-1];
      	coppie[i-1] = coppie[i_rand];
      	coppie[i_rand]=temp;
      }
  
      cudaMemcpy(dev_coppie,coppie,N*sizeof(int),cudaMemcpyHostToDevice);
      //cudaMemcpy(de_TOT,&zero,sizeof(double),cudaMemcpyHostToDevice);
      
      MCstep_disorderp4_placchette(&gen,
				   dev_Ncoppie,
				   dev_icoppia,
				   dev_Nplaqs,
				   dev_alpha_rand,
				   dev_phi1_rand,
				   dev_phi2_rand,
				   nrg,
				   prof_time,
				   &acc_rate,
				   &n_attemp,
				   temperatura,
				   Nplaqs,
				   N,
				   gain,
				   xs,
				   ys,
				   dev_xs,
				   dev_ys,
				   dev_pl_spin_index,
				   dev_pl_ene,
				   dev_pl_de,
				   dev_pl_de_block,
				   dev_pl_de_reduced,
				   dev_pl_ene_new,
				   dev_pl_J,
				   dev_coppie,
				   dev_nx1,
				   dev_nx2,
				   dev_ny1,
				   dev_ny2,
				   de_TOT);

      time_MC_sweep+=(cpuSecond()-time_MC_sweep_START);

      if(i_montecarlo%100==0){

	cudaMemcpy(xs,dev_xs,N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(ys,dev_ys,N*sizeof(double),cudaMemcpyDeviceToHost);

	double magn_x=0;
	double magn_y=0;

	for(int i_spin=0;i_spin<N;i_spin++){
	  magn_x+=(double)xs[i_spin];
	  magn_y+=(double)ys[i_spin];
	}
	
	//thrust::device_ptr<double> d_ene_ptr (dev_pl_ene);
	//double ene_tot_plaqs=thrust::reduce(d_ene_ptr,d_ene_ptr+Nplaqs);
	
	fout = fopen("output.dat","a");

	fprintf(fout,"%g %d %10.4e %10.4e %10.4e %g %8.4e %8.4e %8.4e %8.4e %8.4e %8.4e \n",temperatura, // 1 
		i_montecarlo, // 2 
		(*nrg)/N,     // 3
		magn_x/N,     // 4 
		magn_y/N,     // 5
		(double)acc_rate/n_attemp, //  6 
		prof_time[0]/n_attemp,     //  7 
		prof_time[1]/n_attemp,     //  8
		prof_time[2]/n_attemp,     //  9
		prof_time[3]/n_attemp,     //  10
		prof_time[4]/n_attemp,     //  11
		time_MC_sweep/i_montecarlo); // 12

	fclose(fout);

	printf("%g %d %10.4e %10.4e %10.4e %g %8.4e %8.4e %8.4e %8.4e %8.4e %8.4e \n",temperatura,i_montecarlo,(*nrg)/N,magn_x/N,magn_y/N,(double)acc_rate/n_attemp,prof_time[0]/n_attemp,prof_time[1]/n_attemp,prof_time[2]/n_attemp,prof_time[3]/n_attemp,prof_time[4]/n_attemp,time_MC_sweep/i_montecarlo);

      }
      
    }
    
  }

  curandDestroyGenerator(gen);
  
  cudaFree(dev_Nplaqs);
  
  cudaFree(dev_alpha_rand);
  cudaFree(dev_phi1_rand);
  cudaFree(dev_phi2_rand);
  cudaFree(dev_xs);
  cudaFree(dev_ys);
  cudaFree(dev_pl_spin_index);
  cudaFree(dev_pl_ene);
  cudaFree(dev_pl_de);
  cudaFree(dev_pl_ene_new);
  cudaFree(dev_pl_J);
  cudaFree(dev_coppie);

  //fclose(fout);

  close_rng();
  
  fclose(myfile2);
  //////////////////////////
  
  
// #if ALL_MEASURES  
// #if measure_M2
//   averages(nPT,3,beta);
//   histograms(nPT,3,-2.*N,0.,28,0,beta); //energy
//   histograms(nPT,3,0.5,1.,28,1,beta);   //radius
//   histograms(nPT,3,0.,1.,29,2,beta);    //magnetization |M|^2
// #else
//   averages(nPT,4,beta);
//   histograms(nPT,4,-2.*N,0.,28,0,beta); //energy
//   histograms(nPT,4,0.5,1.,28,1,beta);   //radius
//   histograms(nPT,4,-1.,1.,29,2,beta);   //magnetization Mx
//   histograms(nPT,4,-1.,1.,29,3,beta);   //magnetization My
// #endif
// #else
//   averages(nPT,1,beta);
//   histograms(nPT,1,-2.*N,0.,28,0,beta); //energy
// #endif

// #if NR>=2
//   histogramsOverlap2(nPT,1.,-1.,1.,29,0,beta);
//   histogramsOverlap2(nPT,1.,-1.,1.,29,1,beta);
//   histogramsOverlap2(nPT,1.,-1.,1.,29,2,beta);
//   binderPar(nPT,beta);
//   histogramsOverlapIFO_singleConf(nPT,Size,-1.,1.,59,beta);  // 39
//   //  binderParIFO(nPT,beta);  // versione beta
// #endif



  //  libero memoria:
  //////////////////////////
  
  // free(&beta);
  // free(&xs);
  // free(&ys);
  // free(&coppie);
 
 //////////////////////////
  //libero memoria relativa alla lista di quadruplette:
  //delete [] cumNbq;
  //delete [] quads;
  //////////////////////////
  delete [] ws;
  delete [] wsindices;
  delete [] gain;
  //////////////////////////
 
  //spins_vec.clear();
 
  //printf("%d\n",cumNbq[N]/4);
  
  
  //// --- LIBERA NUOVI VETTORI ------- ////////////// 

  cudaDeviceReset();

   
  return 0;

  
}


