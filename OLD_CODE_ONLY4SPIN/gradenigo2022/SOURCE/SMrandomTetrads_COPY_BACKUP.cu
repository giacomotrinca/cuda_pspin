// -*- c++ -*-

using namespace std;         // Using the standard library namespace.
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdlib.h> // needed for "atoi()" etc.
#include <cuda_runtime.h> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
//#include <thrust/execution_policy.h>

//preferences of the simulation:
///////////////////////////////////////////////////////////////// 
#define Size 64

#define twopi 6.283185307179586
#define I_WANT_GAIN 0  // 1 gain active, 0 gain off // to change gain form see frequencyGeneration.cpp
#define _GainMax_ 1.e-10   // maximum value of the gain (only if I_WANT_GAIN is 1)
#define I_WANT_PT 0  // if 0 no PT, if 1 PT active  
#define EQUISPACEDTS 1 

#define NSTEP 5000
#define NTJUMPS 1

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


//#include "SMrandomTetrads_GPU_functions.cu"
#include "SMrandomTetrads_GPU_functions_spacchetta.cu"
#include "tetrads.cpp"
#include "generateQuadsFully.cpp"
#include "SMrandomTetrads_gpu_data_transfer.h"
#include "parallelMCstep.cu"
#include "kernels_disorder.cpp"


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
  
#define N_THREADS_1_BLOCK n_threads

  printf("#Maximum number of threads per block: %d\n\n", N_THREADS_1_BLOCK);

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////

  // // generate random data serially
  // thrust::host_vector<int> h_vec(100,3);
  // //std::generate(h_vec.begin(), h_vec.end(), rand);
  
  // // transfer to device and compute sum
  // thrust::device_vector<int> d_vec = h_vec;
  // int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  // printf("Sum vec elements = %d\n\n",x);

  // ----- TEST SOMMA -------------------- //////////// 
  
  //int size_energia=123732145;

  printf("# Cerco di allocare la memoria per energia \n");

  //double * d_energia_aux;
  //double * d_energia;
  
  //cudaMalloc((double **) &d_energia_aux,size_energia*sizeof(double));
  //cudaMalloc((double **) &d_energia,size_energia*sizeof(double));
  
  printf("# Ho allocato la memoria per energia \n");
  
  //double * d_energia_aux;
  //d_energia_aux = (double *)calloc(size_energia,sizeof(double));

  //Sum_type energia;
  
  //energia.val = (double *)calloc(size_energia,sizeof(double));
  //for(int i=0; i<size_energia; i++) energia.val[i]=1;
  
  
  //double * data;
  
  //data = (double *)calloc(size_energia,sizeof(double));
  //for(int s=0; s<size_energia; s++) data[s] = 1;
  
  // for(int s=0; s<size_energia; s++){
  //   data[s] = (double *)calloc(1,sizeof(double));
  // }
  
  
  
  //cudaMalloc((double **) &(d_energia_aux),size_energia*sizeof(double));
  
  //cudaMemcpy(d_energia_aux,data,size_energia*sizeof(double),cudaMemcpyHostToDevice);
  //cudaMemcpy(d_energia,data,size_energia*sizeof(double),cudaMemcpyHostToDevice);
  
  //printf("#ARRAY size = %12.8e \n",(double)size_energia*sizeof(double));

  // double prova[10];

  // for(int i=0;i<10;i++){
  //   cudaMemcpy(&(prova[i]),&(d_energia_aux[i]),sizeof(double),cudaMemcpyDeviceToHost);
  //   printf("prova[%i] = %g \n",i,prova[i]);  
  // }

  //cudaMemcpy(&(prova[1]),&(d_energia_aux[1]),sizeof(double),cudaMemcpyDeviceToHost);
  //cudaMemcpy(&(prova[2]),&(d_energia_aux[2]),sizeof(double),cudaMemcpyDeviceToHost);
  //printf("prova[1] = %g \n",prova[1]);
  //printf("prova[2] = %g \n",prova[2]);

  //printf("PRIMA DI analisi preliminare somma logN \n");

  //logN_GPU_summation_prelim(size_energia,&energia,N_THREADS_1_BLOCK);
  
  //for(int i=0;i<energia.n_iterations;i++)    
  //printf("i = %d; x_parity = %d; x_oddity = %d; x_nblocks = %d \n",i, energia.parity[i], energia.oddity[i], energia.nblocks[i]);
  
  
  //printf("Ho realizzato analisi preliminare somma logN \n");

  //printf("x_nblocks[0] = %d \n",ene_nblocks[0]);

  //double somma = logN_GPU_summation_self(size_energia,d_energia_aux,n_threads);

  //double somma;

  //printf("size_energia    IN = %12.8e \n",(double)size_energia);
  //printf("somma energia  OUT = %12.8e \n",(double)somma);
  
  // generate random data serially
  //thrust::host_vector<double> h_energia_th(size_energia,1);
  //std::generate(h_vec.begin(), h_vec.end(), rand);
  
  // transfer to device and compute sum
  
  
  //thrust::device_ptr<double> d_energia_ptr (d_energia);
  //somma=thrust::reduce(d_energia_ptr,d_energia_ptr+size_energia);
  
  //try
  //{
  //thrust::device_ptr<double> d_energia_ptr (d_energia);
  //somma=thrust::reduce(d_energia_ptr,d_energia_ptr+size_energia);
  // }
  //catch(thrust::system_error &e)
  //{
  // output an error message and exit
  //std::cerr << e.what() << std::endl;
  //exit(-1);
  //}
  
  //printf("somma thurst = %12.8e \n",(double)somma);

  //cudaFree(d_energia_aux);

  //printf("Finisco prima per controlli \n\n");
  //exit(0);
  // ------------------------------------ /////////////


#if FULLY
  int   Nt=N*(N-1)*(N-2)*(N-3)/24;
  int * Listone;

  printf("Hai scelto l'opzione FULLY: O(N^4) plaquettes without FMC \n\n"); 
  
  printf("N*(N-1)*(N-2)*(N-3)/24 = %g \n",(double)Nt);

  if((Listone = (int *)calloc(Nt,sizeof(int)))==NULL){
    printf("ERROR: problem in memory allocation for O(N^4) array!!! \n");
    exit(1);
  }
  
  
#else
  //tetrads: unordered list of 4 nodes
  //4plet: each of the nonequivalent orderings of a tetrad which respects the frquency matching condition
  
#if FREQ_ENABLE == 0
  int Nt = 0.03*N*N*N;
  printf("#Nt = 0.03*N*N*N  = %d \n",Nt);
#else
  int Nt = 0.03*N*N*N*N;   // should be order N^4 to avoid PC in the disordered case, but if it is too big the program takes a lot of time to find the quads randomly
  printf("#Nt = 0.03*N*N*N*N  = %d \n",Nt);
#endif
  
  // The Number of Thetrads is forced to be a multiple of N_THREADS_1_BLOCK
  
  //printf("Nt (RECALCULATED): %d \n\n",Nt);

  //printf("Esco prima per dei controlli \n\n");
  //exit(0);
    
  vector<vector<int> > tetrads;
  
  
  printf("#procedo con generateTetradsHashing \n");
  generateTetradsHashing(N,Nt,tetrads,seed+1);	 // Qui costruisce le tetradi random, senza implementazione della FMC
  printf("#Ho concluso con generateTetradsHashing \n");

  //  generateTetrads(N,Nt,tetrads, seed+1);
#endif


  //////////////////////////////////////
  int *quads;
  int *cumNbq = new int [N+1];
  double *ws = new double [N];
  int *wsindices = new int [N];
  double * gain = new double [N];
  //  double gainMax = _GainMax_;    
  
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
  

#if(FULLY)
  vector<vector< vector< int > > > quadlist;
  vector<vector<int> > empty_vector;
  quadlist.assign(N, empty_vector);
  
  generateQuadsFully(N,wsindices,quadlist, cumNbq,1.,seed);
  //I construct quads[]:
  ///////////////////////
  int quadsDim = cumNbq[N];  
  quads = new int [ 4*quadsDim ];
  
  for(int i=0; i<N; i++)
    for(int k=0; k<quadlist[i].size(); k++){
      for(int ei=0;ei<4;ei++){
	quads[4*(cumNbq[i]+k)+ei] = quadlist[i][k][ei];
      }
    }
  ///////////////////////
  quadlist.clear();

#else

  printf("#Comincio a generare il grafo delle interazioni \n");

  vector<vector<int> > quadlist;
  
  ////////////////////////////////////////////////////////////////////////////
  /////////////////// INTRODUZIONE NUOVE STRUTTURE DATI //////////////////////
  ////////////////////////////////////////////////////////////////////////////

  ///////////////----CLASSE vector *SOLO* per l'inizializzazione della struttura dati
  ///////////////////////////////////////////////////////////////////////////////////

  vector<Plaqs_type> placchette_vec; //-----------------GIACOMO ------- 09/2017
  
  vector<Spin_type> spins_vec(N); // -------------------GIACOMO ------- 09/2017


  printf("#Number of tetrads = %d \n",(int)tetrads.size());

  // 1. CREA TUTTE LE PLACCHETTE A IMPLEMENTANDO FMC A PARTIRE DALLE TETRADI    (struttura dati GIACOMO)
  /// ------------------------------------------------------------------------
  ///----Inizializzazione viene fatta con vector, poi va rimosso per la dinamica Monte Carlo
  ///------------------------------------------------------------------------------------------- GIACOMO --------- 09/2017---////////////////////
  ///---------------------------------------------------------------------------------///////////////////////////////////////////////////////////
  generate4pletsCycles_discreteWs_wPlaqs(tetrads,placchette_vec,spins_vec,wsindices); ///////////////////////////////////////////////////////////
  ///---------------------------------------------------------------------------------///////////////////////////////////////////////////////////
  /// -------- questa deve essere una funzione void ----------------------------------///////////////////////////////////////////////////////////
  // questa funzione alloca pure per ciascuno spin la lista delle placchette ad esso attaccate------------------------------------------------//

  //--------- creo un array di oggetti Plaq_type che soddisfano a FMC
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  printf("#Number of plaquettes = %d \n",(int)placchette_vec.size());

  // 1.B CREA QUADS e DISTRUGGE TETRADI RANDOM (Struttura dati FABRIZIO) 
  // -------------------------------------- //////////////////////////// 

  generate4pletsCycles_discreteWs(N,tetrads,quadlist,cumNbq,wsindices); // qui seleziona le tetradi che corrispondono alle FMC, considerando pure eventuali permutazioni

  tetrads.clear(); // cancella le tetradi random perchè non gli servono più e occupano spazio


  //////////////// ---- FABRIZIO -------- trasferimento tetradi from vector CLASS (C++) to int * array 

  int quadsDim = cumNbq[N];  
  quads = new int [ 4*quadsDim ];
  for(int j=0;j<quadsDim;j++)
    for(int ei=0;ei<4;ei++)
      quads[4*j+ei]=quadlist[j][ei];     
  quadlist.clear();
#endif
  ///////////////// ---- END TRASFERIMENTO FABRIZIO ------------- //////////////////////////// 

  printf("#Ho generato il grafo delle interazioni \n");
  
  ///////////////////////////////////////////////////
  // KEY: to access the rth mode in the qth quadruplette containing the  m-th mode :
  // quads[ 4*(cumNbq[m]+q) + r ]
  ///////////////////////////////////////////////////
  
    
  // 2. ALLOCATION AND INITIALIZATION OF THE DEGREES OF FREDOM
  ////////////////////////// ////////////////////////// //////////////////////////
  ////////////////////////// ////////////////////////// //////////////////////////

  spin_t * xs,* ys;
  spin_t * d_xs, * d_ys; 

  size_t size = nPT*N*sizeof(spin_t);

  xs = (spin_t *) malloc(NR*size);
  ys = (spin_t *) malloc(NR*size);

  d_xs = (spin_t *) malloc(NR*size);
  d_ys = (spin_t *) malloc(NR*size);
  
  // 2.2 INITIALIZATION
  //////////////////////////

  uniformInit(xs,sqrt(epsilonSM)/sqrt(2.),NR*N*nPT,seed);   
  uniformInit(ys,sqrt(epsilonSM)/sqrt(2.),NR*N*nPT,seed+1);
  
  

  // 2.3 INITIALIZATION OF SPINS --------------------- GIACOMO ----------- 09/2017 
  // ----------------------------AND TOTAL ENERGY------------------------------
  /////////////////////////////////////////////////////////////////////////////

  double ene_tot=0;

  Spin_type spins[N];
  
  for(int j=0;j<N;j++){
    spins[j].x=xs[j];
    spins[j].y=ys[j];
  }

  // 2.4 INITIALIZATION OF COUPLINGS --------------------- GIACOMO --------- 09/2017
  ////////////////////////// ////////////////////////// //////////////////////////

  //   device allocation of disordered couplings:

  double * J = (double *) malloc((cumNbq[N])*sizeof(double));
  
  //////////    //initializing the J's (order, gaussian and \pm 1): //////////////
  if(DISORDER){
    
    double sigmaJ = sqrt(4. * _sJvalue_ * N / cumNbq[N] );
    double avgJ = 4.* _Jvalue_ * N / cumNbq[N];
    randomGaussianCouplingsp4_v2(cumNbq[N],quads,cumNbq,J,avgJ,sigmaJ,seed+2);  // the change with respect to the version V6 is all here
  }else{
    
    for(int j=0;j<cumNbq[N];j++)     // no-optimized ordered case
      J[j] = 4.* N / cumNbq[N];      // so we are sure that the energy is extensive (if there is no localization)
  }


  ////////// -----GIACOMO 2017 -------- TRANSFER OF J couplings from "quads" DATA STRUCTURE to "placchette_vec" DATA STRUCTURE
  ////////// --------------------------------------------------------------------------///////////////////////////////////////
  //-------------------------------------------------------------------------////////////////////////// 
  randomGaussianCouplingsp4_fromQUADStoPLAQS(placchette_vec,quads,cumNbq,J); // here is used vector data structure
  /////////////////////// 
  
  printf("# Ho inizializzato couplings \n");

  //////// -------------------------------------------------------//////////////////

  //////// --------------------- PRINT IN OUTPUT FILE ALL QUADRUPLETS 
  int contaquads = 0;
  
  FILE *fileq = fopen("quads.dat","w");
  for(int j=0;j<N;j++)
    {
      int d=cumNbq[j+1]-cumNbq[j];
      for(int m=0;m<d;m++)
	{
	  for(int h=0;h<4;h++)
	    {
	      int myq=quads[ 4*(cumNbq[j] +m) + h ];
	      fprintf(fileq," 4*(cumNbq[j] +m) + h = %d ; value = %d \n",4*(cumNbq[j] +m) + h, myq);
	    }
	  //fprintf(fileq,"\n");
	  contaquads++;
	}
    }
  fclose(fileq);
  
  printf("#num of quads = %d \n",contaquads);

  ///////////////////////-----------------------------GIACOMO-----------------07/2017------------------///
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 4. ALLOCATION and LINK of DIMERS (spin doublets with linked lists of plaquettes used for the MC step)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  int Nplaqs = placchette_vec.size();
  printf("#Nplaqs = %d \n",Nplaqs);

  // int Nplaqs_GPU = ((int)(Nplaqs/N_THREADS_1_BLOCK))*N_THREADS_1_BLOCK;
  
  // printf("#Nplaqs_GPU = %d \n",Nplaqs_GPU);
  
  // placchette_vec.erase (placchette_vec.end()-(Nplaqs-Nplaqs_GPU),placchette_vec.end());
  
  // printf("#NEW SIZE OF placchette_vec = %d \n",(int)placchette_vec.size());
  
  // generate random data serially
  //thrust::host_vector<double> h_DE_plaq(Nplaqs_GPU,0);
  //std::generate(h_vec.begin(), h_vec.end(), rand);
  
  // transfer to device and compute sum
  //thrust::device_vector<double> d_DE_plaq = h_DE_plaq;


  //int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  //printf("Sum vec elements = %d\n\n",x);

  

  int Ndimers = N*(N-1)/2 ;

  // -------- Allocazione memoria array di Dimers (ovvero tutte le possibili coppie di spin che possono interagire) 
  //###############################################################################################################

  Dimers_type ** coppie;
  Dimers_type ** d_coppie;

  coppie = (Dimers_type **)calloc(Ndimers,sizeof(Dimers_type *));
  d_coppie = (Dimers_type **)calloc(Ndimers,sizeof(Dimers_type *));
  
  for(int s=0; s<Ndimers; s++){
    coppie[s] = (Dimers_type *)calloc(1,sizeof(Dimers_type));
    d_coppie[s] = (Dimers_type *)calloc(1,sizeof(Dimers_type));
  }
  
  //link_quads_to_dimers(N,quads,coppie,cumNbq);

  //for(int s=0; s<Ndimers; s++)
  // printf("coppia %d: Nplaqs = %d attached to ij \n",s,(int)coppie[s]->ij_quadlist.size());


  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // 4.B ALLOCATION and LINK of PLAQUETTES to COPPIE (spin doublets with linked lists of plaquettes used for the MC step)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  build_dimers(N,Ndimers,coppie);

 
  //for(int s=0; s<Ndimers; s++)
  //printf("#coppia %d:  spin i = %d  spin j = %d \n",s,coppie[s]->spin_i_index,coppie[s]->spin_j_index);

  
  // 3. TRASFERIMENTO STRUTTURE DATI DA vector AD array C normali

  //  int Nplaqs = placchette_vec.size();
  
  Plaqs_type ** placchette; //// array di Plaqs_type che va a sostituire il vector
  Plaqs_type ** d_placchette; 
  
  placchette = (Plaqs_type **)calloc(Nplaqs,sizeof(Plaqs_type *));
  d_placchette = (Plaqs_type **)calloc(Nplaqs,sizeof(Plaqs_type *));

  for(int s=0; s<Nplaqs; s++){
    placchette[s]   = (Plaqs_type *)calloc(1,sizeof(Plaqs_type));
    d_placchette[s] = (Plaqs_type *)calloc(1,sizeof(Plaqs_type));
  }

  plaqs_from_vec_to_array(N,Nplaqs,placchette_vec,placchette); // copia vector <Plaqs_type> in Plaqs_type * array 

  placchette_vec.clear(); // distrugge il vector

  printf("# Ho copiato placchette su un * double e disallocato vector<double> \n\n");

  
  // double ** de_plaqs;
  // double ** d_de_plaqs;
  
  // de_plaqs = (double **)calloc(Nplaqs,sizeof(double *));
  // d_de_plaqs = (double **)calloc(Nplaqs,sizeof(double *));
  
  // for(int s=0; s<Nplaqs_GPU; s++){
  //   de_plaqs[s]   = (double *)calloc(1,sizeof(double));
  //   d_de_plaqs[s] = (double *)calloc(1,sizeof(double));
  // }

  // copy_de_plaqs_host_to_device(Nplaqs_GPU,d_de_plaqs,de_plaqs);

  // 3.a CUDA instructions 
  
  // cudaMalloc((Plaqs_type **) d_placchette,Nplaqs*sizeof(Plaqs_type *));
  
  // for(int s=0; s<Nplaqs; s++)
  //   cudaMalloc((Plaqs_type **) &(d_placchette[s]),sizeof(Plaqs_type));
  
  // cudaMemcpy(d_placchette,placchette,sizeof(placchette),cudaMemcpyHostToDevice);
  
  // for(int s=0; s<Nplaqs; s++){
    
  //   cudaMemcpy(&(d_placchette[s]->ene),&(placchette[s]->ene),sizeof(double),cudaMemcpyHostToDevice);
  //   cudaMemcpy(&(d_placchette[s]->ene_new),&(placchette[s]->ene_new),sizeof(double),cudaMemcpyHostToDevice);
  //   cudaMemcpy(&(d_placchette[s]->J),&(placchette[s]->J),sizeof(double),cudaMemcpyHostToDevice);
  //   cudaMemcpy(&(d_placchette[s]->flag),&(placchette[s]->flag),sizeof(int),cudaMemcpyHostToDevice);
    
  //   for(int i=0; i<4; i++){
  //     cudaMemcpy(&(d_placchette[s]->spin_index[i]),&(placchette[s]->spin_index[i]),sizeof(int),cudaMemcpyHostToDevice);
  //   }
  
  // }


  copy_plaquettes_host_to_device(Nplaqs,d_placchette,placchette);
  
  printf("Ho copiato le placchette sul device \n");

  // Plaqs_type * placchetta_prova; 
  // placchetta_prova = (Plaqs_type *)calloc(1,sizeof(Plaqs_type));
  
  // for(int i=0; i<4; i++){
  //   cudaMemcpy(&(placchetta_prova->spin_index[i]),&(d_placchette[13]->spin_index[i]),sizeof(int),cudaMemcpyDeviceToHost);
  // }
  
  // for(int i=0; i<4; i++){
  //   printf("Host  : spin[%d] = %d \n",i,placchette[13]->spin_index[i]);
  //   printf("Device: spin[%d] = %d \n\n",i,placchetta_prova->spin_index[i]);
  // }
  
  
   
  // 4. CONFRONTO MISURA INIZIALE ENERGIE STRUTTURA DATI Giacomo E STRUTTURA DATI Fabrizio  
  // (2017)
  
  printf("#Nplaqs before CPU energy measures = %d \n",Nplaqs);

  energyPT_disorder1replica(N,xs,ys,gain,cumNbq,quads,J,&ene_tot); 
  printf("#ene iniziale VECCHIA = %g \n",ene_tot);
  
  ene_tot = energyPT_disorder1replica_plaqs(N,Nplaqs,placchette,xs,ys,gain);
  printf("#ene iniziale NUOVA = %g \n",ene_tot);
  
  printf("#Nplaqs *after* CPU energy measures = %d \n",Nplaqs);

  //printf("\n\n\n ANTICIPATE EXIT FOR TRIAL PURPOSES \n\n");
  //exit(0);


  // COPIO GLI SPINS SULLE GPU PER FARE LA DINAMICA MONTE CARLO! 

  printf("#Nplaqs *before* COPY of the SPINS on DEVICE = %d \n",Nplaqs);

  copy_spins_host_to_device(N,&xs,&ys,&d_xs,&d_ys);  

  printf("#Nplaqs *after* COPY of the SPINS on DEVICE = %d \n",Nplaqs);

  double * d_ene;
  cudaMalloc((double **) &d_ene,Nplaqs*sizeof(double));

  printf("Nplaqs before passing to GPU routine = %d \n",Nplaqs);

  double ene_tot_plaqs;
  ene_tot_plaqs = total_energy_parallel(Nplaqs,d_placchette,d_xs,d_ys,d_ene);

  // 4. MC CYCLE :
  ////////////////////////// ////////////////////////// //////////////////////////
  
  // int tsamp=TSAMP;
  // int nbEquil=NBEQUIL;
  // int nbMeas=NBMEAS;

  // write_output(myfile2,N,tsamp,nbEquil,nbMeas,seed,beta[0],DISORDER,Nt,cumNbq[N]/4);
  
  // MCcyclep4Cycles(N,nPT,quads,cumNbq,xs,ys,gain,tsamp,nbEquil,nbMeas,beta,seed,myfile2,nbFreq,wsindices);  
  
  // cout << Nt << " " << " " <<  cumNbq[N]  << " " << endl;

  printf("#Anticipated exit for TEST purposes \n");
  exit(0);

  ////////////////////////// ////////////////////////// //////////////////////////
  ////////////////////////// ////////////////////////// //////////////////////////
  double * nrg  = (double *) calloc(1,sizeof(double));
  double * betas  = (double *) calloc(1,sizeof(double));
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////

  (*nrg) = ene_tot;
  (*betas) = 1./Tmin;


  // init RNG
  ///////////////////////////////////////////////////////////////////
  open_rng(seed+2);
  ///////////////////////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////////////////////
  // PREPARE GPUs FOR MC SWEEPS /////////////////////////////////////
  ///////////////////////////////////////////////////////////////////

  dim3 block (N_THREADS_1_BLOCK);
  dim3 grid ((Nplaqs+block.x-1)/block.x);

  printf("grid.x %d block.x %d \n",grid.x, block.x);
  
  int nthreads = (int)block.x;
  int nblocks = (int)grid.x;
 
  ///////////////////////////////////////////////////////////////////
  // MONTE CARLO DYNAMICS ///////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////

  printf("#Comincio la dinamica Monte Carlo \n");

  FILE *fout = fopen("output.dat","w");
  
  for(int ntemp=NTJUMPS-1;ntemp>-1;ntemp--){

    double temperatura=temp[ntemp];
    
    for(int i_montecarlo=0;i_montecarlo<NSTEP;i_montecarlo++){
      
      //MCstep_disorderp4_withAll(N,xs,ys,gain,cumNbq,quads,betas,J,nrg);
      
      //MCstep_disorderp4_placchette(temperatura,N,Ndimers,coppie,gain,placchette,xs,ys,nrg);
            
      //MCstep_disorderp4_placchette(temperatura,N,Ndimers,coppie,gain,placchette,xs,ys,nrg,d_de_plaqs);
      //MCstep_disorderp4_placchette(nthreads,nblocks,temperatura,N,Ndimers,coppie,gain,d_placchette,xs,ys,d_xs,d_ys,d_de_plaqs,nrg);

      if(i_montecarlo%100==0){
	//energyPT_disorder1replica(N,xs,ys,gain,cumNbq,quads,J,&nrg);
	//ene_tot = energyPT_disorder1replica_plaqs(N,placchette,spins,gain);
	
	double magn_x=0;
	double magn_y=0;

	for(int i_spin=0;i_spin<N;i_spin++){
	  magn_x+=xs[i_spin];
	  magn_y+=ys[i_spin];
	}

	fprintf(fout,"%g %d %g %10.4e %10.4e \n",temperatura,i_montecarlo,(*nrg)/N,magn_x/N,magn_y/N);
	printf("%g %d %g %10.4e %10.4e \n",temperatura,i_montecarlo,(*nrg)/N,magn_x/N,magn_y/N);

	// ene_tot=0;
	// for(int iplaq=0;iplaq<placchette.size();iplaq++)
	// 	ene_tot+=placchette[iplaq].ene;
	// printf("%d %g \n",i_montecarlo,ene_tot);
      }
      //printf("%d %g \n",i_montecarlo,ene_tot);
      
    }
    
  }
  
  fclose(fout);

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
  free(beta);
  free(xs);
  free(ys);
  //////////////////////////
  //libero memoria relativa alla lista di quadruplette:
  delete [] cumNbq;
  delete [] quads;
  //////////////////////////
  delete [] ws;
  delete [] wsindices;
  delete [] gain;
  //////////////////////////
  
  //printf("%d\n",cumNbq[N]/4);
  
  
  //// --- LIBERA NUOVI VETTORI ------- ////////////// 

  free(coppie);

  spins_vec.clear();
  
  return 0;

  
}


