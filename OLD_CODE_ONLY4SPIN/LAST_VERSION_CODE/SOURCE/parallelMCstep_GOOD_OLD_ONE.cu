// -*- c++ -*-

//////////////////////////////////////////////////////////////////////////////////////
////// ----------------------- MONTE CARLO STEP ---------- GIACOMO--------------- 2017 
//////////////////////////////////////////////////////////////////////////////////////

__global__ void the_trivial_stream (){

  int somma = threadIdx.x;
  
  somma = somma + 1;
  
}

double cpuSecond(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_sec*1.e-6);
}

__global__ void InitializeSpinsOrdered (Replica_type * dev_rep){

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  
  dev_rep->spin[idx]=idx;
  
  printf("dev_rep->spin[%d] = %g \n",idx,dev_rep->spin[idx]);
  
}


__global__ void ReadSpinsOrdered (Replica_type * dev_rep){
  
  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  
  printf("dev_rep->spin[%d] = %g \n",idx,dev_rep->spin[idx]);
  
}


__global__ void UpdateEnergy (Conf_type * d_sys){
  
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  
  d_sys->pl_ene[idx]=d_sys->pl_ene_new[idx];
  
}

__global__ void reduceUnrollWarps8 (double *g_idata, double *g_odata, unsigned int n) {
  
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x*blockDim.x*8 + threadIdx.x;
  
  // convert global data pointer to the local pointer of this block
  double *idata = g_idata + blockIdx.x*blockDim.x*8;
  
  // unrolling 8
  if (idx + 7*blockDim.x < n) {
    double a1 = g_idata[idx];
    double a2 = g_idata[idx+blockDim.x];
    double a3 = g_idata[idx+2*blockDim.x];
    double a4 = g_idata[idx+3*blockDim.x];
    double b1 = g_idata[idx+4*blockDim.x];
    double b2 = g_idata[idx+5*blockDim.x];
    double b3 = g_idata[idx+6*blockDim.x];
    double b4 = g_idata[idx+7*blockDim.x];
    g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
  }
  
  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x>=1024 && tid < 512 ) idata[tid] += idata[tid + 512];
  __syncthreads();
  
  if (blockDim.x>=512 && tid < 256 ) idata[tid] += idata[tid + 256];
  __syncthreads();
  
  if (blockDim.x>=256 && tid < 128 ) idata[tid] += idata[tid + 128];
  __syncthreads();
  
  if (blockDim.x>=128 && tid < 64 ) idata[tid] += idata[tid + 64];
  __syncthreads();
  
  // unrolling warp
  if (tid < 32) {
    
    volatile double *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid +  8];
    vmem[tid] += vmem[tid +  4];
    vmem[tid] += vmem[tid +  2];
    vmem[tid] += vmem[tid +  1];
  
  }
  
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
  
}


__global__ void reduceSmem (double *g_idata, double *g_odata, unsigned int n) {

  __shared__ double smem[N_THREADS_1_BLOCK];

  // set thread ID
  unsigned int tid = threadIdx.x;

  // boundary check
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  // convert global data pointer to the local pointer of this block
  double *idata = g_idata + blockIdx.x * blockDim.x;
  
  // set to smem by each threads
  smem[tid] = idata[tid];
  __syncthreads();


  // in-place reduction in shared memory
  if (blockDim.x>=1024 && tid < 512 ) smem[tid] += smem[tid + 512];
  __syncthreads();
  
  if (blockDim.x>=512 && tid < 256 ) smem[tid] += smem[tid + 256];
  __syncthreads();
  
  if (blockDim.x>=256 && tid < 128 ) smem[tid] += smem[tid + 128];
  __syncthreads();
  
  if (blockDim.x>=128 && tid < 64 ) smem[tid] += smem[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    
    volatile double *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid +  8];
    vsmem[tid] += vsmem[tid +  4];
    vsmem[tid] += vsmem[tid +  2];
    vsmem[tid] += vsmem[tid +  1];
  
  }
  
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
  
}

 __global__ void reduceSmemUnroll (double *g_idata, double *g_odata, unsigned int n) {

  __shared__ double smem[N_THREADS_1_BLOCK];

  // set thread ID
  unsigned int tid = threadIdx.x;
  
  // global index, 4 blocks of input data processed at a time
  unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  
  // unrolling 4 blocks 
  double tmpSum = 0;

  // boundary check
  
  if (idx + 3 * blockDim.x <= n){
    double a1 = g_idata[idx];
    double a2 = g_idata[idx + blockDim.x];
    double a3 = g_idata[idx + 2 * blockDim.x];
    double a4 = g_idata[idx + 3 * blockDim.x];
    tmpSum = a1 + a2 + a3 + a4 ;
  }

  smem[tid] = tmpSum; 
  __syncthreads();
  
  
  // in-place reduction and complete unroll
  if (blockDim.x>=1024 && tid < 512 ) smem[tid] += smem[tid + 512];
  __syncthreads();
  
  if (blockDim.x>=512 && tid < 256 ) smem[tid] += smem[tid + 256];
  __syncthreads();
  
  if (blockDim.x>=256 && tid < 128 ) smem[tid] += smem[tid + 128];
  __syncthreads();
  
  if (blockDim.x>=128 && tid < 64 ) smem[tid] += smem[tid + 64];
  __syncthreads();

  
  // unrolling warp
  if (tid < 32) {
    
    volatile double *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid +  8];
    vsmem[tid] += vsmem[tid +  4];
    vsmem[tid] += vsmem[tid +  2];
    vsmem[tid] += vsmem[tid +  1];
  
  }
  
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
  
}


__global__ void SpinUpdateGPU (int icoppia, Conf_type * d_sys, MC_type * d_mc_step){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  int ind_1 = d_mc_step->coppie[2*icoppia];
  int ind_2 = d_mc_step->coppie[2*icoppia+1];
  
  if(i==0)					
    d_sys->xs[ind_1]=d_mc_step->nx1[icoppia];		
  if(i==1)	  				
    d_sys->xs[ind_2]=d_mc_step->nx2[icoppia];		
  if(i==2)	  				
    d_sys->ys[ind_1]=d_mc_step->ny1[icoppia];		
  if(i==3)	  				
    d_sys->ys[ind_2]=d_mc_step->ny2[icoppia];  
  
  __syncthreads();

}


__global__ void  AcceptRejectDevice(int istream, int icoppia, double the_random_number, double mbeta, double de_tot_plaqs, Clock_type * d_clock, Conf_type * d_sys, MC_type * d_mc_step){
  
  // int istream = threadIdx.x;
  // double * target_energy=d_sys->pl_ene;
  // double * source_energy=d_sys->pl_ene_new;

  d_clock->n_attemp[istream]++;
  
  d_mc_step->flag = 0;

  if(de_tot_plaqs<0){
    
    d_clock->acc_rate[istream]++;
    
    // copy the new values of the spins on the DEVICE arrays of spins ---------- // 
    
    int ind_1 = d_mc_step->coppie[2*icoppia];
    int ind_2 = d_mc_step->coppie[2*icoppia+1];
    
    d_sys->xs[ind_1]=d_mc_step->nx1[icoppia];
    d_sys->xs[ind_2]=d_mc_step->nx2[icoppia];
    
    d_sys->ys[ind_1]=d_mc_step->ny1[icoppia];
    d_sys->ys[ind_2]=d_mc_step->ny2[icoppia]; 
    
    
    // -- copy old energy into new energy -------------------------------------- // 

    //UpdateEnergy_device <<< grid_dim, block_dim >>> (target_energy,source_energy);
    d_mc_step->flag = 1;

    //cudaDeviceSynchronize();
    
    d_clock->nrg[istream] += de_tot_plaqs;
    
  }else if(the_random_number < exp(mbeta*de_tot_plaqs)){ 
      
      d_clock->acc_rate[istream]++;
      
      // copy the new values of the spins on the DEVICE arrays of
      // spins ---------- //
      
      int ind_1 = d_mc_step->coppie[2*icoppia];
      int ind_2 = d_mc_step->coppie[2*icoppia+1];
      
      d_sys->xs[ind_1]=d_mc_step->nx1[icoppia];
      d_sys->xs[ind_2]=d_mc_step->nx2[icoppia];
      
      d_sys->ys[ind_1]=d_mc_step->ny1[icoppia];
      d_sys->ys[ind_2]=d_mc_step->ny2[icoppia]; 
      
      // -- copy old energy into new energy -------------------------------------- //
      
      //UpdateEnergy_device <<< grid_dim, block_dim >>> (target_energy,source_energy);
      
      d_mc_step->flag = 1;
      
      //cudaDeviceSynchronize();
      
      d_clock->nrg[istream]+= de_tot_plaqs;
      
    }
  
}
 

__global__ void CreateProposedUpdates(MC_type * d_mc_step, Conf_type * d_sys){
  
  int icoppia = blockIdx.x * blockDim.x + threadIdx.x;
  
  spin_t x1,x2;
  spin_t y1,y2;
  spin_t r1,r2,factor;
  spin_t alpha,phi1,phi2;
  spin_t sp1,cp1,sp2,cp2,sa,ca;
  spin_t nx1,nx2,ny1,ny2;
  
  int ind_1 = d_mc_step->coppie[2*icoppia]; 
  int ind_2 = d_mc_step->coppie[2*icoppia+1]; 
  
  ///////////////////////////////////////////////////////////////////
  ////////////// ------------------ ATTEMPTED UPDATE OF THE TWO SPINS
  ///////////////////////////////////////////////////////////////////
  ////////////// ------------------ i -> 1
  ////////////// ------------------ j -> 2
  
  x1 = d_sys->xs[ind_1]; //spins[i].x;  //xs[i];      
  x2 = d_sys->xs[ind_2]; //spins[j].x;  //xs[j];      
  y1 = d_sys->ys[ind_1]; //spins[i].y;  //ys[i];
  y2 = d_sys->ys[ind_2]; //spins[j].y;  //ys[j];
  
  r1 = x1*x1+y1*y1;  
  r2 = x2*x2+y2*y2;
  factor=mysqrt((r1+r2)/r1);
  /////////////////////////////////////////
  
  // Metropolis
  /////////////////////////////////////////
  
  alpha = (double)d_mc_step->alpha_rand[icoppia]* 6.283185307179586; //*twopi;
  phi1  = (double)d_mc_step->phi1_rand[icoppia] * 6.283185307179586; //*twopi;
  phi2  = (double)d_mc_step->phi2_rand[icoppia] * 6.283185307179586; //*twopi;
  
  mysincos(phi1,&sp1,&cp1);
  mysincos(phi2,&sp2,&cp2);
  mysincos(alpha,&sa,&ca);
  
  nx1 = x1*cp1+y1*sp1;
  ny1 =-x1*sp1+y1*cp1;
  nx2 = x2*cp2+y2*sp2;
  ny2 =-x2*sp2+y2*cp2;
  
  nx1=nx1*factor*ca;
  ny1=ny1*factor*ca;
  
  factor=mysqrt((r1+r2)/r2);
  
  nx2=nx2*factor*sa;
  ny2=ny2*factor*sa;
  
  d_mc_step->nx1[icoppia] = nx1;
  d_mc_step->nx2[icoppia] = nx2;
  d_mc_step->ny1[icoppia] = ny1;
  d_mc_step->ny2[icoppia] = ny2;
  
  __syncthreads();
  
}


__global__ void NewEnergyPlaquetteGPU(int icoppia, MC_type * d_mc_step, Conf_type * d_sys, Int_type * d_inter, double * pl_de_block){

  int iplaq = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  __shared__ double de_local[N_THREADS_1_BLOCK]; 
  
  int i0,i1,i2,i3;
  
  i0 = d_inter->spin_index[4*iplaq+0];
  i1 = d_inter->spin_index[4*iplaq+1];
  i2 = d_inter->spin_index[4*iplaq+2];
  i3 = d_inter->spin_index[4*iplaq+3];
  
  int ind_1;
  int ind_2;
  
  ind_1 = d_mc_step->coppie[2*icoppia];
  ind_2 = d_mc_step->coppie[2*icoppia+1];
  
  spin_t s1x_new,s1y_new,s2x_new,s2y_new;
  
  s1x_new = d_mc_step->nx1[icoppia];
  s1y_new = d_mc_step->ny1[icoppia];
  s2x_new = d_mc_step->nx2[icoppia];
  s2y_new = d_mc_step->ny2[icoppia];
  
  spin_t x[4],y[4];
  
  
  if(i0!=ind_2 && i0!=ind_1 && i1!=ind_2 && i1!=ind_1 && i2!=ind_2 && i2!=ind_1 && i3!=ind_2 && i3!=ind_1){
    
    d_sys->pl_ene_new[iplaq] = d_sys->pl_ene[iplaq];
    
    d_sys->pl_de[iplaq] = 0;
    
    de_local[tid] = 0;
    
  }else{
    
    x[0] = (i0==ind_1)*s1x_new + (i0==ind_2)*s2x_new + (i0!=ind_2 && i0!=ind_1)*d_sys->xs[i0];
    y[0] = (i0==ind_1)*s1y_new + (i0==ind_2)*s2y_new + (i0!=ind_2 && i0!=ind_1)*d_sys->ys[i0];
    
    x[1] = (i1==ind_1)*s1x_new + (i1==ind_2)*s2x_new + (i1!=ind_2 && i1!=ind_1)*d_sys->xs[i1];
    y[1] = (i1==ind_1)*s1y_new + (i1==ind_2)*s2y_new + (i1!=ind_2 && i1!=ind_1)*d_sys->ys[i1];
    
    x[2] = (i2==ind_1)*s1x_new + (i2==ind_2)*s2x_new + (i2!=ind_2 && i2!=ind_1)*d_sys->xs[i2];
    y[2] = (i2==ind_1)*s1y_new + (i2==ind_2)*s2y_new + (i2!=ind_2 && i2!=ind_1)*d_sys->ys[i2];
    
    x[3] = (i3==ind_1)*s1x_new + (i3==ind_2)*s2x_new + (i3!=ind_2 && i3!=ind_1)*d_sys->xs[i3];
    y[3] = (i3==ind_1)*s1y_new + (i3==ind_2)*s2y_new + (i3!=ind_2 && i3!=ind_1)*d_sys->ys[i3];
    
    d_sys->pl_ene_new[iplaq] = - d_inter->J[iplaq] * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3] + ((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
    d_sys->pl_de[iplaq] = d_sys->pl_ene_new[iplaq] - d_sys->pl_ene[iplaq]; 
    
    de_local[tid] = d_sys->pl_de[iplaq];
    
  }
  
  __syncthreads();
  
  
  // // in-place reduction and complete unroll
  
  // if (blockDim.x>=1024 && tid < 512 ) de_local[tid] += de_local[tid + 512];
  // __syncthreads();
  
  // if (blockDim.x>=512 && tid < 256 ) de_local[tid] += de_local[tid + 256];
  // __syncthreads();
  
  // if (blockDim.x>=256 && tid < 128 ) de_local[tid] += de_local[tid + 128];
  // __syncthreads();
  
  if (blockDim.x>=128 && tid < 64 ) de_local[tid] += de_local[tid + 64];
  __syncthreads();
  
  // unrolling warp
  if (tid < 32) {
    
    volatile double *vde_local = de_local;
    vde_local[tid] += vde_local[tid + 32];
    vde_local[tid] += vde_local[tid + 16];
    vde_local[tid] += vde_local[tid +  8];
    vde_local[tid] += vde_local[tid +  4];
    vde_local[tid] += vde_local[tid +  2];
    vde_local[tid] += vde_local[tid +  1];
    
  }
  
  __syncthreads();
  
  // write result for this block to global mem
  
  if (tid == 0) pl_de_block[blockIdx.x] = de_local[0];

  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// PARALLEL MC STEP  PARALLEL MC STEP  ///////////////////////////////////////////////////////////////////////////////////////////
////////////////////// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// MCstep_disorderp4_placchette_parallel(gen,nT,d_clock,clock,d_sys,sys,d_inter,inter,d_mc_step,mc_step);

void MCstep_disorderp4_placchette_parallel(unsigned long long random_seed, 
					   int nT, 
					   double * mbeta,
					   Clock_type * d_clock, 
					   Clock_type * clock, 
					   Conf_type ** d_sys, 
					   Conf_type ** sys, 
					   Int_type * d_inter, 
					   Int_type * inter, 
					   MC_type ** d_mc_step, 
					   MC_type ** mc_step){

  //printf("Temperature = %g \n",T);

  double * point_double;
  spin_t * point_spin_t;
  int * point_int;
  float * point_float;
  
  double iStart_0, iStart_1, iStart_2, iStart_3, iStart_4, iStart_DE;
  double zero  = 0;
  
  // double mbeta[nT];
  // for(int i=0; i<nT; i++) mbeta[i] = -1./mc_step[i]->T;

  int k,icoppia;
  int ind_1,ind_2;
  int Ncoppie = sys[0]->N/2;
  int Nplaqs = inter->Nplaqs;

  //////////////////////////////////////////////////////////////////////////////////////////
  ///// ------ THIS ALLOCATION CAN BE DONE SOMEWHERE ELSE TO SAVE TIME ------- /////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  double ** pl_de_block;
  pl_de_block = (double **)calloc(nT,sizeof(double *));
  
  double ** pl_de_reduced;
  pl_de_reduced = (double **)calloc(nT,sizeof(double *));
  
  for(int i=0; i<nT; i++){
    cudaMalloc((double **) &(pl_de_block[i]),(PLAQ_NUMBER/N_THREADS_1_BLOCK)*sizeof(double));
    
#ifdef USE_REDUCE_UNROLL
    cudaMalloc((double **) &(pl_de_reduced[i]),N_DE_REDUCED*sizeof(double));
#else
    cudaMalloc((double **) &(pl_de_reduced[i]),2*sizeof(double));
#endif
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////
  ///// ------ END OF ALLOCATION --------------------------------------------- /////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  
  
  unsigned int nplacche_ridotte=(PLAQ_NUMBER/N_THREADS_1_BLOCK);
 
  
  //// Prepare random numbers that you will use -------------------------- // 
  
  iStart_0 = cpuSecond(); // ----------------------------------------------------  CONTEGGIO 0 ----- // 
  
  // // -- Ncoppie floats on the device //  
    double random_numbers[Ncoppie];

  for(int i=0; i<nT; i++){
  
    // CUDA_CALL(cudaMemcpy(&point_float,&(d_mc_step[i]->alpha_rand),sizeof(float *),cudaMemcpyDeviceToHost));
    
    for(int ii=0; ii<Ncoppie; ii++) random_numbers[ii]=rand_double();
    cudaMemcpy(&point_double,&(d_mc_step[i]->alpha_rand),sizeof(double *),cudaMemcpyDeviceToHost);
    cudaMemcpy(point_double,random_numbers,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);
    
    for(int ii=0; ii<Ncoppie; ii++) random_numbers[ii]=rand_double();
    cudaMemcpy(&point_double,&(d_mc_step[i]->phi1_rand),sizeof(double *),cudaMemcpyDeviceToHost);
    cudaMemcpy(point_double,random_numbers,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);

    for(int ii=0; ii<Ncoppie; ii++) random_numbers[ii]=rand_double();
    cudaMemcpy(&point_double,&(d_mc_step[i]->phi2_rand),sizeof(double *),cudaMemcpyDeviceToHost);
    cudaMemcpy(point_double,random_numbers,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);    
  
  }
  
  // printf("\n\n# 1) Ho creato i numeri random che mi servono \n");

  // //////////// ----------------------------------------------------------- //

  double de;
  double de_tot_plaqs[nT];
  
  
#ifdef USE_REDUCE_UNROLL
  double de_to_be_summed[nT][N_DE_REDUCED];
#else
  double de_to_be_summed[nT][PLAQ_NUMBER/N_THREADS_1_BLOCK];
#endif

  // Allocate memory and copy the number of plaquettes on the device ------------------ //
  
  
  clock->prof_time[0] += (cpuSecond()-iStart_0); // --------------------  END CONTEGGIO 0 ---------------- //
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  

  iStart_1 = cpuSecond();   // ----------------------------------------------------  CONTEGGIO 1 ----- // 
  
  unsigned int dim_block;
  unsigned int dim_grid;

  if(Ncoppie>N_THREADS_1_BLOCK){
    dim_block=N_THREADS_1_BLOCK;
    dim_grid=(unsigned int)((Ncoppie+dim_block-1)/dim_block);
  }else{
    dim_block=Ncoppie;
    dim_grid=1;
  }

  dim3 blockPRE (dim_block);
  dim3 gridPRE  (dim_grid);
  

  ///////////////////////////////////////////////
  ///////////////////////////////////////////////
  
  for(int i=0; i<nT; i++){
    CreateProposedUpdates <<< gridPRE, blockPRE >>> (d_mc_step[i],d_sys[i]);     
  }
  
  /////////////////////////////////////////////////
  ///// ---- DISTRUZIONE DEGLI STREAMS ------ ///// 
  /////////////////////////////////////////////////
  
  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  
  // printf("\n\n# 2) Ho generato i nuovi valori degli spin da proporre \n");
  
  //cudaDeviceSynchronize();
  
  ///////////////////////////////////////////////////////////////////
  // PREPARE GPUs FOR MC SWEEPS /////////////////////////////////////
  ///////////////////////////////////////////////////////////////////
  
  
  dim3 block (N_THREADS_1_BLOCK); //(BLOCK_SIZE);
  dim3 grid ((Nplaqs+block.x-1)/block.x);
  
  // printf("\n\n# 2.a) Nplaqs = %d \n",Nplaqs);
  
  clock->prof_time[1] += (cpuSecond()-iStart_1); // --------------------  END CONTEGGIO 1 ---------------- //
  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  
  
//   //////////////////////////////////////////////////////////////////////////////////////////////////////////
//   /// START ////////////////////// START LOOP ON SPIN DOUBLETS /////////////////////////////////////////////
//   //////////////////////////////////////////////////////////////////////////////////////////////////////////
  
//   // printf("\n\n# 3) Comincio il loop per l'update delle energie \n");

#ifdef CYCLE_UPDATE_ENERGY
  
  for(int icoppia=0;icoppia<Ncoppie;icoppia++){
    
    iStart_2 = cpuSecond(); // ----------------------------------------------------  CONTEGGIO 3 ----- // 
    
    
    for(int i=0; i<nT; i++){
      
      NewEnergyPlaquetteGPU <<< grid, block >>> (icoppia,d_mc_step[i],d_sys[i],d_inter,pl_de_block[i]);
      
#ifdef USE_REDUCE_UNROLL
      
      reduceSmemUnroll <<<  N_DE_REDUCED, block >>> (pl_de_block[i],pl_de_reduced[i],nplacche_ridotte);
      cudaMemcpy(de_to_be_summed[i],pl_de_reduced[i],N_DE_REDUCED*sizeof(double),cudaMemcpyDeviceToHost);
      
      de_tot_plaqs[i]=0;
      for(int ii=0; ii<N_DE_REDUCED; ii++) de_tot_plaqs[i]+=de_to_be_summed[i][ii];
      
#else
      cudaMemcpy(de_to_be_summed[i],pl_de_block[i],(PLAQ_NUMBER/N_THREADS_1_BLOCK)*sizeof(double),cudaMemcpyDeviceToHost);
      
      de_tot_plaqs[i]=0;
      for(int ii=0; ii<(PLAQ_NUMBER/N_THREADS_1_BLOCK); ii++) de_tot_plaqs[i]+=de_to_be_summed[i][ii];

#endif
      

    
    }
    
    clock->prof_time[2]+=(cpuSecond()-iStart_2);

   
    // iStart_4 = cpuSecond();   // ----------------------------------------------------  CONTEGGIO 3 ----- // 
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// START ////////////////////// ACCEPTANCE REJECTION DECISION ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    double the_random_number[nT];
    
    for(int i=0; i<nT; i++){

      the_random_number[i]=rand_double();

      AcceptRejectDevice <<< 1, 1 >>>(i,icoppia,the_random_number[i],mbeta[i],de_tot_plaqs[i],d_clock,d_sys[i],d_mc_step[i]);

      cudaMemcpy(&(mc_step[i]->flag),&(d_mc_step[i]->flag),sizeof(int),cudaMemcpyDeviceToHost);
      
      if(mc_step[i]->flag){
    	UpdateEnergy <<< grid, block >>>(d_sys[i]);
      }
      
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// END ////////////////////// ACCEPTANCE REJECTION DECISION /////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /// clock->prof_time[4] += (cpuSecond()-iStart_4); // --------------------  END CONTEGGIO 1 ---------------- //    
  
    // printf("\n\n# 7) Ho deciso se accettare o meno la mossa \n");  
    // exit(0);    

  }

#endif
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// END ////////////////////// END LOOP ON SPIN DOUBLETS /////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // CICLO SULLE N COPPIE DI SPIN DA AGGIORNARE //  for(int k=0;k<N;k++){
  
  //cudaDeviceReset();

  for(int i=0; i<nT; i++){
    cudaFree(pl_de_block[i]);
    cudaFree(pl_de_reduced[i]);
  }
  
  free(pl_de_block);
  free(pl_de_reduced);
  
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// OLD SERIAL MC UPDATE ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void MCstep_disorderp4_withAll(int N,spin_t *xs, spin_t *ys, double *gain, int *cumNbq,int *quads,double *betas,double *J,double *nrg) 
{
  
  

  double numero_casuale=rand_double();
  int n = N*numero_casuale;  
  
  //printf("N = %d; numero = %g \n",N,numero_casuale);

  int d1 = cumNbq[n+1]-cumNbq[n];
  
  //printf("dentro MC step \n"); 

  for(int k=0;k<N;k++)
    {
      if(k!=n)
	{
	  
	  spin_t x1 = xs[n];      
	  spin_t x2 = xs[k];      
	  spin_t y1 = ys[n];
	  spin_t y2 = ys[k];
	  spin_t r1 = x1*x1+y1*y1;  
	  spin_t r2 = x2*x2+y2*y2;
	  spin_t factor=mysqrt((r1+r2)/r1);

	  /////////////////////////////////////////
	  
	  // Metropolis
	  /////////////////////////////////////////
	  spin_t alpha = rand_double()*twopi;
	  spin_t phi1  = rand_double()*twopi;
	  spin_t phi2  = rand_double()*twopi;
	  
	  spin_t sp1,cp1,sp2,cp2,sa,ca;

	  mysincos(phi1,&sp1,&cp1);
	  mysincos(phi2,&sp2,&cp2);
	  mysincos(alpha,&sa,&ca);

	  spin_t nx1=x1*cp1+y1*sp1;
	  spin_t ny1=-x1*sp1+y1*cp1;
	  spin_t nx2=x2*cp2+y2*sp2;
	  spin_t ny2=-x2*sp2+y2*cp2;

	  nx1=nx1*factor*ca;
	  ny1=ny1*factor*ca;
	  
	  factor=mysqrt((r1+r2)/r2);
	  
	  nx2=nx2*factor*sa;
	  ny2=ny2*factor*sa;	  
	  //////////////////////////////
	  
  	  
	  double de=0.;
	  
#if I_WANT_GAIN	  
	  de = gain[n]*(x1*x1+y1*y1-nx1*nx1-ny1*ny1) + gain[k]*(x2*x2+y2*y2-nx2*nx2-ny2*ny2);      
#endif	  

	  //de di n
	  for(int qi=0;qi<d1;qi++)
	    {
	      int myquad=cumNbq[n]+qi;
	      
	      int myquadm[4];
	      spin_t x[4],y[4];	      
	      
	      int whoism=-1;
	      int whoisp=-1;
	      for(int j=0;j<4;j++)
		{		  
		  myquadm[j]= quads[4*myquad+j];
		  if(n==myquadm[j]){whoism=j;}
		  if(k==myquadm[j]){whoisp=j;}
		  x[j] = xs[ myquadm[j] ];
		  y[j] = ys[ myquadm[j] ];
		}
	      
	      de +=  J[myquad] * (((x[0]*x[1]-y[0]*y[1])*x[2] +
				   (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]
				  +
				  ((y[0]*y[1]-x[0]*x[1])*y[2] +
				   (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
	      
	      
	      x[whoism]=nx1;
	      y[whoism]=ny1;
	      if(whoisp!=-1)
		{
		  x[whoisp]=nx2;
		  y[whoisp]=ny2;
		}
	      
	      //new energy:
	      de -=  J[myquad] * (((x[0]*x[1]-y[0]*y[1])*x[2] +
				   (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]
				  +
				  ((y[0]*y[1]-x[0]*x[1])*y[2] +
				   (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);


	    }
	  
	  
	  
	  
	  //fine delta nrg of n
	  ////////////////////////////////////////////////////////////
	  //delta nrg of pi
	  ///////////////////////////////////////////////////

	  int d2 = cumNbq[k+1]-cumNbq[k];

	  for(int qi=0;qi<d2;qi++)
	    {
	      int myquad = cumNbq[k]+qi;
	      
	      int myquadm[4];
	      spin_t x[4],y[4];
	      	      
	      int whoism = -1;
	      int whoisp = -1;
	      for(int j=0;j<4;j++)
		{
		  myquadm[j] = quads[4*myquad+j];
		  if(n==myquadm[j]){whoism=j;}
		  if(k==myquadm[j]){whoisp=j;}
		  x[j] = xs[ myquadm[j] ];
		  y[j] = ys[ myquadm[j] ];
		}
	      
	      if(whoism==-1)
		{		  
		  
		  de +=  J[myquad] * (((x[0]*x[1]-y[0]*y[1])*x[2] +
				       (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]
				      +
				      ((y[0]*y[1]-x[0]*x[1])*y[2] +
				       (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
		  
		  
		  x[whoisp]=nx2;
		  y[whoisp]=ny2;
		  
		  //new energy:
		  de -=  J[myquad] * (((x[0]*x[1]-y[0]*y[1])*x[2] +
				       (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]
				      +
				      ((y[0]*y[1]-x[0]*x[1])*y[2] +
				       (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
		  
		}
	      //      printf("%d %d : %d %d : %f\n",k,qi,whoisp,whoism,de); 
	      
	    }
	  
	  //fine delta nrg of m21 
	  ////////////////////////////////////////////////////////////
	  
	  
	  
	  double beta=(*betas);  //  this index is NOT myq	  
	  if( rand_double() < exp(-beta*(de)) )
	    { 
	      xs[n] = nx1;
	      xs[k] = nx2;
	      ys[n] = ny1;
	      ys[k] = ny2;
	      (*nrg) += de;
	    }
	  /////////////////////////////////////////
	  
	}
    }
  
  //-----------------------------------------------------------------------------
  ////////////////////////////////////////////////////////////////
  
  //return;
  
}
