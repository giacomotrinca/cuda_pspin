void randomize_spin_coppie_host(int seed, int N, int * coppie){
  
  open_rng(seed);

  for(int i=N-1; i>0; i--){
    int i_rand     = (int)floor((i-1)*rand_double());
    int temp       = coppie[i];
    coppie[i]    = coppie[i_rand];
    coppie[i_rand] = temp;
  }

  close_rng();

}

double energy_plaquette(int iplaq, Conf_type * sys, Int_type * inter){

  double ene=0;
  int Nplaqs=inter->Nplaqs;

  double x[4];
  double y[4];

  for(int j=0;j<4;j++){
    
    x[j] = sys->xs[inter->spin_index[4*iplaq+j]];
    y[j] = sys->ys[inter->spin_index[4*iplaq+j]];

  }

  ///// ------ IMPLEMENTATION  "#if I WANT GAIN" ------  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

  ene  =   - inter->J[iplaq] * (+x[0]*x[1]*x[2]*x[3] 
			        +y[0]*y[1]*y[2]*y[3] 
			        -x[0]*x[1]*y[2]*y[3] 
			        -y[0]*y[1]*x[2]*x[3] 
			        +x[0]*y[1]*x[2]*y[3] 
			        +y[0]*x[1]*y[2]*x[3] 
			        +x[0]*y[1]*y[2]*x[3] 
			        +y[0]*x[1]*x[2]*y[3]);

  sys->pl_ene[iplaq] = ene;

  //placchette[iplaq].ene = ene;
  //placchette[iplaq].ene_new = ene;

  return ene;
  
}

double energyPT_disorder1replica_plaqs(Conf_type * sys, Int_type * inter) {

  double enetot=0.;      

  for(int k=0; k<inter->Nplaqs; k++){
    enetot+=energy_plaquette(k,sys,inter); // Inizializza sia ene che ene_new 
  }
  
// #if I_WANT_GAIN

//   for(int k=0; k<sys->N; k++)
//     enetot += sys->gain[k]*(sys->xs[k]*sys->xs[k]+sys->ys[k]*sys->ys[k]);

// #endif

  return enetot;

}

void InitGraphStructure(int seed_int, int N, Plaqs_type * placchette, int * wsindices){
  
 // ----- TEST SOMMA -------------------- //////////// 
  
#if FREQ_ENABLE == 0
  int Nt = 0.03*N*N*N;
  printf("#Nt = 0.03*N*N*N  = %d \n",Nt);
#else
  int Nt = 0.032*N*N*N*N;   // should be order N^4 to avoid PC in the disordered case, but if it is too big the program takes a lot of time to find the quads randomly
  printf("#Nt = 0.032*N*N*N*N  = %d \n",Nt);
#endif
  
  vector<vector<int> > tetrads;
    
  printf("#procedo con generateTetradsHashing \n");
  generateTetradsHashing_giacomo(N,tetrads,seed_int+1);	 // Qui costruisce le tetradi random, senza implementazione della FMC
  printf("#Ho concluso con generateTetradsHashing \n");


  printf("#Comincio a generare il grafo delle interazioni \n");

  vector<Plaqs_type> placchette_vec; //-----------------GIACOMO ------- 09/2017
 
  printf("#Number of tetrads = %d \n",(int)tetrads.size());

  generate4pletsCycles_discreteWs_wPlaqs(tetrads,placchette_vec,wsindices); 
  
  
  printf("#Number of plaquettes (ORIGINAL) = %d \n",(int)placchette_vec.size());

  tetrads.clear(); // cancella le tetradi random perchè non gli servono più e occupano spazio

  int Nplaqs = placchette_vec.size();

  //int Nplaqs_new = (int)floor((double)Nplaqs/N_THREADS_1_BLOCK)*N_THREADS_1_BLOCK;
  
  if(Nplaqs>=PLAQ_NUMBER)
    placchette_vec.resize(PLAQ_NUMBER);
  else{
    printf("there is a problem in Interactions Initialization \n");
    exit(0);
  }

  Nplaqs = PLAQ_NUMBER;
    
  printf("#Ho generato il grafo delle interazioni with Nplaqs = %d \n",Nplaqs);
  
  if(DISORDER){
    
    double sigmaJ = sqrt( 1. * _sJvalue_ * N / Nplaqs );
    double avgJ = 1. * _Jvalue_ * N / Nplaqs;
    
    randomGaussianCouplingsp4_giacomo(Nplaqs,placchette_vec,avgJ,sigmaJ,seed_int+2);  // the change with respect to the version V6 is all here
    
  }else{
    
    for(int j=0;j<Nplaqs;j++)                     // no-optimized ordered case
      placchette_vec[j].J = 1. * N / Nplaqs;      // so we are sure that the energy is extensive (if there is no localization)
    
  }
  
  printf("# Ho inizializzato couplings \n");

  plaqs_from_vec_to_array(N,Nplaqs,placchette_vec,placchette); // copia vector <Plaqs_type> in Plaqs_type * array 

  placchette_vec.clear(); // distrugge il vector

  printf("# Ho copiato placchette su un * double e disallocato vector<double> \n\n");
  
  
}


void init_Interactions_host(int Nplaqs, Int_type * inter, Plaqs_type * placchette){
  
  for(int np=0; np<Nplaqs; np++){
    
    inter->J[np]=placchette[np].J;
    
    for(int ispin=0; ispin<4; ispin++){ 
      inter->spin_index[4*np+ispin]=placchette[np].spin_index[ispin];
      if(inter->spin_index[4*np+ispin]!=placchette[np].spin_index[ispin]){
	printf("ERROR \n");
	exit(1);
      }
    }
  }
  
  printf("# I allocated and initialized INTERACTIONS on host \n");
  
}

void init_System_device(int ireplica, int seed, int N, int Nplaqs, double temperature, Conf_type * sys, Conf_type * d_sys){ 
  
  open_rng(seed);
  
  sys->identity = (char *)calloc(NCHAR_IDENTITY,sizeof(char));      
  sys->xs = (spin_t *)calloc(N,sizeof(spin_t));      
  sys->ys = (spin_t *)calloc(N,sizeof(spin_t));
  sys->pl_ene = (double *)calloc(Nplaqs,sizeof(double));
  sys->N = N; 
  sys->T = temperature; 

  // printf("# I allocated SPINS on host \n");
  
  for(int i=0; i<N; i++){
    
    uniformInit(sys->xs,sqrt(epsilonSM)/sqrt(2.),N,seed);   
    uniformInit(sys->ys,sqrt(epsilonSM)/sqrt(2.),N,seed+1);
    
  }
 
  if(ireplica<10){
    sprintf(sys->identity,"%d%d%d%d%d%d%d%d",0,ireplica,0,ireplica,0,ireplica,0,ireplica);
  }else{
    sprintf(sys->identity,"%d%d%d%d",ireplica,ireplica,ireplica,ireplica);
  }

  // printf("# I allocated and initialized SPINS on host \n");
  
  ////////////////////////////////////////////////////////////// 
  /// ALLOCATE SPACE AND COPY "SPINS" ON THE DEVICE ////////////
  //////////////////////////////////////////////////////////////
  
  spin_t *point_xs, *point_ys;
  
  cudaMalloc((spin_t **) &(point_xs),N*sizeof(spin_t));
  cudaMalloc((spin_t **) &(point_ys),N*sizeof(spin_t));
  
  cudaMemcpy(&(d_sys->xs),&(point_xs),sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->ys),&(point_ys),sizeof(spin_t *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_xs,sys->xs,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ys,sys->ys,N*sizeof(spin_t),cudaMemcpyHostToDevice);
  
  char *point_identity;

  cudaMalloc((char **) &(point_identity),NCHAR_IDENTITY*sizeof(char));
  cudaMemcpy(&(d_sys->identity),&(point_identity),sizeof(char *),cudaMemcpyHostToDevice);
  cudaMemcpy(point_identity,sys->identity,NCHAR_IDENTITY*sizeof(char),cudaMemcpyHostToDevice);
  
  ///////////////////////////////////////////////////////////////// 
  /// ALLOCATE SPACE FOR PLAQUETTES ON THE DEVICE /////////////////
  /////////////////////////////////////////////////////////////////
  
  double *point_ene, *point_ene_new, *point_de;
  double *point_de_block, *point_de_reduced;
  double *zeri_ene,*zeri_ene_block,*zeri_ene_reduced;

  zeri_ene = (double *)calloc(Nplaqs,sizeof(double));
  zeri_ene_block = (double *)calloc(Nplaqs/N_THREADS_1_BLOCK,sizeof(double));
  
  cudaMalloc((double **) &point_ene,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_ene_new,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_de,Nplaqs*sizeof(double));
  cudaMalloc((double **) &point_de_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double));
  
  cudaMemcpy(&(d_sys->N),&N,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_ene),&(point_ene),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_ene_new),&(point_ene_new),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_de),&(point_de),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sys->pl_de_block),&(point_de_block),sizeof(double *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_ene,sys->pl_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ene_new,zeri_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_de,zeri_ene,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_de_block,zeri_ene_block,(Nplaqs/N_THREADS_1_BLOCK)*sizeof(double),cudaMemcpyHostToDevice);
    
  cudaMemcpy(&(d_sys->T),&(sys->T),sizeof(double),cudaMemcpyHostToDevice);

  //printf("# I allocated and initialized plaquettes values on the DEVICE \n");     

  printf("# I allocated and initialized SPINS and ENERGY for REPLICA n°= %d \n",ireplica);

} 


void init_Inter_device(int Nplaqs, Int_type * inter, Int_type * d_inter){

  int *point_spin_index;
  int *point_Nplaqs,*pt_Nplaqs;
  
  double *point_J;
  
  cudaMalloc((int **) &(point_spin_index),4*Nplaqs*sizeof(int));
  cudaMalloc((double **) &(point_J),Nplaqs*sizeof(double));
  
  cudaMemcpy(&(d_inter->Nplaqs),&Nplaqs,sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_inter->spin_index),&(point_spin_index),sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_inter->J),&(point_J),sizeof(double *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_spin_index,inter->spin_index,4*Nplaqs*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(point_J,inter->J,Nplaqs*sizeof(double),cudaMemcpyHostToDevice);
  
}


void Initialize_counters(Clock_type * clock, Clock_type * d_clock){
  
  clock->prof_time = (double *)calloc(NTIMES_PROFILING,sizeof(double));  
  clock->nrg = (double *)calloc(NPT,sizeof(double));
  clock->acc_rate = (int *)calloc(NPT,sizeof(int));
  clock->n_attemp = (int *)calloc(NPT,sizeof(int));
  
  double * point_prof_time, * point_nrg;
  int * point_acc_rate, * point_n_attemp;
  
  double * zeri_NTIMES_double, * zeri_NPT_double;
  int * zeri_NPT_int;
  
  zeri_NTIMES_double = (double *)calloc(NTIMES_PROFILING,sizeof(double));
  zeri_NPT_double = (double *)calloc(NPT,sizeof(double));
  zeri_NPT_int = (int *)calloc(NPT,sizeof(int));
  
  cudaMalloc((double **) &point_prof_time,NTIMES_PROFILING*sizeof(double));
  cudaMalloc((double **) &point_nrg,NPT*sizeof(double));
  cudaMalloc((int **) &point_acc_rate,NPT*sizeof(int));
  cudaMalloc((int **) &point_n_attemp,NPT*sizeof(int));
  
  cudaMemcpy(&(d_clock->prof_time),&(point_prof_time),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_clock->nrg),&(point_nrg),sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_clock->acc_rate),&(point_acc_rate),sizeof(int *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_clock->n_attemp),&(point_n_attemp),sizeof(int *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_prof_time,zeri_NTIMES_double,NTIMES_PROFILING*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_nrg,zeri_NPT_double,NPT*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_acc_rate,zeri_NPT_int,NPT*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(point_n_attemp,zeri_NPT_int,NPT*sizeof(int),cudaMemcpyHostToDevice);

}

void init_MC_step_variables(int ireplica, double T, int N, int Ncoppie, MC_type * mc_step, MC_type * d_mc_step){

  int sequence_spin[N];
  for(int i=0; i<N; i++) sequence_spin[i]=i;

  double zeri_nrandom_coppie[N];
  for(int i=0; i<N; i++) zeri_nrandom_coppie[i]=0;

  double zeri_coppie[Ncoppie];
  for(int i=0; i<Ncoppie; i++) zeri_coppie[i]=0;

  
  mc_step->T = T;
  mc_step->flag = 0;
  
  spin_t * point_nx1, * point_nx2, * point_ny1, * point_ny2;
  double * point_alpha_rand, * point_phi1_rand, * point_phi2_rand, * point_nrandom_coppie;
  int * point_coppie;
  int zero=0;

  // int * coppie;
  // coppie = (int *)calloc(N,sizeof(int));
  
  cudaMalloc((spin_t **) &point_nx1,Ncoppie*sizeof(spin_t));
  cudaMalloc((spin_t **) &point_nx2,Ncoppie*sizeof(spin_t));
  
  cudaMalloc((spin_t **) &point_ny1,Ncoppie*sizeof(spin_t));
  cudaMalloc((spin_t **) &point_ny2,Ncoppie*sizeof(spin_t));

  cudaMemcpy(&(d_mc_step->nx1),&point_nx1,sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->nx2),&point_nx2,sizeof(spin_t *),cudaMemcpyHostToDevice);

  cudaMemcpy(&(d_mc_step->ny1),&point_ny1,sizeof(spin_t *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->ny2),&point_ny2,sizeof(spin_t *),cudaMemcpyHostToDevice);

  cudaMemcpy(point_nx1,zeri_coppie,Ncoppie*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_nx2,zeri_coppie,Ncoppie*sizeof(spin_t),cudaMemcpyHostToDevice);
  	              
  cudaMemcpy(point_ny1,zeri_coppie,Ncoppie*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(point_ny2,zeri_coppie,Ncoppie*sizeof(spin_t),cudaMemcpyHostToDevice);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
    
  cudaMalloc((double **) &point_alpha_rand,Ncoppie*sizeof(double));
  cudaMalloc((double **) &point_phi1_rand,Ncoppie*sizeof(double));
  cudaMalloc((double **) &point_phi2_rand,Ncoppie*sizeof(double));
  
  cudaMemcpy(&(d_mc_step->alpha_rand),&point_alpha_rand,sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->phi1_rand),&point_phi1_rand,sizeof(double *),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->phi2_rand),&point_phi2_rand,sizeof(double *),cudaMemcpyHostToDevice);
  
  cudaMemcpy(point_alpha_rand,zeri_coppie,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_phi1_rand ,zeri_coppie,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(point_phi2_rand ,zeri_coppie,Ncoppie*sizeof(double),cudaMemcpyHostToDevice);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  

  cudaMemcpy(&(d_mc_step->T),&(mc_step->T),sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->flag),&(mc_step->flag),sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->Ncoppie),&Ncoppie,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_mc_step->icoppia),&zero,sizeof(int),cudaMemcpyHostToDevice);

  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  cudaMalloc((int **) &point_coppie,N*sizeof(int));
  cudaMemcpy(&(d_mc_step->coppie),&point_coppie,sizeof(int *),cudaMemcpyHostToDevice);  
  cudaMemcpy(point_coppie,sequence_spin,N*sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((double **) &point_nrandom_coppie,N*sizeof(double));
  cudaMemcpy(&(d_mc_step->rnumbers_coppie),&point_nrandom_coppie,sizeof(double *),cudaMemcpyHostToDevice);  
  cudaMemcpy(point_nrandom_coppie,zeri_nrandom_coppie,N*sizeof(double),cudaMemcpyHostToDevice);

  // INITIALIZATION SPIN INDICES ON HOST
  mc_step->coppie = (int *)calloc(N,sizeof(int));
  for(int i=0; i<N; i++) mc_step->coppie[i]=i;

  printf("# I initialized temporary vars for MC DYNAMICS : REPLICA n°= %d \n",ireplica);

}

void crea_replica(int seed, int N, int Nplaqs, int Ncoppie, double * temp, Conf_type ** sys, Conf_type ** d_sys, MC_type ** mc_step, MC_type ** d_mc_step){
    
  // ALLOCA LA STRUTTURA DATI "conf"
  
  for(int i=0;i<NPT;i++){
    
    sys[i] = (Conf_type *)calloc(NPT,sizeof(Conf_type));
    
    cudaMalloc((Conf_type **) &(d_sys[i]),sizeof(Conf_type));
    
    init_System_device(i,seed,N,Nplaqs,temp[i],sys[i],d_sys[i]);
    
  }
  
  
  // ALLOCA LA STRUTTURA DATI "MC_steps"
  
  for(int i=0; i<NPT; i++){
    
    mc_step[i] = (MC_type *)calloc(1,sizeof(MC_type));    
    
    cudaMalloc((MC_type **) &(d_mc_step[i]),sizeof(MC_type)); 
    
    init_MC_step_variables(i,temp[i],N,Ncoppie,mc_step[i],d_mc_step[i]);
    
  }
  
}


void print_replica(int ind_iter, double * temp, int N, Clock_type * clock, FILE * fout_parallel){
      
#ifdef REPLICA_EXCHANGE

 /*  fprintf(fout_parallel," %d %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4 \
e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e\n",
	  (ind_iter+1)*NSTEP,                                                                            // $1 
	  temp[1],(double)clock->acc_rate_exchange[0]/clock->n_attemp_exchange[0],clock->nrg[1]/N,       // $4 
	  temp[2],(double)clock->acc_rate_exchange[1]/clock->n_attemp_exchange[1],clock->nrg[2]/N,       // $7 
	  temp[3],(double)clock->acc_rate_exchange[2]/clock->n_attemp_exchange[2],clock->nrg[3]/N,       // $10
	  temp[4],(double)clock->acc_rate_exchange[3]/clock->n_attemp_exchange[3],clock->nrg[4]/N,       // $13
	  temp[5],(double)clock->acc_rate_exchange[4]/clock->n_attemp_exchange[4],clock->nrg[5]/N,       // $16
	  temp[6],(double)clock->acc_rate_exchange[5]/clock->n_attemp_exchange[5],clock->nrg[6]/N,       // $19
	  temp[7],(double)clock->acc_rate_exchange[6]/clock->n_attemp_exchange[6],clock->nrg[7]/N,       // $22
	  temp[8],(double)clock->acc_rate_exchange[7]/clock->n_attemp_exchange[7],clock->nrg[8]/N,       // $25
	  temp[9],(double)clock->acc_rate_exchange[8]/clock->n_attemp_exchange[8],clock->nrg[9]/N,       // $28
	  temp[10],(double)clock->acc_rate_exchange[9]/clock->n_attemp_exchange[9],clock->nrg[10]/N,     // $31
	  temp[11],(double)clock->acc_rate_exchange[10]/clock->n_attemp_exchange[10],clock->nrg[11]/N,   // $34
	  temp[12],(double)clock->acc_rate_exchange[11]/clock->n_attemp_exchange[11],clock->nrg[12]/N,   // $37
	  temp[13],(double)clock->acc_rate_exchange[12]/clock->n_attemp_exchange[12],clock->nrg[13]/N,   // $40
	  temp[14],(double)clock->acc_rate_exchange[13]/clock->n_attemp_exchange[13],clock->nrg[14]/N,   // $43
	  temp[15],(double)clock->acc_rate_exchange[14]/clock->n_attemp_exchange[14],clock->nrg[15]/N,   // $46
	  temp[16],(double)clock->acc_rate_exchange[15]/clock->n_attemp_exchange[15],clock->nrg[16]/N,   // $49
	  temp[17],(double)clock->acc_rate_exchange[16]/clock->n_attemp_exchange[16],clock->nrg[17]/N,   // $52
	  temp[18],(double)clock->acc_rate_exchange[17]/clock->n_attemp_exchange[17],clock->nrg[18]/N,   // $55
	  temp[19],(double)clock->acc_rate_exchange[18]/clock->n_attemp_exchange[18],clock->nrg[19]/N,   // $58
	  temp[20],(double)clock->acc_rate_exchange[19]/clock->n_attemp_exchange[19],clock->nrg[20]/N,   // $61
	  temp[21],(double)clock->acc_rate_exchange[20]/clock->n_attemp_exchange[20],clock->nrg[21]/N,   // $64
	  temp[22],(double)clock->acc_rate_exchange[21]/clock->n_attemp_exchange[21],clock->nrg[22]/N,   // $67
	  temp[23],(double)clock->acc_rate_exchange[22]/clock->n_attemp_exchange[22],clock->nrg[23]/N,   // $70
	  temp[24],(double)clock->acc_rate_exchange[23]/clock->n_attemp_exchange[23],clock->nrg[24]/N,   // $73
	  temp[25],(double)clock->acc_rate_exchange[24]/clock->n_attemp_exchange[24],clock->nrg[25]/N,   // $76
	  temp[26],(double)clock->acc_rate_exchange[25]/clock->n_attemp_exchange[25],clock->nrg[26]/N,   // $79
	  temp[27],(double)clock->acc_rate_exchange[26]/clock->n_attemp_exchange[26],clock->nrg[27]/N,   // $82
	  temp[28],(double)clock->acc_rate_exchange[27]/clock->n_attemp_exchange[27],clock->nrg[28]/N,   // $85
	  temp[29],(double)clock->acc_rate_exchange[28]/clock->n_attemp_exchange[28],clock->nrg[29]/N,   // $88
	  temp[30],(double)clock->acc_rate_exchange[29]/clock->n_attemp_exchange[29],clock->nrg[30]/N,  // $91
	  temp[31],(double)clock->acc_rate_exchange[30]/clock->n_attemp_exchange[30],clock->nrg[31]/N);  // $94 */

  fprintf(fout_parallel," %d \t", (ind_iter+1)*NSTEP);
  for(int i=0;i<NTJUMPS;i++){
    fprintf(fout_parallel," %g %8.4e %8.8e \t",  temp[i+1],(double)clock->acc_rate_exchange[i]/clock->n_attemp_exchange[i],clock->nrg[i+1]/N);
  }
  fprintf(fout_parallel, "\n"); 

  fclose(fout_parallel);
  
  /* printf(" %d %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e \n",
	 (ind_iter+1)*NSTEP,
	 temp[1],(double)clock->acc_rate_exchange[0]/clock->n_attemp_exchange[0],clock->nrg[1]/N,
	 temp[9],(double)clock->acc_rate_exchange[8]/clock->n_attemp_exchange[8],clock->nrg[9]/N,
	 temp[17],(double)clock->acc_rate_exchange[16]/clock->n_attemp_exchange[16],clock->nrg[17]/N,
	 temp[25],(double)clock->acc_rate_exchange[24]/clock->n_attemp_exchange[24],clock->nrg[25]/N,
	 temp[29],(double)clock->acc_rate_exchange[28]/clock->n_attemp_exchange[28],clock->nrg[29]/N); */

   printf(" %d %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e %g %8.4e %8.8e \n",
	 (ind_iter+1)*NSTEP,
	 temp[1],(double)clock->acc_rate_exchange[0]/clock->n_attemp_exchange[0],clock->nrg[1]/N,
	 temp[NPT/4+1],(double)clock->acc_rate_exchange[NPT/4]/clock->n_attemp_exchange[NPT/4],clock->nrg[NPT/4+1]/N,
	 temp[NPT/2+1],(double)clock->acc_rate_exchange[NPT/2]/clock->n_attemp_exchange[NPT/2],clock->nrg[NPT/2+1]/N,
	 temp[NPT/4*3+1],(double)clock->acc_rate_exchange[NPT/4*3]/clock->n_attemp_exchange[NPT/4*3],clock->nrg[NPT/4*3+1]/N,
	 temp[NPT-1],(double)clock->acc_rate_exchange[NPT-2]/clock->n_attemp_exchange[NPT-2],clock->nrg[NPT-1]/N);  
  
  
#else
  
  fprintf(fout_parallel," %d %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e \n",
	  (ind_iter+1)*NSTEP,           // $1 
	  temp[1],clock->nrg[1]/N,      // $3
	  temp[2],clock->nrg[2]/N,	// $5 
	  temp[3],clock->nrg[3]/N,	// $7
	  temp[4],clock->nrg[4]/N,	// $9
	  temp[5],clock->nrg[5]/N,	// $11
	  temp[6],clock->nrg[6]/N,	// $13
	  temp[7],clock->nrg[7]/N,	// $15
	  temp[8],clock->nrg[8]/N,	// $17
	  temp[9],clock->nrg[9]/N,	// $19
	  temp[10],clock->nrg[10]/N,	// $21
	  temp[11],clock->nrg[11]/N,	// $23
	  temp[12],clock->nrg[12]/N,	// $25
	  temp[13],clock->nrg[13]/N,	// $27
	  temp[14],clock->nrg[14]/N,	// $29
	  temp[15],clock->nrg[15]/N,	// $31
	  temp[16],clock->nrg[16]/N,	// $33
	  temp[17],clock->nrg[17]/N,	// $35
	  temp[18],clock->nrg[18]/N,	// $37
	  temp[19],clock->nrg[19]/N,	// $39
	  temp[20],clock->nrg[20]/N,	// $41
	  temp[21],clock->nrg[21]/N,	// $43
	  temp[22],clock->nrg[22]/N,	// $45
	  temp[23],clock->nrg[23]/N,	// $47
	  temp[24],clock->nrg[24]/N,	// $49
	  temp[25],clock->nrg[25]/N,	// $51
	  temp[26],clock->nrg[26]/N,	// $53
	  temp[27],clock->nrg[27]/N,	// $55
	  temp[28],clock->nrg[28]/N,	// $57
	  temp[29],clock->nrg[29]/N,	// $59
	  temp[30],clock->nrg[30]/N); 	// $61
  
  fclose(fout_parallel);
  
  printf(" %d %g %8.8e %g %8.8e %g %8.8e %g %8.8e %g %8.8e \n",
	 (ind_iter+1)*NSTEP,
	 temp[1] ,clock->nrg[1]/N,
	 temp[9],clock->nrg[9]/N,
	 temp[17],clock->nrg[17]/N,
	 temp[25],clock->nrg[25]/N,
	 temp[29],clock->nrg[29]/N);
#endif
  
}




void exchange_replicas_Parallel_Tempering(int seed, double * temp, Clock_type * clock, Clock_type * d_clock, Conf_type ** sys, Conf_type ** d_sys, MC_type ** mc_step, MC_type ** d_mc_step){
  
  Conf_type * sys_temp;
  Conf_type * d_sys_temp;
  
  MC_type * mc_temp;
  MC_type * d_mc_temp;
  
  double nrg_temp; 
  
  double * point_double;
  
  open_rng(seed);

#ifdef REPLICA_EXCHANGE
  
  for(int ind=0; ind<NPT-1; ind++){
    
    clock->n_attemp_exchange[ind]++;
    
    double betaDE=(clock->nrg[ind+1]-clock->nrg[ind])/temp[ind]+(clock->nrg[ind]-clock->nrg[ind+1])/temp[ind+1];
    
    if(betaDE<0){
      
      clock->acc_rate_exchange[ind]++;
      
      // exchange ENERGY on HOST 
      nrg_temp = clock->nrg[ind+1];
      clock->nrg[ind+1] = clock->nrg[ind];
      clock->nrg[ind] = nrg_temp;
      
      // exchange ENERGY on DEVICE
      
      cudaMemcpy(&point_double,&(d_clock->nrg),sizeof(double *),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&nrg_temp,&(point_double[ind+1]),sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(&(point_double[ind+1]),&(point_double[ind]),sizeof(double),cudaMemcpyDeviceToDevice);
      cudaMemcpy(&(point_double[ind]),&nrg_temp,sizeof(double),cudaMemcpyHostToDevice);
      
      // exchange HOST struct pointer MC_type
      mc_temp = mc_step[ind+1];
      mc_step[ind+1] = mc_step[ind];
      mc_step[ind] = mc_temp;
      
      // exchange DEV struct pointer MC_type
      d_mc_temp = d_mc_step[ind+1];
      d_mc_step[ind+1] = d_mc_step[ind];
      d_mc_step[ind] = d_mc_temp;
      
      // exchange HOST struct pointer CONF_type
      sys_temp = sys[ind+1];
      sys[ind+1] = sys[ind];
      sys[ind] = sys_temp;
      
      // exchange DEV struct pointer CONF_type
      d_sys_temp = d_sys[ind+1];
      d_sys[ind+1] = d_sys[ind];
      d_sys[ind] = d_sys_temp;
      
      // set TEMPERATURE to the correct value HOST 
      sys[ind]->T = temp[ind];
      sys[ind+1]->T = temp[ind+1];
      mc_step[ind]->T = temp[ind];
      mc_step[ind+1]->T = temp[ind+1];
      
      // set TEMPERATURE to the correct value on DEV 
      cudaMemcpy(&(d_mc_step[ind]->T),&(temp[ind]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_mc_step[ind+1]->T),&(temp[ind+1]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_sys[ind]->T),&(temp[ind]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_sys[ind+1]->T),&(temp[ind+1]),sizeof(double),cudaMemcpyHostToDevice);
      
      
    }else if(rand_double() < exp(-betaDE)){
      
      clock->acc_rate_exchange[ind]++;
      
      // exchange ENERGY on HOST
      nrg_temp = clock->nrg[ind+1];
      clock->nrg[ind+1] = clock->nrg[ind];
      clock->nrg[ind] = nrg_temp;
      
      // exchange ENERGY on DEVICE
      cudaMemcpy(&point_double,&(d_clock->nrg),sizeof(double *),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&nrg_temp,&(point_double[ind+1]),sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(&(point_double[ind+1]),&(point_double[ind]),sizeof(double),cudaMemcpyDeviceToDevice);
      cudaMemcpy(&(point_double[ind]),&nrg_temp,sizeof(double),cudaMemcpyHostToDevice);
      
      // exchange HOST struct pointer MC_type
      mc_temp = mc_step[ind+1];
      mc_step[ind+1] = mc_step[ind];
      mc_step[ind] = mc_temp;
      
      // exchange DEV struct pointer MC_type
      d_mc_temp = d_mc_step[ind+1];
      d_mc_step[ind+1] = d_mc_step[ind];
      d_mc_step[ind] = d_mc_temp;
      
      // exchange HOST struct pointer CONF_type
      sys_temp = sys[ind+1];
      sys[ind+1] = sys[ind];
      sys[ind] = sys_temp;
      
      // exchange DEV struct pointer CONF_type
      d_sys_temp = d_sys[ind+1];
      d_sys[ind+1] = d_sys[ind];
      d_sys[ind] = d_sys_temp;
      
      // set TEMPERATURE to the correct value HOST 
      sys[ind]->T = temp[ind];
      sys[ind+1]->T = temp[ind+1];
      mc_step[ind]->T = temp[ind];
      mc_step[ind+1]->T = temp[ind+1];
      
      // set TEMPERATURE to the correct value on DEV 
      cudaMemcpy(&(d_mc_step[ind]->T),&(temp[ind]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_mc_step[ind+1]->T),&(temp[ind+1]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_sys[ind]->T),&(temp[ind]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&(d_sys[ind+1]->T),&(temp[ind+1]),sizeof(double),cudaMemcpyHostToDevice);
      
    }
    
  }
  
#endif

  close_rng();

}


void print_configuration(int nreplica, double * temp, int ind_iter, int N, Int_type * inter, Conf_type ** sys, Conf_type **d_sys){

  spin_t * point_spin_t_x;
  spin_t * point_spin_t_y;
  double * point_pl_ene;

  char * point_identity;

  char nomeprint[60];
  sprintf(nomeprint,"config_nrep%d_iter_%d.dat",nreplica,ind_iter);
  FILE *fprint = fopen(nomeprint,"w");
  

  for(int itemp=NPT-1; itemp>-1; itemp--){
   
    cudaMemcpy(&(point_spin_t_x),&(d_sys[itemp]->xs),sizeof(spin_t *),cudaMemcpyDeviceToHost);
    cudaMemcpy(sys[itemp]->xs,point_spin_t_x,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
    
    cudaMemcpy(&(point_spin_t_y),&(d_sys[itemp]->ys),sizeof(spin_t *),cudaMemcpyDeviceToHost);
    cudaMemcpy(sys[itemp]->ys,point_spin_t_y,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
    
    cudaMemcpy(&(point_identity),&(d_sys[itemp]->identity),sizeof(spin_t *),cudaMemcpyDeviceToHost);
    cudaMemcpy(sys[itemp]->identity,point_identity,NCHAR_IDENTITY*sizeof(char),cudaMemcpyDeviceToHost);

    for(int ispin=0; ispin<N; ispin++)
      fprintf(fprint,"%g %s %12.8e %12.8e \n",temp[itemp],sys[itemp]->identity,sys[itemp]->xs[ispin],sys[itemp]->ys[ispin]);
    
    fprintf(fprint,"\n");
    
  }

  fclose(fprint);
  
  /* sprintf(nomeprint,"plaqENERGIES_nrep%d_iter_%d.dat",nreplica,ind_iter); */
  
  /* fprint = fopen(nomeprint,"w"); */

  /* cudaMemcpy(&(point_pl_ene),&(d_sys[NPT-1]->pl_ene),sizeof(double *),cudaMemcpyDeviceToHost); */
  /* cudaMemcpy(sys[NPT-1]->pl_ene,point_pl_ene,PLAQ_NUMBER*sizeof(double),cudaMemcpyDeviceToHost); */

  /* cudaMemcpy(&(point_pl_ene),&(d_sys[NPT-2]->pl_ene),sizeof(double *),cudaMemcpyDeviceToHost); */
  /* cudaMemcpy(sys[NPT-2]->pl_ene,point_pl_ene,PLAQ_NUMBER*sizeof(double),cudaMemcpyDeviceToHost); */
  
  /* for(int iplaq=0; iplaq<PLAQ_NUMBER; iplaq++) */
  /*   fprintf(fprint,"%d %d %d %d %12.8e %12.8e \n",inter->spin_index[4*iplaq], inter->spin_index[4*iplaq+1], inter->spin_index[4*iplaq+2], inter->spin_index[4*iplaq+3], sys[NPT-1]->pl_ene[iplaq], sys[NPT-2]->pl_ene[iplaq]); */
  
  /* fclose(fprint); */
  
}

__global__ void plaquette_energy(Conf_type * d_sys, Int_type * d_inter){

  int iplaq  = blockIdx.x * blockDim.x + threadIdx.x;
  int Nplaqs = d_inter->Nplaqs;

  //printf("iplaq = %d ; Nplaqs = %d \n",iplaq,Nplaqs);

  if(iplaq<Nplaqs){

    spin_t x[4],y[4];
    
    for(int i=0;i<4;i++){

      int ispin=d_inter->spin_index[4*iplaq+i];
      
      x[i]=d_sys->xs[ispin];
      y[i]=d_sys->ys[ispin];   

    }
    
    d_sys->pl_ene[iplaq] = - d_inter->J[iplaq] *(+x[0]*x[1]*x[2]*x[3] 
			        +y[0]*y[1]*y[2]*y[3] 
			        -x[0]*x[1]*y[2]*y[3] 
			        -y[0]*y[1]*x[2]*x[3] 
			        +x[0]*y[1]*x[2]*y[3] 
			        +y[0]*x[1]*y[2]*x[3] 
			        +x[0]*y[1]*y[2]*x[3] 
			        +y[0]*x[1]*x[2]*y[3]);

  }

  __syncthreads();

}


void total_energy_parallel(Conf_type * d_sys, Int_type * d_inter){
  
  int dev=0;
  int Nplaqs;
  
  cudaMemcpy(&Nplaqs,&(d_inter->Nplaqs),sizeof(int),cudaMemcpyDeviceToHost);

  dim3 block (N_THREADS_1_BLOCK);
  dim3 grid (Nplaqs/N_THREADS_1_BLOCK);
  
  //printf("# Nplaqs (inside total energy) = %d \n",Nplaqs);

  plaquette_energy <<< grid, block >>> (d_sys,d_inter);
  
  cudaDeviceSynchronize();

  return;
}


void initialize_energy_plaquettes(int nreplica, Int_type * d_inter, Conf_type ** sys, Conf_type ** d_sys, Clock_type * clock, Clock_type * d_clock){

  double * point_pl_ene;
  
  for(int irep=0; irep<NPT; irep++){
    
    total_energy_parallel(d_sys[irep],d_inter); // initialize energy of each plaquette
    
    cudaMemcpy(&(point_pl_ene),&(d_sys[irep]->pl_ene),sizeof(double *),cudaMemcpyDeviceToHost);
    
    cudaMemcpy(sys[irep]->pl_ene,point_pl_ene,PLAQ_NUMBER*sizeof(double),cudaMemcpyDeviceToHost);

    clock->nrg[irep]=0;
    
    for(int iplaq=0; iplaq<PLAQ_NUMBER; iplaq++) clock->nrg[irep]+=sys[irep]->pl_ene[iplaq];
    
    printf("# REPLICA %d PT n° = %d : ene iniziale DEVICE (summed on HOST) = %g \n",nreplica,irep,clock->nrg[irep]);
    
  }
  
  cudaMemcpy(&(point_pl_ene),&(d_clock->nrg),sizeof(double *),cudaMemcpyDeviceToHost);  
  cudaMemcpy(point_pl_ene,clock->nrg,NPT*sizeof(double),cudaMemcpyDeviceToHost);  

}

void read_configuration(double * temp, int N, Conf_type ** sys, char * nome_file){

  FILE *fin;

  char input_string[150],parola[NCHAR_IDENTITY];
  char parola1[40],parola2[40],parola3[40];
  double spin_re, spin_im, spin_temp;
  
  

  if((fin=fopen(nome_file,"r"))==NULL){
    printf("impossibile aprire %s \n ",nome_file);
    exit(1);
  }else{
    printf("ho aperto per lettura %s \n",nome_file);
  }
  
  int k_spin[NPT];
  for(int i=0;i<NPT;i++) k_spin[i]=0;
  
  while ( ungetc(fgetc(fin),fin)!=EOF ) {
    
    fgets(input_string,150,fin);
    
    //printf("%s",input_string);
    
    sscanf(input_string, "%s %s %s %s ", parola1, parola, parola2, parola3);
    
    spin_temp = atof(parola1); 
    spin_re = atof(parola2); 
    spin_im = atof(parola3); 
    
    //printf("%g %12.8e %12.8e \n",spin_temp,spin_re,spin_im);
    
    for(int itemp=0;itemp<NPT;itemp++){
      
      if(fabs(spin_temp-temp[itemp])<1.e-4 && k_spin[itemp]<N){
	
    	sys[itemp]->xs[k_spin[itemp]]=spin_re;
    	sys[itemp]->ys[k_spin[itemp]]=spin_im;
	
    	//printf("%g %d %12.8e %12.8e \n",spin_temp,k_spin[itemp],sys[itemp]->xs[k_spin[itemp]],sys[itemp]->ys[k_spin[itemp]]);
	
    	k_spin[itemp]++;
      }
      
    }
    
  }
  
  fclose(fin);
  
  /* if((fin=fopen("prova_output.dat","w"))==NULL){ */
  /*   printf("impossibile aprire %s \n ","prova_output.dat"); */
  /*   exit(1); */
  /* }else{ */
  /*   printf("ho aperto per lettura %s \n","prova_output.dat"); */
  /* } */
  
  
  /* for(int itemp=0;itemp<NPT;itemp++){ */
    
  /*   for(int ispin=0; ispin<N; ispin++)	 */
  /*     fprintf(fin,"%g %d %12.8e %12.8e \n",temp[itemp],ispin,sys[itemp]->xs[ispin],sys[itemp]->ys[ispin]); */
    
  /*   fprintf(fin,"\n"); */
  /* } */
  
  /* fclose(fin); */
  
}

double compute_overlap(int N, int itemp, Conf_type ** sysA, Conf_type ** sysB){

  double overlap=0;
  double local_overlap;
  double normA,normB;

  for(int i=0; i<N; i++){

    local_overlap=sysA[itemp]->xs[i]*sysB[itemp]->xs[i]+sysA[itemp]->ys[i]*sysB[itemp]->ys[i];

    //normA=pow(sysA[itemp]->xs[i]*sysA[itemp]->xs[i]+sysA[itemp]->ys[i]*sysA[itemp]->ys[i],0.5);
    //normB=pow(sysB[itemp]->xs[i]*sysB[itemp]->xs[i]+sysB[itemp]->ys[i]*sysB[itemp]->ys[i],0.5);

    //overlap+=local_overlap/(normA*normB)/N;

    overlap+=local_overlap/N;
  }

  return overlap;

}

double compute_spectral_entropy(int itemp, int N, Conf_type ** sys){

  double entropy=0;
  double sum=0;
  double I[N];
  
  for(int k=0; k<N; k++){
    I[k]=sys[itemp]->xs[k]*sys[itemp]->xs[k]+sys[itemp]->ys[k]*sys[itemp]->ys[k];
    sum+=I[k];
  }
  
  for(int k=0; k<N; k++) I[k]=I[k]/sum;
  
  for(int k=0; k<N; k++)
    entropy+=-I[k]*log(I[k]);
  
  return exp(entropy)/N;
    
}

void compute_overlap_local(int N, int itemp, Conf_type ** sysA, Conf_type ** sysB, double * q_local){

  double overlap=0;
  
  for(int i=0; i<N; i++){
    q_local[i]=sysA[itemp]->xs[i]*sysB[itemp]->xs[i]+sysA[itemp]->ys[i]*sysB[itemp]->ys[i];
  }

}

double compute_overlap_overlap(int N,int itemp,double * q_local_A,double * q_local_B){

  double overlap=0;

  for(int i=0; i<N; i++){
    overlap+=(q_local_A[i]*q_local_B[i])/N;
  }
  
  return overlap;

}


double spin_IFO_accumulate_AV(int N, int itemp, Conf_type ** sysA, Conf_type ** sysB, Conf_type ** sysC, Conf_type ** sysD, double ** spin_overlaps_AV){
  
  
  for(int i=0; i<N; i++){
    
    /* spin_overlaps_AV[itemp][4*i+0]+=pow(sysA[itemp]->xs[i]*sysA[itemp]->xs[i]+sysA[itemp]->ys[i]*sysA[itemp]->ys[i],0.5);   */
    			               
    /* spin_overlaps_AV[itemp][4*i+1]+=pow(sysB[itemp]->xs[i]*sysB[itemp]->xs[i]+sysB[itemp]->ys[i]*sysB[itemp]->ys[i],0.5); */
    			            
    /* spin_overlaps_AV[itemp][4*i+2]+=pow(sysC[itemp]->xs[i]*sysC[itemp]->xs[i]+sysC[itemp]->ys[i]*sysC[itemp]->ys[i],0.5); */
    			            
    /* spin_overlaps_AV[itemp][4*i+3]+=pow(sysD[itemp]->xs[i]*sysD[itemp]->xs[i]+sysD[itemp]->ys[i]*sysD[itemp]->ys[i],0.5); */


    spin_overlaps_AV[itemp][4*i+0]+=sysA[itemp]->xs[i]*sysA[itemp]->xs[i]+sysA[itemp]->ys[i]*sysA[itemp]->ys[i];  
    			            
    spin_overlaps_AV[itemp][4*i+1]+=sysB[itemp]->xs[i]*sysB[itemp]->xs[i]+sysB[itemp]->ys[i]*sysB[itemp]->ys[i];
    			            
    spin_overlaps_AV[itemp][4*i+2]+=sysC[itemp]->xs[i]*sysC[itemp]->xs[i]+sysC[itemp]->ys[i]*sysC[itemp]->ys[i];
    			            
    spin_overlaps_AV[itemp][4*i+3]+=sysD[itemp]->xs[i]*sysD[itemp]->xs[i]+sysD[itemp]->ys[i]*sysD[itemp]->ys[i];
    
  }
    
}


double link_IFO_accumulate_AV(int Nplaqs, int itemp, Conf_type ** sysA, Conf_type ** sysB, Conf_type ** sysC, Conf_type ** sysD, Int_type * inter, double ** link_overlaps_AV){
  
  
  for(int i=0; i<Nplaqs; i++){
    
    double eneA=0;
    double eneB=0;
    double eneC=0;
    double eneD=0;
    
    double xa[4],ya[4];
    double xb[4],yb[4];
    double xc[4],yc[4];
    double xd[4],yd[4];

    int i0=inter->spin_index[4*i];
    int i1=inter->spin_index[4*i+1];
    int i2=inter->spin_index[4*i+2];
    int i3=inter->spin_index[4*i+3];

    xa[0]=sysA[itemp]->xs[i0];
    xa[1]=sysA[itemp]->xs[i1];
    xa[2]=sysA[itemp]->xs[i2];
    xa[3]=sysA[itemp]->xs[i3];

    ya[0]=sysA[itemp]->ys[i0];
    ya[1]=sysA[itemp]->ys[i1];
    ya[2]=sysA[itemp]->ys[i2];
    ya[3]=sysA[itemp]->ys[i3];

    /////////////////////////

    xb[0]=sysB[itemp]->xs[i0];
    xb[1]=sysB[itemp]->xs[i1];
    xb[2]=sysB[itemp]->xs[i2];
    xb[3]=sysB[itemp]->xs[i3];
    
    yb[0]=sysB[itemp]->ys[i0];
    yb[1]=sysB[itemp]->ys[i1];
    yb[2]=sysB[itemp]->ys[i2];
    yb[3]=sysB[itemp]->ys[i3];

    /////////////////////////

    xc[0]=sysC[itemp]->xs[i0];
    xc[1]=sysC[itemp]->xs[i1];
    xc[2]=sysC[itemp]->xs[i2];
    xc[3]=sysC[itemp]->xs[i3];
    
    yc[0]=sysC[itemp]->ys[i0];
    yc[1]=sysC[itemp]->ys[i1];
    yc[2]=sysC[itemp]->ys[i2];
    yc[3]=sysC[itemp]->ys[i3];

    /////////////////////////

    xd[0]=sysD[itemp]->xs[i0];
    xd[1]=sysD[itemp]->xs[i1];
    xd[2]=sysD[itemp]->xs[i2];
    xd[3]=sysD[itemp]->xs[i3];
    
    yd[0]=sysD[itemp]->ys[i0];
    yd[1]=sysD[itemp]->ys[i1];
    yd[2]=sysD[itemp]->ys[i2];
    yd[3]=sysD[itemp]->ys[i3];


    eneA = ((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3];
    
    eneB = ((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3];
    
    eneC = ((xc[0]*xc[1]-yc[0]*yc[1])*xc[2] + (yc[0]*xc[1]+xc[0]*yc[1])*yc[2])*xc[3] + ((yc[0]*yc[1]-xc[0]*xc[1])*yc[2] + (xc[0]*yc[1]+yc[0]*xc[1])*xc[2])*yc[3];
    
    eneD = ((xd[0]*xd[1]-yd[0]*yd[1])*xd[2] + (yd[0]*xd[1]+xd[0]*yd[1])*yd[2])*xd[3] + ((yd[0]*yd[1]-xd[0]*xd[1])*yd[2] + (xd[0]*yd[1]+yd[0]*xd[1])*xd[2])*yd[3];
    

    link_overlaps_AV[itemp][4*i+0]+=pow(eneA*eneA,0.5);  
    				               
    link_overlaps_AV[itemp][4*i+1]+=pow(eneB*eneB,0.5);
    				               
    link_overlaps_AV[itemp][4*i+2]+=pow(eneC*eneC,0.5);
    				               
    link_overlaps_AV[itemp][4*i+3]+=pow(eneD*eneD,0.5);

  }

}


double compute_link_overlap(int Nplaqs, int itemp, Conf_type ** sysA, Conf_type ** sysB, Int_type * inter){

  double overlap=0;
  
  
  for(int i=0; i<Nplaqs; i++){
    
    double xa[4],ya[4];
    double xb[4],yb[4];

    int i0=inter->spin_index[4*i];
    int i1=inter->spin_index[4*i+1];
    int i2=inter->spin_index[4*i+2];
    int i3=inter->spin_index[4*i+3];

    xa[0]=sysA[itemp]->xs[i0];
    xa[1]=sysA[itemp]->xs[i1];
    xa[2]=sysA[itemp]->xs[i2];
    xa[3]=sysA[itemp]->xs[i3];

    ya[0]=sysA[itemp]->ys[i0];
    ya[1]=sysA[itemp]->ys[i1];
    ya[2]=sysA[itemp]->ys[i2];
    ya[3]=sysA[itemp]->ys[i3];

    xb[0]=sysB[itemp]->xs[i0];
    xb[1]=sysB[itemp]->xs[i1];
    xb[2]=sysB[itemp]->xs[i2];
    xb[3]=sysB[itemp]->xs[i3];
    
    yb[0]=sysB[itemp]->ys[i0];
    yb[1]=sysB[itemp]->ys[i1];
    yb[2]=sysB[itemp]->ys[i2];
    yb[3]=sysB[itemp]->ys[i3];
    
    // double eneA = inter->J[i]*(((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3]);
    // double eneB = inter->J[i]*(((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3]);
  
    double eneA = (((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3]);
    double eneB = (((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3]);

    //overlap+=Nplaqs*eneA*eneB/(4*Nplaqs*N);

    overlap+=eneA*eneB/(2*Nplaqs);

    //eneA+=(inter->J[i]*(((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3]))/N;
    //eneB+=(inter->J[i]*(((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3]))/N;
  
  }

  //overlap=eneA*eneB;

  return overlap;

}

/* void compute_IFO_energies(int Nplaqs, int itemp, Conf_type ** sysA, Conf_type ** sysB, Conf_type ** sysC, Conf_type ** sysD, Int_type * inter, int * icount_energy_IFO, double ** overlaps_energy_IFO){ */

/*   double ENE_av; */
/*   double dE[4]; */
/*   double norm_dE[4]; */
/*   double ENE_sysA,ENE_sysB,ENE_sysC,ENE_sysD; */
/*   double C_01,C_02,C_03,C_12,C_13,C_23; */

/*   ENE_av=0; */

/*   ENE_sysA=0; */
/*   ENE_sysB=0; */
/*   ENE_sysC=0; */
/*   ENE_sysD=0; */

/*   for(int k=0; k<Nplaqs; k++){ */
    
/*     double xa[4],ya[4]; */
/*     double xb[4],yb[4]; */
/*     double xc[4],yc[4]; */
/*     double xd[4],yd[4]; */

/*     int i0=inter->spin_index[4*k]; */
/*     int i1=inter->spin_index[4*k+1]; */
/*     int i2=inter->spin_index[4*k+2]; */
/*     int i3=inter->spin_index[4*k+3]; */

/*     xa[0]=sysA[itemp]->xs[i0]; */
/*     xa[1]=sysA[itemp]->xs[i1]; */
/*     xa[2]=sysA[itemp]->xs[i2]; */
/*     xa[3]=sysA[itemp]->xs[i3]; */

/*     ya[0]=sysA[itemp]->ys[i0]; */
/*     ya[1]=sysA[itemp]->ys[i1]; */
/*     ya[2]=sysA[itemp]->ys[i2]; */
/*     ya[3]=sysA[itemp]->ys[i3]; */

/*     ///////////////////////// */

/*     xb[0]=sysB[itemp]->xs[i0]; */
/*     xb[1]=sysB[itemp]->xs[i1]; */
/*     xb[2]=sysB[itemp]->xs[i2]; */
/*     xb[3]=sysB[itemp]->xs[i3]; */
    
/*     yb[0]=sysB[itemp]->ys[i0]; */
/*     yb[1]=sysB[itemp]->ys[i1]; */
/*     yb[2]=sysB[itemp]->ys[i2]; */
/*     yb[3]=sysB[itemp]->ys[i3]; */

/*     ///////////////////////// */

/*     xc[0]=sysC[itemp]->xs[i0]; */
/*     xc[1]=sysC[itemp]->xs[i1]; */
/*     xc[2]=sysC[itemp]->xs[i2]; */
/*     xc[3]=sysC[itemp]->xs[i3]; */
    
/*     yc[0]=sysC[itemp]->ys[i0]; */
/*     yc[1]=sysC[itemp]->ys[i1]; */
/*     yc[2]=sysC[itemp]->ys[i2]; */
/*     yc[3]=sysC[itemp]->ys[i3]; */

/*     ///////////////////////// */

/*     xd[0]=sysD[itemp]->xs[i0]; */
/*     xd[1]=sysD[itemp]->xs[i1]; */
/*     xd[2]=sysD[itemp]->xs[i2]; */
/*     xd[3]=sysD[itemp]->xs[i3]; */
    
/*     yd[0]=sysD[itemp]->ys[i0]; */
/*     yd[1]=sysD[itemp]->ys[i1]; */
/*     yd[2]=sysD[itemp]->ys[i2]; */
/*     yd[3]=sysD[itemp]->ys[i3]; */


/*     ENE_sysA+=-inter->J[k]*(((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3]); */
    
/*     ENE_sysB+=-inter->J[k]*(((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3]); */
    
/*     ENE_sysC+=-inter->J[k]*(((xc[0]*xc[1]-yc[0]*yc[1])*xc[2] + (yc[0]*xc[1]+xc[0]*yc[1])*yc[2])*xc[3] + ((yc[0]*yc[1]-xc[0]*xc[1])*yc[2] + (xc[0]*yc[1]+yc[0]*xc[1])*xc[2])*yc[3]); */
    
/*     ENE_sysD+=-inter->J[k]*(((xd[0]*xd[1]-yd[0]*yd[1])*xd[2] + (yd[0]*xd[1]+xd[0]*yd[1])*yd[2])*xd[3] + ((yd[0]*yd[1]-xd[0]*xd[1])*yd[2] + (xd[0]*yd[1]+yd[0]*xd[1])*xd[2])*yd[3]); */
    
/*   } */
  
/*   ENE_av=(ENE_sysA+ENE_sysB+ENE_sysC+ENE_sysD)/4; */
  
/*   dE[0]= ENE_sysA-ENE_av; */
/*   dE[1]= ENE_sysB-ENE_av; */
/*   dE[2]= ENE_sysC-ENE_av; */
/*   dE[3]= ENE_sysD-ENE_av; */
  
    
/*   for(int i=0; i<4; i++){ */
/*     norm_dE[i]=pow(dE[i]*dE[i],0.5); */
/*   } */
  
/*   C_01=dE[0]*dE[1];///(norm_dE[0]*norm_dE[1]); */
/*   C_02=dE[0]*dE[2];///(norm_dE[0]*norm_dE[2]); */
/*   C_03=dE[0]*dE[3];///(norm_dE[0]*norm_dE[3]); */
/*   C_12=dE[1]*dE[2];///(norm_dE[1]*norm_dE[2]); */
/*   C_13=dE[1]*dE[3];///(norm_dE[1]*norm_dE[3]); */
/*   C_23=dE[2]*dE[3];///(norm_dE[2]*norm_dE[3]); */

/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_01; */
/*   icount_energy_IFO[itemp]++; */
  
/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_02; */
/*   icount_energy_IFO[itemp]++; */
  
/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_03; */
/*   icount_energy_IFO[itemp]++; */
  
/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_12; */
/*   icount_energy_IFO[itemp]++; */
  
/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_13; */
/*   icount_energy_IFO[itemp]++; */
  
/*   overlaps_energy_IFO[itemp][icount_energy_IFO[itemp]] = C_23; */
/*   icount_energy_IFO[itemp]++; */
  
/* } */


void compute_IFO_plaquettes(int Nplaqs, int itemp, Conf_type ** sysA, Conf_type ** sysB, Conf_type ** sysC, Conf_type ** sysD, Int_type * inter, int * icount_link_IFO, double ** overlaps_link_IFO, double ** link_overlaps_AV){

  double I_av[Nplaqs];
  double dI[4][Nplaqs];
  double norm_dI[4];
  double Emod_sys0=0,Emod_sys1=0,Emod_sys2=0,Emod_sys3=0;
  double C_01=0,C_02=0,C_03=0,C_12=0,C_13=0,C_23=0;

  
  for(int i=0; i<Nplaqs; i++){
    
    double xa[4],ya[4];
    double xb[4],yb[4];
    double xc[4],yc[4];
    double xd[4],yd[4];

    int i0=inter->spin_index[4*i];
    int i1=inter->spin_index[4*i+1];
    int i2=inter->spin_index[4*i+2];
    int i3=inter->spin_index[4*i+3];

    xa[0]=sysA[itemp]->xs[i0];
    xa[1]=sysA[itemp]->xs[i1];
    xa[2]=sysA[itemp]->xs[i2];
    xa[3]=sysA[itemp]->xs[i3];

    ya[0]=sysA[itemp]->ys[i0];
    ya[1]=sysA[itemp]->ys[i1];
    ya[2]=sysA[itemp]->ys[i2];
    ya[3]=sysA[itemp]->ys[i3];

    /////////////////////////

    xb[0]=sysB[itemp]->xs[i0];
    xb[1]=sysB[itemp]->xs[i1];
    xb[2]=sysB[itemp]->xs[i2];
    xb[3]=sysB[itemp]->xs[i3];
    
    yb[0]=sysB[itemp]->ys[i0];
    yb[1]=sysB[itemp]->ys[i1];
    yb[2]=sysB[itemp]->ys[i2];
    yb[3]=sysB[itemp]->ys[i3];

    /////////////////////////

    xc[0]=sysC[itemp]->xs[i0];
    xc[1]=sysC[itemp]->xs[i1];
    xc[2]=sysC[itemp]->xs[i2];
    xc[3]=sysC[itemp]->xs[i3];
    
    yc[0]=sysC[itemp]->ys[i0];
    yc[1]=sysC[itemp]->ys[i1];
    yc[2]=sysC[itemp]->ys[i2];
    yc[3]=sysC[itemp]->ys[i3];

    /////////////////////////

    xd[0]=sysD[itemp]->xs[i0];
    xd[1]=sysD[itemp]->xs[i1];
    xd[2]=sysD[itemp]->xs[i2];
    xd[3]=sysD[itemp]->xs[i3];
    
    yd[0]=sysD[itemp]->ys[i0];
    yd[1]=sysD[itemp]->ys[i1];
    yd[2]=sysD[itemp]->ys[i2];
    yd[3]=sysD[itemp]->ys[i3];

    link_overlaps_AV;

    Emod_sys0=((xa[0]*xa[1]-ya[0]*ya[1])*xa[2] + (ya[0]*xa[1]+xa[0]*ya[1])*ya[2])*xa[3] + ((ya[0]*ya[1]-xa[0]*xa[1])*ya[2] + (xa[0]*ya[1]+ya[0]*xa[1])*xa[2])*ya[3];
    Emod_sys1=((xb[0]*xb[1]-yb[0]*yb[1])*xb[2] + (yb[0]*xb[1]+xb[0]*yb[1])*yb[2])*xb[3] + ((yb[0]*yb[1]-xb[0]*xb[1])*yb[2] + (xb[0]*yb[1]+yb[0]*xb[1])*xb[2])*yb[3];
    Emod_sys2=((xc[0]*xc[1]-yc[0]*yc[1])*xc[2] + (yc[0]*xc[1]+xc[0]*yc[1])*yc[2])*xc[3] + ((yc[0]*yc[1]-xc[0]*xc[1])*yc[2] + (xc[0]*yc[1]+yc[0]*xc[1])*xc[2])*yc[3];
    Emod_sys3=((xd[0]*xd[1]-yd[0]*yd[1])*xd[2] + (yd[0]*xd[1]+xd[0]*yd[1])*yd[2])*xd[3] + ((yd[0]*yd[1]-xd[0]*xd[1])*yd[2] + (xd[0]*yd[1]+yd[0]*xd[1])*xd[2])*yd[3];
    
    Emod_sys0=pow(Emod_sys0*Emod_sys0,0.5);
    Emod_sys1=pow(Emod_sys1*Emod_sys1,0.5);
    Emod_sys2=pow(Emod_sys2*Emod_sys2,0.5);
    Emod_sys3=pow(Emod_sys3*Emod_sys3,0.5);

    /* dI[0][i]= Emod_sys0-link_overlaps_AV[itemp][4*i+0]; */
    /* dI[1][i]= Emod_sys1-link_overlaps_AV[itemp][4*i+1]; */
    /* dI[2][i]= Emod_sys2-link_overlaps_AV[itemp][4*i+2]; */
    /* dI[3][i]= Emod_sys3-link_overlaps_AV[itemp][4*i+3]; */

    double mean_AV=(link_overlaps_AV[itemp][4*i+0]+link_overlaps_AV[itemp][4*i+1]+link_overlaps_AV[itemp][4*i+2]+link_overlaps_AV[itemp][4*i+3])/4;

    dI[0][i]= Emod_sys0-mean_AV;
    dI[1][i]= Emod_sys1-mean_AV;
    dI[2][i]= Emod_sys2-mean_AV;
    dI[3][i]= Emod_sys3-mean_AV;
    
  }
  
  for(int i=0; i<4; i++){
    norm_dI[i]=0;
    for(int k=0; k<Nplaqs; k++) norm_dI[i]+=dI[i][k]*dI[i][k];
    norm_dI[i]=pow(norm_dI[i],0.5);
  }
  
  for(int k=0; k<Nplaqs; k++) C_01+=(dI[0][k]*dI[1][k]);
  for(int k=0; k<Nplaqs; k++) C_02+=(dI[0][k]*dI[2][k]);
  for(int k=0; k<Nplaqs; k++) C_03+=(dI[0][k]*dI[3][k]);
  for(int k=0; k<Nplaqs; k++) C_12+=(dI[1][k]*dI[2][k]);
  for(int k=0; k<Nplaqs; k++) C_13+=(dI[1][k]*dI[3][k]);
  for(int k=0; k<Nplaqs; k++) C_23+=(dI[2][k]*dI[3][k]);
  
  C_01=C_01/(norm_dI[0]*norm_dI[1]);
  C_02=C_02/(norm_dI[0]*norm_dI[2]);
  C_03=C_03/(norm_dI[0]*norm_dI[3]);
  C_12=C_12/(norm_dI[1]*norm_dI[2]);
  C_13=C_13/(norm_dI[1]*norm_dI[3]);
  C_23=C_23/(norm_dI[2]*norm_dI[3]);
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_01;
  icount_link_IFO[itemp]++;
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_02;
  icount_link_IFO[itemp]++;
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_03;
  icount_link_IFO[itemp]++;
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_12;
  icount_link_IFO[itemp]++;
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_13;
  icount_link_IFO[itemp]++;
  
  overlaps_link_IFO[itemp][icount_link_IFO[itemp]] = C_23;
  icount_link_IFO[itemp]++;
  
}


void compute_IFO(int N, int itemp, Conf_type ** sys0, Conf_type ** sys1, Conf_type ** sys2, Conf_type ** sys3, int * icount_IFO, double ** overlaps_IFO, double ** spin_overlaps_AV){

  double I_av[N];
  double dI[4][N];
  double norm_dI[4];
  double a2_sys0[N],a2_sys1[N],a2_sys2[N],a2_sys3[N];
  double C_01=0,C_02=0,C_03=0,C_12=0,C_13=0,C_23=0;

  for(int k=0; k<N; k++){
    
    /* a2_sys0[k]=pow(sys0[itemp]->xs[k]*sys0[itemp]->xs[k]+sys0[itemp]->ys[k]*sys0[itemp]->ys[k],0.5); */
    /* a2_sys1[k]=pow(sys1[itemp]->xs[k]*sys1[itemp]->xs[k]+sys1[itemp]->ys[k]*sys1[itemp]->ys[k],0.5); */
    /* a2_sys2[k]=pow(sys2[itemp]->xs[k]*sys2[itemp]->xs[k]+sys2[itemp]->ys[k]*sys2[itemp]->ys[k],0.5); */
    /* a2_sys3[k]=pow(sys3[itemp]->xs[k]*sys3[itemp]->xs[k]+sys3[itemp]->ys[k]*sys3[itemp]->ys[k],0.5); */

    a2_sys0[k]=sys0[itemp]->xs[k]*sys0[itemp]->xs[k]+sys0[itemp]->ys[k]*sys0[itemp]->ys[k];
    a2_sys1[k]=sys1[itemp]->xs[k]*sys1[itemp]->xs[k]+sys1[itemp]->ys[k]*sys1[itemp]->ys[k];
    a2_sys2[k]=sys2[itemp]->xs[k]*sys2[itemp]->xs[k]+sys2[itemp]->ys[k]*sys2[itemp]->ys[k];
    a2_sys3[k]=sys3[itemp]->xs[k]*sys3[itemp]->xs[k]+sys3[itemp]->ys[k]*sys3[itemp]->ys[k];
   
    /* dI[0][k]= a2_sys0[k]-spin_overlaps_AV[itemp][4*k+0]; */
    /* dI[1][k]= a2_sys1[k]-spin_overlaps_AV[itemp][4*k+1]; */
    /* dI[2][k]= a2_sys2[k]-spin_overlaps_AV[itemp][4*k+2]; */
    /* dI[3][k]= a2_sys3[k]-spin_overlaps_AV[itemp][4*k+3]; */

    //double mean_AV=(spin_overlaps_AV[itemp][4*k+0]+spin_overlaps_AV[itemp][4*k+1]+spin_overlaps_AV[itemp][4*k+2]+spin_overlaps_AV[itemp][4*k+3])/4;
    
    dI[0][k]= a2_sys0[k]-spin_overlaps_AV[itemp][4*k+0];
    dI[1][k]= a2_sys1[k]-spin_overlaps_AV[itemp][4*k+1];
    dI[2][k]= a2_sys2[k]-spin_overlaps_AV[itemp][4*k+2];
    dI[3][k]= a2_sys3[k]-spin_overlaps_AV[itemp][4*k+3];
    
  }

  for(int i=0; i<4; i++){
    norm_dI[i]=0;
    for(int k=0; k<N; k++) norm_dI[i]+=dI[i][k]*dI[i][k];
    norm_dI[i]=pow(norm_dI[i],0.5);
  }

  
  for(int k=0; k<N; k++) C_01+=(dI[0][k]*dI[1][k]);
  for(int k=0; k<N; k++) C_02+=(dI[0][k]*dI[2][k]);
  for(int k=0; k<N; k++) C_03+=(dI[0][k]*dI[3][k]);
  for(int k=0; k<N; k++) C_12+=(dI[1][k]*dI[2][k]);
  for(int k=0; k<N; k++) C_13+=(dI[1][k]*dI[3][k]);
  for(int k=0; k<N; k++) C_23+=(dI[2][k]*dI[3][k]);

  C_01=C_01/(norm_dI[0]*norm_dI[1]);
  C_02=C_02/(norm_dI[0]*norm_dI[2]);
  C_03=C_03/(norm_dI[0]*norm_dI[3]);
  C_12=C_12/(norm_dI[1]*norm_dI[2]);
  C_13=C_13/(norm_dI[1]*norm_dI[3]);
  C_23=C_23/(norm_dI[2]*norm_dI[3]);
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_01;
  icount_IFO[itemp]++;
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_02;
  icount_IFO[itemp]++;
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_03;
  icount_IFO[itemp]++;
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_12;
  icount_IFO[itemp]++;
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_13;
  icount_IFO[itemp]++;
  
  overlaps_IFO[itemp][icount_IFO[itemp]] = C_23;
  icount_IFO[itemp]++;
  
}


void compute_phases(int itemp, int N, Conf_type ** sys, double * phi){
  
  int i;
  double x,y;

  double pigreco=3.141592653589793238462643383279;

  for(i=0;i<N;i++){
    
    x=sys[itemp]->xs[i];
    y=sys[itemp]->ys[i];

    if(x>0){
      
      phi[i]=atan(y/x);
    
    }else{
      
      phi[i]=atan(y/x)+pigreco;
      
    }
    
  }
  
  for(i=0;i<N;i++){
    if(phi[i]<0)
      phi[i]=2.*pigreco+phi[i];
  }
  
}

double compute_cos_deltaphi_plaquettes(int itemp, int Nplaqs, double * phi, Int_type * inter, double * deltaphi){

  double pphi[4];
  double phi_sum[2];
  double delta_phi;

  double sum=0;

  for(int iplaq=0;iplaq<Nplaqs;iplaq++){

    for(int i=0;i<4;i++)
      pphi[i] = phi[inter->spin_index[4*iplaq+i]];
    
    phi_sum[0] = pphi[0]+pphi[1];
    phi_sum[1] = pphi[2]+pphi[3];
    
    //phi_sum[0] = phi_sum[0]-twopi*(int)floor(phi_sum[0]/twopi);
    //phi_sum[1] = phi_sum[1]-twopi*(int)floor(phi_sum[1]/twopi);
    
    delta_phi = phi_sum[0]-phi_sum[1];

    delta_phi = cos(delta_phi-twopi*rint(delta_phi/twopi));

    sum+=delta_phi/Nplaqs;

  }

  return sum;

}

double compute_cos_phi_spins(int itemp, int N, double * phi){

  double sum=0;

  for(int i=0;i<N;i++){

    sum+=cos(phi[i])/N;

  }

  return sum;

}

double compute_modulus_spins(int itemp, int N, Conf_type ** sys){

  double sum=0;
  
  for(int i=0;i<N;i++){
    sum+=pow(sys[itemp]->xs[i]*sys[itemp]->xs[i]+sys[itemp]->ys[i]*sys[itemp]->ys[i],0.5)/N;
  }
  
  return sum;

}


void trajectory_MonteCarlo(FILE * fout,
			   int ind_iter,
			   double * time_MC_sweep,
			   int nT, 
			   int seed,
			   int N, 
			   int Nplaqs, 
			   Clock_type * d_clock, 
			   Clock_type * clock, 
			   Conf_type ** d_sys, 
			   Conf_type ** sys, 
			   Int_type * d_inter, 
			   Int_type * inter, 
			   MC_type ** d_mc_step, 
			   MC_type ** mc_step){
  

  //////////////////////////////////////////////////////////////////////////
  ////  CUDA RANDOM GENERATOR INITIALIZATION////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  double time_MC_sweep_START=0;
  
  double mbeta[nT];
  for(int i=0; i<nT; i++) mbeta[i] = -1./mc_step[i]->T;
  
  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  
  double * point_double;
  spin_t * point_spin_t_x;
  spin_t * point_spin_t_y;
  int * point_int;
  float * point_float;
  

  for(int i=0; i<nT; i++){
    randomize_spin_coppie_host(seed,N,mc_step[i]->coppie);
    cudaMemcpy(&point_int,&(d_mc_step[i]->coppie),sizeof(int *),cudaMemcpyDeviceToHost);
    cudaMemcpy(point_int,mc_step[i]->coppie,N*sizeof(int),cudaMemcpyHostToDevice);
  }


  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  
  dim3 block (N_THREADS_1_BLOCK);
  dim3 grid ((Nplaqs+block.x-1)/block.x);
  
  // printf("N_THREADS_1_BLOCK = %d \n",N_THREADS_1_BLOCK);
  
  
  for(int i_montecarlo=0;i_montecarlo<NSTEP;i_montecarlo++){
    
    //////////////////////////////////////////////////////////////////////////////////
    ////// ---------- RESHUFFLE SPIN ORDER WITHIN COUPLES --------------------- ////// 
    //////////////////////////////////////////////////////////////////////////////////
    
    for(int i=0; i<nT; i++){
      randomize_spin_coppie_host(seed+1,N,mc_step[i]->coppie);
      cudaMemcpy(&point_int,&(d_mc_step[i]->coppie),sizeof(int *),cudaMemcpyDeviceToHost);
      cudaMemcpy(point_int,mc_step[i]->coppie,N*sizeof(int),cudaMemcpyHostToDevice);
    }
    
    ///////////////////////////////////////////////
    
    time_MC_sweep_START=cpuSecond();
    
    MCstep_disorderp4_placchette_parallel(seed+2,nT,mbeta,d_clock,clock,d_sys,sys,d_inter,inter,d_mc_step,mc_step);
    
    (*time_MC_sweep)+=(cpuSecond()-time_MC_sweep_START);
   
    //printf("MC steps = %d \n",NSTEP*ind_iter+i_montecarlo);

#ifdef SINGLE_REPLICA_OUTPUT

    if((NSTEP*ind_iter+i_montecarlo)%PRINT_FREQUENCY==0){
    
      cudaMemcpy(&(point_double),&(d_clock->nrg),sizeof(double *),cudaMemcpyDeviceToHost);
      cudaMemcpy(clock->nrg,point_double,NPT*sizeof(double),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&(point_int),&(d_clock->acc_rate),sizeof(int *),cudaMemcpyDeviceToHost);
      cudaMemcpy(clock->acc_rate,point_int,NPT*sizeof(int),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&(point_int),&(d_clock->n_attemp),sizeof(int *),cudaMemcpyDeviceToHost);
      cudaMemcpy(clock->n_attemp,point_int,NPT*sizeof(int),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&(point_spin_t_x),&(d_sys[0]->xs),sizeof(spin_t *),cudaMemcpyDeviceToHost);
      cudaMemcpy(sys[0]->xs,point_spin_t_x,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
      
      cudaMemcpy(&(point_spin_t_y),&(d_sys[0]->ys),sizeof(spin_t *),cudaMemcpyDeviceToHost);
      cudaMemcpy(sys[0]->ys,point_spin_t_y,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
      
      double magn_x=0;
      double magn_y=0;
      
      for(int i_spin=0;i_spin<N;i_spin++){
    	magn_x+=sys[0]->xs[i_spin];
    	magn_y+=sys[0]->ys[i_spin];
      }
      
      fout = fopen("output.dat","a");
      
      fprintf(fout,"%g %d %10.4e %10.4e %10.4e %g %8.4e %8.4e %8.4e %8.4e %8.4e %8.4e \n",
    	      sys[0]->T,                    // 1
    	      NSTEP*ind_iter+i_montecarlo,  // 2
    	      clock->nrg[0]/N,              // 3
    	      magn_x/N,                     // 4
    	      magn_y/N,                     // 5
    	      (double)clock->acc_rate[0]/clock->n_attemp[0],    //  6
    	      clock->prof_time[0]/clock->n_attemp[0],           //  7
    	      clock->prof_time[1]/clock->n_attemp[0],           //  8
    	      clock->prof_time[2]/clock->n_attemp[0],           //  9
    	      clock->prof_time[3]/clock->n_attemp[0],           //  10
    	      clock->prof_time[4]/clock->n_attemp[0],           //  11
    	      (*time_MC_sweep)/(NSTEP*ind_iter+i_montecarlo));  //  12
      
      fclose(fout);
      
      printf("%g %d %10.4e %10.4e %10.4e %g %8.4e %8.4e %8.4e %8.4e %8.4e %8.4e \n",
      	     sys[0]->T,                   // 1
      	     NSTEP*ind_iter+i_montecarlo, // 2
      	     clock->nrg[0]/N,             // 3
      	     magn_x/N,                    // 4
      	     magn_y/N,                    // 5
      	     (double)clock->acc_rate[0]/clock->n_attemp[0],    //  6
      	     clock->prof_time[0]/clock->n_attemp[0],           //  7
      	     clock->prof_time[1]/clock->n_attemp[0],           //  8
      	     clock->prof_time[2]/clock->n_attemp[0],           //  9
      	     clock->prof_time[3]/clock->n_attemp[0],           //  10
      	     clock->prof_time[4]/clock->n_attemp[0],           //  11
      	     (*time_MC_sweep)/(NSTEP*ind_iter+i_montecarlo));  //  12
      
    }
    
#endif
    
    //////////////////////////////////////////////////////////////////////    
    ////// ---------- END OF MC STEP INSTRUCTIONS ----------------- ////// 
    //////////////////////////////////////////////////////////////////////    
    
  }
  
}
