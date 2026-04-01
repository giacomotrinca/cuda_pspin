// ------------ GIACOMO---- 09-2017 ---//

typedef struct plaqs{

  int spin_index[4];

  double J;

} Plaqs_type;


typedef struct interactions{

  int Nplaqs;
  int * spin_index;
  double * J;

} Int_type;


typedef struct config{
  
  int N;
  double T; // TO BE UPDATED AT REPLICA EXCHANGE 

  char * identity; 

  double * gain;
  double * pl_ene;
  double * pl_ene_new;
  double * pl_de;
  double * pl_de_block; 
  
  spin_t * xs;
  spin_t * ys;
  
} Conf_type;


typedef struct monte_carlo{

  double T; // TO BE UPDATED AT REPLICA EXCHANGE 
  int flag;
  int icoppia;
  int Ncoppie;
  
  // char * identity; 

  int * coppie; 
  float * rnumbers_coppie;

  spin_t * nx1;
  spin_t * nx2; 
  
  spin_t * ny1;
  spin_t * ny2;

  double * alpha_rand;
  double * phi1_rand;
  double * phi2_rand;
  
} MC_type;


typedef struct clocks{

  double * prof_time;
  double * nrg; // TO BE EXCHANGED AT REPLICA EXCHANGE
  
  int n_attemp_exchange[NTJUMPS];
  int acc_rate_exchange[NTJUMPS];

  int * acc_rate;
  int * n_attemp;
  
} Clock_type;


typedef struct rep{

  double * spin;

} Replica_type;

