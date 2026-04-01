#include<assert.h>
//#define twopi 6.283185307179586



// inits the coupling matrix with Gaussian random entries
// data[] is of dimension #neighbors=2.#bonds
// in such a way that both neighbors of the same bond have the same entry
// Box-Muller transform used
///////////////////////////////////////////////////////////////////////


void randomGaussianCouplingsp4_giacomo(int Nplaqs, vector<Plaqs_type> &placchette, spin_t avrg, spin_t sigma, int seed){
  
  open_rng(seed);
  
  double radicedi2=sqrt(2.);
  double ran1,ran2;
  
  for(int i=0; i<Nplaqs; i++){
    
    ran1=rand_double(); 
    ran2=rand_double();
    
    placchette[i].J = sin(ran2*twopi)*radicedi2*sigma*sqrt(-log(ran1))+avrg ;
    
  }
  
  close_rng();
  
}


void randomGaussianCouplingsp4_v2(int NbQuads,int *quads,int *cumNbq,double* data,spin_t avrg,spin_t sigma,int seed) 
{
  open_rng(seed);
  double radicedi2=sqrt(2.);
  double ran1,ran2;
  
  int * flag = (int *) malloc(NbQuads*sizeof(int));
  for(int j=0;j<NbQuads;j++)  
    flag[j]=1;   // flag[q] is 1 is if data[q] is still to be assigned
  
  
  for (int q=0;q<NbQuads;q++)
    {
      if(flag[q])
	{
	  //RNGeneration
	  /////////////////////////////
	  ran1=rand_double(); 
	  ran2=rand_double();
	  double J=sin(ran2*twopi)*radicedi2*sigma*sqrt(-log(ran1))+avrg;
	  /////////////////////////////
	  
	  data[q]=J;
	  flag[q]=0;
	  
	  int m1=quads[4*q];
	  int m2=quads[4*q+1];
	  int m3=quads[4*q+2];
	  int m4=quads[4*q+3];

	  /////////////////////////////////////////////////////////////////////////////////////	  
	  // now we assign J to all the time the quadruplet appears in quads

	  { // for m1  // this is not stricly necessary, since we start from q=0 increasing, so cumNbq[m1]+q2 = q (but better safe than sorry)
	    int d=cumNbq[m1+1]-cumNbq[m1];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m1]+q2)]==m1) 
			   && (quads[4*(cumNbq[m1]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m1]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m1]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m1]+q2]=J;
		  flag[cumNbq[m1]+q2]=0;
		}
	      }
	  }

	 
	  { // idem m2
	    int d=cumNbq[m2+1]-cumNbq[m2];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m2]+q2)]==m1) 
			   && (quads[4*(cumNbq[m2]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m2]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m2]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m2]+q2]=J;
		  flag[cumNbq[m2]+q2]=0;
		}
	      }
	  }
	  
	  {
	    //idem m3
	    int d=cumNbq[m3+1]-cumNbq[m3];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m3]+q2)]==m1) 
			   && (quads[4*(cumNbq[m3]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m3]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m3]+q2) +3]==m4) );
		if(equal){
		  data[cumNbq[m3]+q2]=J;		
		  flag[cumNbq[m3]+q2]=0;
		}
	      }
	  }
	  
	  
	  {    //idem m4
	    int d=cumNbq[m4+1]-cumNbq[m4];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m4]+q2)]==m1) 
			   && (quads[4*(cumNbq[m4]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m4]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m4]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m4]+q2]=J;
		  flag[cumNbq[m4]+q2]=0;
		}
	      }
	  }
	  /////////////////////////////////////////////////////////////////////////////////////	  
	  

	  /////////////////////////////////////////////////////////////////////////////////////	  
	  // now we assign J to all the time the first permutation (1432) of the quadruplet appears in quads	  
	  m1=quads[4*q];
	  m2=quads[4*q+3];
	  m3=quads[4*q+2];
	  m4=quads[4*q+1];

	  { // for m1 
	    int d=cumNbq[m1+1]-cumNbq[m1];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m1]+q2)]==m1) 
			   && (quads[4*(cumNbq[m1]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m1]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m1]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m1]+q2]=J;
		  flag[cumNbq[m1]+q2]=0;
		}
	      }
	  }

	 
	  { // idem m2
	    int d=cumNbq[m2+1]-cumNbq[m2];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m2]+q2)]==m1) 
			   && (quads[4*(cumNbq[m2]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m2]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m2]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m2]+q2]=J;
		  flag[cumNbq[m2]+q2]=0;
		}
	      }
	  }
	  
	  {
	    //idem m3
	    int d=cumNbq[m3+1]-cumNbq[m3];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m3]+q2)]==m1) 
			   && (quads[4*(cumNbq[m3]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m3]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m3]+q2) +3]==m4) );
		if(equal){
		  data[cumNbq[m3]+q2]=J;		
		  flag[cumNbq[m3]+q2]=0;
		}
	      }
	  }
	  
	  
	  {    //idem m4
	    int d=cumNbq[m4+1]-cumNbq[m4];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m4]+q2)]==m1) 
			   && (quads[4*(cumNbq[m4]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m4]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m4]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m4]+q2]=J;
		  flag[cumNbq[m4]+q2]=0;
		}
	      }
	  }
	  /////////////////////////////////////////////////////////////////////////////////////	  



	  /////////////////////////////////////////////////////////////////////////////////////	  
	  // now we assign J to all the time the second permutation (1324) of the quadruplet appears in quads	  
	  m1=quads[4*q];
	  m2=quads[4*q+2];
	  m3=quads[4*q+1];
	  m4=quads[4*q+3];
	  
	  { // for m1 
	    int d=cumNbq[m1+1]-cumNbq[m1];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m1]+q2)]==m1) 
			   && (quads[4*(cumNbq[m1]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m1]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m1]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m1]+q2]=J;
		  flag[cumNbq[m1]+q2]=0;
		}
	      }
	  }

	  
	  { // idem m2
	    int d=cumNbq[m2+1]-cumNbq[m2];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m2]+q2)]==m1) 
			   && (quads[4*(cumNbq[m2]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m2]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m2]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m2]+q2]=J;
		  flag[cumNbq[m2]+q2]=0;
		}
	      }
	  }
	  
	  {
	    //idem m3
	    int d=cumNbq[m3+1]-cumNbq[m3];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m3]+q2)]==m1) 
			   && (quads[4*(cumNbq[m3]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m3]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m3]+q2) +3]==m4) );
		if(equal){
		  data[cumNbq[m3]+q2]=J;		
		  flag[cumNbq[m3]+q2]=0;
		}
	      }
	  }
	  
	  
	  {    //idem m4
	    int d=cumNbq[m4+1]-cumNbq[m4];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m4]+q2)]==m1) 
			   && (quads[4*(cumNbq[m4]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m4]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m4]+q2) +3]==m4) );
		if(equal) {
		  data[cumNbq[m4]+q2]=J;
		  flag[cumNbq[m4]+q2]=0;
		}
	      }
	  }
	  /////////////////////////////////////////////////////////////////////////////////////	  	  
	  
	  
	}
      
    }
  
  /*  for(int q=0;q<NbQuads;q++)
    {
      int    m1=quads[4*q];
      int    m2=quads[4*q+1];
      int    m3=quads[4*q+2];
      int    m4=quads[4*q+3];
      cout << m1 << " " << m2 << " " << m3 << " " << m4 << " " << data[q] << endl;
      }*/
  
  close_rng();
}
///////////////////////////////////////////////////////////////////////



// inits the coupling matrix with Gaussian random entries
// data[] is of dimension #neighbors=2.#bonds
// in such a way that both neighbors of the same bond have the same entry
// Box-Muller transform used
///////////////////////////////////////////////////////////////////////
void randomGaussianCouplingsp4(int NbQuads,int *quads,int *cumNbq,double* data,spin_t avrg,spin_t sigma,int seed) 
{
  open_rng(seed);
  double radicedi2=sqrt(2.);
  double ran1,ran2;
  
  double flag=-300.;
  
  for(int j=0;j<NbQuads;j++)  
    data[j]=flag;
  
  for (int q = 0;q<NbQuads;q++)
    {
      if(fabs(data[q]-flag)<1.0E-7)
	{
	  //RNGeneration
	  /////////////////////////////
	  ran1=rand_double(); 
	  ran2=rand_double();
	  double J=sin(ran2*twopi)*radicedi2*sigma*sqrt(-log(ran1))+avrg;
	  /////////////////////////////
	  
	  data[q]=J;
	  
	  int    m1=quads[4*q];
	  int    m2=quads[4*q+1];
	  int    m3=quads[4*q+2];
	  int    m4=quads[4*q+3];
	  
	  int d=cumNbq[m2+1]-cumNbq[m2];
	  for(int q2=0;q2<d;q2++)
	    {
	      int equal=((quads[4*(cumNbq[m2]+q2)]==m1) 
			 && (quads[4*(cumNbq[m2]+q2) +1]==m2)
			 && (quads[4*(cumNbq[m2]+q2) +2]==m3)
			 && (quads[4*(cumNbq[m2]+q2) +3]==m4) );
	      if(equal) data[cumNbq[m2]+q2]=J;
	      
	    }
	  
	  {
	    //idem m3
	    int d=cumNbq[m3+1]-cumNbq[m3];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m3]+q2)]==m1) 
			   && (quads[4*(cumNbq[m3]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m3]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m3]+q2) +3]==m4) );
		if(equal) data[cumNbq[m3]+q2]=J;
		
	      }
	  }
	  
	  
	  {    //idem m4
	    int d=cumNbq[m4+1]-cumNbq[m4];
	    for(int q2=0;q2<d;q2++)
	      {
		int equal=((quads[4*(cumNbq[m4]+q2)]==m1) 
			   && (quads[4*(cumNbq[m4]+q2) +1]==m2)
			   && (quads[4*(cumNbq[m4]+q2) +2]==m3)
			   && (quads[4*(cumNbq[m4]+q2) +3]==m4) );
		if(equal) data[cumNbq[m4]+q2]=J;
	      }
	  }	  
	}
            
    }
    
  /*  for(int q=0;q<NbQuads;q++)
    {
      int    m1=quads[4*q];
      int    m2=quads[4*q+1];
      int    m3=quads[4*q+2];
      int    m4=quads[4*q+3];
      cout << m1 << " " << m2 << " " << m3 << " " << m4 << " " << data[q] << endl;
      }*/
  
  close_rng();
}
///////////////////////////////////////////////////////////////////////





void ManyBigInit (spin_t * xs, spin_t * ys, spin_t eps, int N, int nPT, int nr, int my_s, double maxF) {

  assert(maxF > 0. && maxF <= 1.);
  
  for(int i=0; i<N*nPT*NR; i++){
    ys[i]=0.;
    xs[i]=-1.;
  }

  open_rng(my_s);

  for(int ri=0; ri<nr; ri++){
    for(int ti=0; ti<nPT; ti++){
      
      double sum=0;
      int noB=0;
      while(1){
	double big_r = maxF  * N * eps * rand_double();
	if((sum + big_r) > 1. * N * eps)
	  break;
	sum += big_r;
	noB++;
	int big_index = rand_double()*N;
	while( xs[big_index + ti*N + ri*N*nPT] != -1.  )
	  big_index = rand_double()*N;
	xs[big_index + ti*N + ri*N*nPT] = sqrt(big_r);
	}
      

      double the_small_ones = sqrt((eps*N-sum)/(1.*N-noB));
      for(int j=0; j<N; j++)
	if( xs[j + ti*N + ri*N*nPT] == -1.)
	  xs[j + ti*N + ri*N*nPT]=the_small_ones;
    }
  }
    
  close_rng();
}


/*
void dirichletInit (spin_t * xs, spin_t * ys, spin_t eps, int N, int nPT, int nr, int my_seed){  
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, my_seed);
  double * alpha = (double *) malloc(N*sizeof(double));
  for(int i=0;i<N;i++) alpha[i]= 1.;
  double * radii = (double *) malloc(N*sizeof(double));

  for(int ri=0;ri<nr;ri++){
    for(int ti=0;ti<nPT;ti++){
      gsl_ran_dirichlet (r,N, (const double *) alpha, radii);
      for(int i=0;i<N;i++){
	radii[i] *= N*eps;
	xs[i + ti*N + ri*N*nPT] = sqrt(radii[i]);
	ys[i + ti*N + ri*N*nPT] = 0.;
      }
    }
  }

  gsl_rng_free (r);
  free(alpha);
  free(radii);
}
*/

void randomInit(spin_t* data, spin_t Max, int size,int seed) {
open_rng(seed);

 for (int i = 0; i < size; ++i)
   data[i]=Max*rand_double();
 close_rng();
}


void randomInitDouble(double* data, spin_t Max, int size,int seed) {
  open_rng(seed);
  
  for (int i = 0; i < size; ++i)    
    data[i]=Max*rand_double();

  close_rng();  
}


void randomInitRandomSeeds(spin_t2 * data, spin_t Max, int nbblocks,int seed) {
  open_rng(seed);

 for(int bi=0;bi<nbblocks;bi++)
   {
    for (int i = 0; i < rfib; ++i)
      {
	data[i + bi*(rfib+sfib)]=Max*rand_double();
      }

   }
    close_rng();
}


void OneBigInit (spin_t * xs, spin_t * ys, spin_t eps, int N, int nPT, int nr, int my_s) {

  for(int i=0; i<N*nPT*NR; i++)
    ys[i]=0.;

  open_rng(my_s);

  double the_big_one = 1.*N/100.;
  double the_small_ones = sqrt((eps*N-the_big_one*the_big_one)/(1.*N-1.));
  for(int i=0; i<nPT; i++){
    int the_chosen_one = rand_double()*N;
    for(int q=0; q<nr; q++){
      for(int j=0; j<N; j++)
        xs[j + i*N + q*N*nPT]=the_small_ones;
      xs[the_chosen_one + i*N + q*N*nPT] = the_big_one;
    }
  }

  close_rng();
}


void uniformInit(spin_t *data, spin_t Max, int size, int seed) 
{
  open_rng(seed);
  
  for (int i = 0; i < size; ++i)
    if( rand_double() < 0.55 )
      data[i] = Max;
    else
      data[i]=-Max;
  
  close_rng();
}


/******************************************************/
void write_vector(int length,double *vec,FILE *file)
{
 int j;

 for(j=0;j<length;j++)
  fprintf(file,"%.14lf \n",vec[j]);   

}
/******************************************************/

/******************************************************/
void write_conf2D(int L,spin_t *C,FILE *config)
{
 int k,i,j;
 for(i=1;i<=L;i++)
 {
  for(j=1;j<=L;j++)
  {
   k=(i-1)*L+j-1;
   fprintf(config,"%f ",fmod(C[k]+2.*M_PI,2.*M_PI));  
   //    fprintf(config,"%f ",C[k]);   
  }
  fprintf(config,"\n");
 }
}
/******************************************************/


void write_output(FILE *myfile,int N,int tsamp,int nbEquil,int tmax,int seed,double beta,int disorder,int initnbQ,int finalnbQ)
{
  double thisJ;

#if INTERACTION_SCALING==0
#if FREQ_ENABLE>0
  thisJ = _Jvalue_ / (N*N);
#else
  thisJ = _Jvalue_ / (N*N*N);
#endif
  
#if DILUITE_TEST
  thisJ  *= 1./FRACTION;
#endif
#else
  thisJ =  _Jvalue_ * N / finalnbQ;    // so we are sure that the enrgy is extensive (if there is no localization)
#endif
  
  fprintf(myfile,"#main: N, total numer of modes:	%d \n", N);
#if DILUITE_TEST
  fprintf(myfile,"#main: FRACTION: %lf \n", FRACTION);
#endif
  fprintf(myfile,"#value of J interaction: %lf \n",thisJ);
  fprintf(myfile,"#initial tetrads : %d \n",initnbQ);
  fprintf(myfile,"# 4-plettes : %d \n",finalnbQ);
  fprintf(myfile,"#  _Jvalue_ : %f \n", _Jvalue_);
#if PRINT_CONFS>0
  fprintf(myfile,"#number of printed confs: %d",(int) NB_PRINTED_CONFS);
  fprintf(myfile,"#PRINT_CONFS = %d (if 0 does not print confs, if 2 print from the beginning, if 1 just in measures)",PRINT_CONFS);
#endif
  fprintf(myfile,"#Maximum Gain : %f \n", _GainMax_);
  if(_GainMax_ > 1e-10)
    fprintf(myfile,"#Reference temperature for the gain : %lf \n", _Tref_ );
  fprintf(myfile,"#main: disorder (0=no):              %d \n",disorder);
  fprintf(myfile,"#main: tsamp, #MCSs between measures:	%d \n",tsamp);
  fprintf(myfile,"#main: tmax, number of MC steps for Equil:	%d \n",nbEquil);
  fprintf(myfile,"#main: tmax, number of measures:	%d \n",tmax);
  fprintf(myfile,"#main: total number of MCSs:		%d \n", (nbEquil+tmax)*tsamp);
  fprintf(myfile,"#main: seed, RNG seed:			%d \n",seed);
  fprintf(myfile,"#main: beta, inverse temperature:	%f \n",beta);
  fprintf(myfile,"#main: size of the spin type:		%ld \n",sizeof(spin_t));
  fprintf(myfile,"#main: rFib, sFib, parameters of the Fibonacci RNG: %d , %d \n",rfib,sfib);
  fflush(myfile);
}


/*
void load_conf(int nm,spin_t *xs,spin_t *ys,FILE *config)
{
  int k;
  int aux;
  spin_t x,y;

  for(k=0;k<nm;k++)
    {
      fscanf(config,"%e %e %d",&x,&y,&aux);  
      xs[k]=x;
      ys[k]=y;
    }
}
*/


/******************************************************
double SMconstraint(int nm,FILE *config)
{
  int k;
  int aux;
  spin_t x,y;
  double R=0.;
  for(k=0;k<nm;k++)
    {
      fscanf(config,"%f %f %d",&x,&y,&aux);  
      R+=x*x+y*y;
    }
 return R;
}
/******************************************************/ 


void averages(int nPT,int C,double *betas)
{
/*********************************************/
 FILE *infile,*outfile;
 int q,k=0,c,auxi;
 double daux;
/*********************************************/
 double *d       = (double*) calloc(C*nPT,sizeof(double)); 
 double *dm      = (double*) calloc(C*nPT,sizeof(double)); 
 double *d2m     = (double*) calloc(C*nPT,sizeof(double));
/*********************************************/
 for(q=0;q<C*nPT;q++)
 {
  d[q]=0.;
  dm[q]=0.;
  d2m[q]=0.;
 }
/*********************************************/

 infile=fopen("measures.dat","r");
 outfile=fopen("av.dat","w");

 if(infile==NULL){
   printf("ERROR: measures.dat not found!!!\n\n");
   exit(1);
 }

/*********************************************/
   while( !(feof(infile)) )
   {k++;

    fscanf(infile,"%d ",&auxi);

   for(c=0;c<C;c++) 
    for(q=0;q<nPT;q++) 
    {
     fscanf(infile,"%lf ",&daux);

     dm[c + q*C]   += daux;
     d2m[c + q*C]  += daux*daux;
    }

   }
/*********************************************/




 fclose(infile);
/*********************************************/


 for(q=0;q<nPT;q++)
 {
   double T=1./betas[q];
   fprintf(outfile,"%f ",T);
  for(c=0;c<C;c++)
   fprintf(outfile,"%.14lf ",dm[c+q*C]/k);
  for(c=0;c<C;c++)
   fprintf(outfile,"%.14lf ",d2m[c+q*C]/k);

  fprintf(outfile,"\n");
 }

 free(d);
 free(dm);
 free(d2m); 

 fclose(outfile);

 return ;
}
///////////////////////////////////////////////////////////////////



void histograms(int nPT,int nObs,double xmin,double xmax,int nbins,int oindex,double *betas)
{
  /*********************************************/
  double length=(xmax-xmin);
  /*********************************************/
  FILE *infile,*outfile;
  double *h       = (double*) calloc(nbins*nPT,sizeof(double)); 
  double *norm       = (double*) calloc(nPT,sizeof(double)); 
  /*********************************************/
  
  infile=fopen("measures.dat","r");
  
  for(int q=0;q<nbins*nPT;q++)
    {
      h[q]=0.;
    }
  for(int q=0;q<nPT;q++)
    {
      norm[q]=0.;
    }
  /*********************************************/
    
  if(infile==NULL){
    printf("ERROR: measures.dat not found!!!\n\n");
    exit(1);
  }
      
  /*********************************************/
  int k=0;
  int auxi;
  double daux;
  while( !(feof(infile)) )
    {k++;
      
      fscanf(infile,"%d ",&auxi);
      
      for(int c=0;c<nObs;c++) 
	{
	  if(c==oindex)
	    {
	      for(int q=0;q<nPT;q++) 
		{
		  fscanf(infile,"%lf ",&daux);
		  
		  int bin=((daux-xmin)/length)*nbins;
		  if((bin>=0) && (bin<nbins))
		    {
		      h[bin+nbins*q]   += 1.;
		      norm[q] += 1.;
		    }
		}
	    }//if c==oindex
	  else
	    {
	      for(int q=0;q<nPT;q++) 
		fscanf(infile,"%lf ",&daux);
	    }
	}
      
    }
  fclose(infile);
  /*********************************************/
  
  
  /////////////////////////////////////
  char mychain[100];
  sprintf(mychain,"hisc%d.dat",oindex);
  outfile=fopen(mychain,"w");
  
  for(int bi=0;bi<nbins;bi++)
    {
      double x=xmin+(bi+0.5)*(length/nbins);
      fprintf(outfile,"%f ",x);
      for(int q=0;q<nPT;q++)
	{
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f ",thish);
	}
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  /////////////////////////////////////
  
  
  /////////////////////////////////////
  // histograms in 3D format
  sprintf(mychain,"his3Dc%d.dat",oindex);
  outfile=fopen(mychain,"w");
  
  
  for(int q=0;q<nPT;q++)
    {
      
      for(int bi=0;bi<nbins;bi++)
	{
	  double x=xmin+(bi+0.5)*(length/nbins);
	  fprintf(outfile,"%f ",x);
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f %lf %d \n",thish,1./betas[q],q);
	}
      fprintf(outfile,"\n");
    }
  /////////////////////////////////////

    
  free(h);
  free(norm);
  
  fclose(outfile);
  
  return ;
}


void histogramsOverlap2(int nPT,int nObs,double xmin,double xmax,int nbins,int Oindex,double *betas)
{
  /*********************************************/
  double length=(xmax-xmin);
  /*********************************************/
  FILE *infile,*outfile;
  double *h       = (double*) calloc(nbins*nPT,sizeof(double)); 
  double *norm       = (double*) calloc(nPT,sizeof(double)); 
  /*********************************************/
  
  if (Oindex==0)
    infile=fopen("overlapsQ.dat","r");
  else if (Oindex==1)
    infile=fopen("overlapsR.dat","r");
  else if (Oindex==2)
    infile=fopen("overlapsT.dat","r");

  for(int q=0;q<nbins*nPT;q++)
    {
      h[q]=0.;
    }
  for(int q=0;q<nPT;q++)
    {
      norm[q]=0.;
    }
  /*********************************************/
    
  if(infile==NULL){
    printf("ERROR: overlaps.dat not found!!!\n\n");
    exit(1);
  }
      
  /*********************************************/
  int k=0;
  int auxi;
  double daux;
  while( !(feof(infile)) )
    {k++;
      
      fscanf(infile,"%d ",&auxi);
      
      for(int c=0;c<nObs;c++) 
	{
	  if(c==0)
	    for(int q=0;q<nPT;q++) 
	      {
		fscanf(infile,"%lf ",&daux);
		
		int bin=((daux-xmin)/length)*nbins;
		if((bin>=0) && (bin<nbins))
		  {
		    h[bin+nbins*q]   += 1.;
		    norm[q] += 1.;
		}
	    }
	  else
	    {
	      for(int q=0;q<nPT;q++) 
		fscanf(infile,"%lf ",&daux);
	    }
	}    
    }
  fclose(infile);
  /*********************************************/
    
  /////////////////////////////////////
  char mychain[100];
  if (Oindex==0)
    sprintf(mychain,"hiscOverlapQ.dat");
  else if (Oindex==1)
    sprintf(mychain,"hiscOverlapR.dat");
  else if (Oindex==2)
    sprintf(mychain,"hiscOverlapT.dat");
  outfile=fopen(mychain,"w");
  
  for(int bi=0;bi<nbins;bi++)
    {
      double x=xmin+(bi+0.5)*(length/nbins);
      fprintf(outfile,"%f ",x);
      for(int q=0;q<nPT;q++)
	{
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f ",thish);
	}
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  /////////////////////////////////////  
  
  /////////////////////////////////////
  // histograms in 3D format
  if (Oindex==0)
    sprintf(mychain,"his3DcOverlapQ.dat");
  else if (Oindex==1)
    sprintf(mychain,"his3DcOverlapR.dat");
  else if (Oindex==2)
    sprintf(mychain,"his3DcOverlapT.dat");
  outfile=fopen(mychain,"w");
    
  for(int q=0;q<nPT;q++)
    {
      
      for(int bi=0;bi<nbins;bi++)
	{
	  double x=xmin+(bi+0.5)*(length/nbins);
	  fprintf(outfile,"%f ",x);
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f %lf %d\n",thish,1./betas[q], q);
	}
      fprintf(outfile,"\n");
    }
  /////////////////////////////////////
  
  free(h);
  free(norm);  
  fclose(outfile);
  
  return ;
}


void histogramsOverlap(int nPT,int nObs,double xmin,double xmax,int nbins,int oindex,double *betas)
{
  /*********************************************/
  double length=(xmax-xmin);
  /*********************************************/
  FILE *infile,*outfile;
  double *h       = (double*) calloc(nbins*nPT,sizeof(double)); 
  double *norm       = (double*) calloc(nPT,sizeof(double)); 
  /*********************************************/
  
  infile=fopen("overlaps.dat","r");
  
  for(int q=0;q<nbins*nPT;q++)
    {
      h[q]=0.;
    }
  for(int q=0;q<nPT;q++)
    {
      norm[q]=0.;
    }
  /*********************************************/
    
  if(infile==NULL){
    printf("ERROR: overlaps.dat not found!!!\n\n");
    exit(1);
  }
      
  /*********************************************/
  int k=0;
  int auxi;
  double daux;
  while( !(feof(infile)) )
    {k++;
      
      fscanf(infile,"%d ",&auxi);
      
      for(int c=0;c<nObs;c++) 
	{
	  if(c==oindex)
	    {
	      for(int q=0;q<nPT;q++) 
		{
		  fscanf(infile,"%lf ",&daux);
		  
		  int bin=((daux-xmin)/length)*nbins;
		  if((bin>=0) && (bin<nbins))
		    {
		      h[bin+nbins*q]   += 1.;
		      norm[q] += 1.;
		    }
		}
	    }//if c==oindex
	  else
	    {
	      for(int q=0;q<nPT;q++) 
		fscanf(infile,"%lf ",&daux);
	    }
	}    
    }
  fclose(infile);
  /*********************************************/
    
  /////////////////////////////////////
  char mychain[100];
  sprintf(mychain,"hiscOverlap%d.dat",oindex);
  outfile=fopen(mychain,"w");
  
  for(int bi=0;bi<nbins;bi++)
    {
      double x=xmin+(bi+0.5)*(length/nbins);
      fprintf(outfile,"%f ",x);
      for(int q=0;q<nPT;q++)
	{
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f ",thish);
	}
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  /////////////////////////////////////  
  
  /////////////////////////////////////
  // histograms in 3D format
  sprintf(mychain,"his3DcOverlap%d.dat",oindex);
  outfile=fopen(mychain,"w");
    
  for(int q=0;q<nPT;q++)
    {
      
      for(int bi=0;bi<nbins;bi++)
	{
	  double x=xmin+(bi+0.5)*(length/nbins);
	  fprintf(outfile,"%f ",x);
	  double thish=h[nbins*q + bi]*(nbins)/(length*norm[q]);
	  fprintf(outfile,"%f %lf %d\n",thish,1./betas[q],q);
	}
      fprintf(outfile,"\n");
    }
  /////////////////////////////////////
  
  free(h);
  free(norm);  
  fclose(outfile);
  
  return ;
}


void histogramsOverlapIFO_singleConf(int nPT,int N, double xmin,double xmax,int nbins,double *betas)
{

#if BINARY
  FILE * infile=fopen("overlapsIFO.dat","rb");
#else
  FILE * infile=fopen("overlapsIFO.dat","r");
#endif


  if(infile==NULL){
    printf("ERROR: overlapsIFO.dat not found!!!\n\n");
    exit(1);
  }

  double * Is;
  int size = NR*nPT*N*sizeof(double);
  Is = (double *) malloc(size);
  for(int a=0; a<NR*N*nPT; a++)
    Is[a]=0.;

  float * aux = (float *) malloc(NR*nPT*N*sizeof(float));

#if BINARY

  int c=0;
  while( !(feof(infile)) ) {
    fread (aux,sizeof(float),NR*nPT*N,infile);
    for(int r1=0;r1<NR;r1++) {
      for(int bi=0;bi<nPT;bi++)
	for(int j=0;j<N;j++) {
	  //	  Is[j + bi*N + N*nPT*r1] += aux[j + bi*N + N*nPT*r1];
	  Is[j + bi*N + N*nPT*r1] = (Is[j + bi*N + N*nPT*r1]*c + aux[j + bi*N + N*nPT*r1])/(c+1);
	  }
    }
    c++;
  }  

#else

  int c=0;
  while( !(feof(infile)) ) {
    for(int r1=0;r1<NR;r1++) {
      for(int bi=0;bi<nPT;bi++)
	for(int j=0;j<N;j++) {
	    float aux;
	    fscanf(infile,"%f ", &aux);
	    Is[j + bi*N + N*nPT*r1] = (Is[j + bi*N + N*nPT*r1]*c + aux)/(c+1);
	  //	    Is[j + bi*N + N*nPT*r1] += aux;	    
	  }
      //      fscanf(infile,"\n");
    }
    c++;
  }  

#endif

  /*
  for(int bi=0;bi<nPT;bi++) {
    for(int j=0;j<N;j++) {
      printf("%d %le %d\n",j, (Is[j + bi*N] + Is[j + bi*N + N*NPT] + Is[j + bi*N + 2.*N*NPT] + Is[j + bi*N + 3.*N*NPT])/4.,bi);
    }
    printf("\n");
  }
  */
  // Is[a] contiene il valor medio I_k^a per ogni spin, per ogni T, per ogni replica

  rewind(infile);

  double length=(xmax-xmin);

  double * h = (double*) calloc(nbins*nPT,sizeof(double)); 
  double * norm = (double*) calloc(nPT,sizeof(double)); 
  
  for(int q=0;q<nbins*nPT;q++) {
      h[q]=0.;
    }
  for(int q=0;q<nPT;q++) {
      norm[q]=0.;
    }

  int k=0;
  while( !(feof(infile)) )
    {k++;

#if BINARY
      fread (aux,sizeof(float),NR*nPT*N,infile);
#else
      for(int r1=0;r1<NR;r1++) 
	for(int bi=0;bi<nPT;bi++)
	  for(int j=0;j<N;j++) 
	    fscanf(infile,"%f ", aux + j + bi*N + N*nPT*r1);
#endif

      for(int q=0;q<nPT;q++) {
	for(int r1=1;r1<NR;r1++) {
	  for(int r2=0;r2<r1;r2++) {
	    
	    double Iov = 0., n1 = 0., n2 = 0.;
	    
	    for(int j=0;j<N;j++) {
	      
	      double f1 = aux[j + q*N + N*nPT*r1] - Is[j + q*N + N*nPT*r1];
	      double f2 = aux[j + q*N + N*nPT*r2] - Is[j + q*N + N*nPT*r2];
	      
	      Iov += f1 * f2;
	      n1  += f1 * f1;
	      n2  += f2 * f2;
	      
	    }
	    
	    double thisIFO = 0.;
	    if (n1 * n2 > 0.)
	      thisIFO = Iov / ( sqrt( n1 * n2 ) );
	    
	    assert(thisIFO>=-1. && thisIFO<=1.);
	    
	    int bin=((thisIFO-xmin)/length)*nbins;
	    if((bin>=0) && (bin<nbins)) {
	      h[bin+nbins*q] += 1.;
	      norm[q] += 1.;
	    }
	  }	  
	  
	}	
      }
    }

  fclose(infile);

    
  /////////////////////////////////////
  char mychain[100];
  sprintf(mychain,"hiscOverlapIFO.dat");
  FILE * outfile;
  outfile=fopen(mychain,"w");
  
  for(int bi=0;bi<nbins;bi++)
    {
      double x = xmin+(bi+0.5)*(length/nbins);
      fprintf(outfile,"%lf ",x);
      for(int q=0;q<nPT;q++)
	{
	  double thish = h[nbins*q + bi]*nbins/(length*norm[q]);
	  fprintf(outfile,"%le ",thish);
	}
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  /////////////////////////////////////  
  
  /////////////////////////////////////
  // histograms in 3D format
  sprintf(mychain,"his3DcOverlapIFO.dat");
  outfile=fopen(mychain,"w");
    
  for(int q=0;q<nPT;q++)
    {      
      for(int bi=0;bi<nbins;bi++)
	{
	  double x = xmin+(bi+0.5)*(length/nbins);
	  double thish = h[nbins*q + bi]*nbins/(length*norm[q]);
	  fprintf(outfile,"%le %le %lf %d\n",x,thish,1./betas[q],q);
	}
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  /////////////////////////////////////
  
  free(h);
  free(norm);  
  free(Is);
  free(aux);

  
  return ;
}



void averagesCorrelations (int nPT,double *betas)
{
/*********************************************/
 FILE *infile,*outfile;
 int q,k=0,c,auxi;
 double daux;
 int C=1;
/*********************************************/
 double *d       = (double*) calloc(C*nPT,sizeof(double)); 
 double *dm      = (double*) calloc(C*nPT,sizeof(double)); 
 double *d2m     = (double*) calloc(C*nPT,sizeof(double));
/*********************************************/
 for(q=0;q<C*nPT;q++)
 {
  d[q]=0.;
  dm[q]=0.;
  d2m[q]=0.;
 }
/*********************************************/

 infile=fopen("conn_corr_versors_integral.dat","r");
 outfile=fopen("av_conn.dat","w");

 if(infile==NULL){
   printf("ERROR: conn_corr_versors_integral.dat not found!!!\n\n");
   return;
 }

/*********************************************/
   while( !(feof(infile)) )
   {k++;

    fscanf(infile,"%d ",&auxi);

   for(c=0;c<C;c++) 
    for(q=0;q<nPT;q++) 
    {
     fscanf(infile,"%lf ",&daux);

     dm[c + q*C]   += daux;
     d2m[c + q*C]  += daux*daux;
    }

   }
/*********************************************/

 fclose(infile);
/*********************************************/

 for(q=0;q<nPT;q++)
 {
   double T=1./betas[q];
   fprintf(outfile,"%f ",T);
  for(c=0;c<C;c++)
   fprintf(outfile,"%.14lf ",dm[c+q*C]/k);
  for(c=0;c<C;c++)
   fprintf(outfile,"%.14lf ",d2m[c+q*C]/k);

  fprintf(outfile,"\n");
 }

 free(d);
 free(dm);
 free(d2m); 

 fclose(outfile);

 return ;
}
///////////////////////////////////////////////////////////////////



void binderPar(int nPT, double *betas)
{

  double * Q   = (double*) calloc(nPT,sizeof(double)); 
  double * Q2   = (double*) calloc(nPT,sizeof(double)); 
  double * Q3   = (double*) calloc(nPT,sizeof(double)); 
  double * Q4   = (double*) calloc(nPT,sizeof(double)); 
  
  /*********************************************/
  {
    FILE * infile=fopen("overlapsQ.dat","r");
    FILE * outfile=fopen("binderQ.dat","w");
    
    for(int q=0;q<nPT;q++) {
      Q[q]=0.;
      Q2[q]=0.;
      Q3[q]=0.;
      Q4[q]=0.;
    }
    
    if(infile==NULL){
      printf("ERROR: overlaps.dat not found!!!\n\n");
      exit(1);
    }
    
    int k=0;
    int auxi;
    double daux;
    while( !(feof(infile)) )
      {      
	fscanf(infile,"%d ",&auxi);
	
	for(int q=0;q<nPT;q++) {
	  fscanf(infile,"%lf ",&daux);
	  
	  Q[q] = (Q[q]*k + daux)/(k+1);
	  Q2[q] = (Q2[q]*k + daux*daux)/(k+1);
	  Q3[q] = (Q3[q]*k + daux*daux*daux)/(k+1);
	  Q4[q] = (Q4[q]*k + daux*daux*daux*daux)/(k+1);
	  
	}
	k++;	
      }
    fclose(infile);
    
    fprintf(outfile,"#   T   Binder   Q    Q2   Q3   Q4   Tidx \n");  
    for(int q=0;q<nPT;q++) {
      double thisVar = Q2[q] - Q[q]*Q[q];
      double thisB = 3. - 2. * (Q4[q] - 4.*Q3[q]*Q[q] - 3.*Q2[q]*Q2[q] + 12.*Q2[q]*Q[q]*Q[q] - 6.*Q[q]*Q[q]*Q[q]*Q[q])/ (thisVar*thisVar);
      fprintf(outfile,"%f %le  %le  %le  %le  %le %d \n", 1./betas[q], thisB, Q[q], Q2[q], Q3[q], Q4[q], q);
    }
    fclose(outfile);
   
  }
  /*********************************************/    


  /*********************************************/
  {
    FILE * infile=fopen("overlapsR.dat","r");
    FILE * outfile=fopen("binderR.dat","w");
    
    for(int q=0;q<nPT;q++) {
      Q[q]=0.;
      Q2[q]=0.;
      Q3[q]=0.;
      Q4[q]=0.;
    }
    
    if(infile==NULL){
      printf("ERROR: overlapsR.dat not found!!!\n\n");
      exit(1);
    }
    
    int k=0;
    int auxi;
    double daux;
    while( !(feof(infile)) )
      {      
	fscanf(infile,"%d ",&auxi);
	
	for(int q=0;q<nPT;q++) {
	  fscanf(infile,"%lf ",&daux);
	  
	  Q[q] = (Q[q]*k + daux)/(k+1);
	  Q2[q] = (Q2[q]*k + daux*daux)/(k+1);
	  Q3[q] = (Q3[q]*k + daux*daux*daux)/(k+1);
	  Q4[q] = (Q4[q]*k + daux*daux*daux*daux)/(k+1);
	  
	}
	k++;	
      }
    fclose(infile);
    
    fprintf(outfile,"#   T   Binder   R    R2   R3   R4   Tidx \n");  
    for(int q=0;q<nPT;q++) {
      double thisVar = Q2[q] - Q[q]*Q[q];
      double thisB = 3. - 2. * (Q4[q] - 4.*Q3[q]*Q[q] - 3.*Q2[q]*Q2[q] + 12.*Q2[q]*Q[q]*Q[q] - 6.*Q[q]*Q[q]*Q[q]*Q[q])/ (thisVar*thisVar);
      fprintf(outfile,"%f %le  %le  %le  %le  %le %d \n", 1./betas[q], thisB, Q[q], Q2[q], Q3[q], Q4[q], q);
    }
    fclose(outfile);
   
  }
  /*********************************************/    


  /*********************************************/
  {
    FILE * infile=fopen("overlapsT.dat","r");
    FILE * outfile=fopen("binderT.dat","w");
    
    for(int q=0;q<nPT;q++) {
      Q[q]=0.;
      Q2[q]=0.;
      Q3[q]=0.;
      Q4[q]=0.;
    }
    
    if(infile==NULL){
      printf("ERROR: overlapsT.dat not found!!!\n\n");
      exit(1);
    }
    
    int k=0;
    int auxi;
    double daux;
    while( !(feof(infile)) )
      {      
	fscanf(infile,"%d ",&auxi);
	
	for(int q=0;q<nPT;q++) {
	  fscanf(infile,"%lf ",&daux);
	  
	  Q[q] = (Q[q]*k + daux)/(k+1);
	  Q2[q] = (Q2[q]*k + daux*daux)/(k+1);
	  Q3[q] = (Q3[q]*k + daux*daux*daux)/(k+1);
	  Q4[q] = (Q4[q]*k + daux*daux*daux*daux)/(k+1);
	  
	}
	k++;	
      }
    fclose(infile);
    
    fprintf(outfile,"#   Temp   Binder   T    T2   T3   T4   Tidx \n");  
    for(int q=0;q<nPT;q++) {
      double thisVar = Q2[q] - Q[q]*Q[q];
      double thisB = 3. - 2. * (Q4[q] - 4.*Q3[q]*Q[q] - 3.*Q2[q]*Q2[q] + 12.*Q2[q]*Q[q]*Q[q] - 6.*Q[q]*Q[q]*Q[q]*Q[q])/ (thisVar*thisVar);
      fprintf(outfile,"%f %le  %le  %le  %le  %le %d \n", 1./betas[q], thisB, Q[q], Q2[q], Q3[q], Q4[q], q);
    }
    fclose(outfile);
   
  }
  /*********************************************/    
    

  free(Q);
  free(Q2);
  free(Q3);
  free(Q4);
  
  return ;
}



void binderParIFO(int nPT, double *betas)
{

  double * Q   = (double*) calloc(nPT,sizeof(double)); 
  double * Q2   = (double*) calloc(nPT,sizeof(double)); 
  double * Q3   = (double*) calloc(nPT,sizeof(double)); 
  double * Q4   = (double*) calloc(nPT,sizeof(double)); 
  
  /*********************************************/
  {
    FILE * infile=fopen("overlapsIFO.dat","r");
    FILE * outfile=fopen("binderIFO.dat","w");
    
    for(int q=0;q<nPT;q++) {
      Q[q]=0.;
      Q2[q]=0.;
      Q3[q]=0.;
      Q4[q]=0.;
    }
    
    if(infile==NULL){
      printf("ERROR: overlapsIFO.dat not found!!!\n\n");
      exit(1);
    }
    
    int k=0;
    int auxi;
    double daux;
    while( !(feof(infile)) )
      {      
	fscanf(infile,"%d ",&auxi);
	
	for(int q=0;q<nPT;q++) {
	  fscanf(infile,"%lf ",&daux);
	  
	  Q[q] = (Q[q]*k + daux)/(k+1);
	  Q2[q] = (Q2[q]*k + daux*daux)/(k+1);
	  Q3[q] = (Q3[q]*k + daux*daux*daux)/(k+1);
	  Q4[q] = (Q4[q]*k + daux*daux*daux*daux)/(k+1);

	// forzare la simmetria q -> -q
	  /*
	  Q[q] = (Q[q]*k - daux)/(k+2);
	  Q2[q] = (Q2[q]*k + daux*daux)/(k+2);
	  Q3[q] = (Q3[q]*k - daux*daux*daux)/(k+2);
	  Q4[q] = (Q4[q]*k + daux*daux*daux*daux)/(k+2);
	  */
	}
	k++; 
	//	k += 2;
      }
    fclose(infile);
    
    fprintf(outfile,"#   T   Binder   q    q2   q3   q4   Tidx \n");  
    for(int q=0;q<nPT;q++) {
      double thisVar = Q2[q] - Q[q]*Q[q];
      double thisB = 3. - 2. * (Q4[q] - 4.*Q3[q]*Q[q] - 3.*Q2[q]*Q2[q] + 12.*Q2[q]*Q[q]*Q[q] - 6.*Q[q]*Q[q]*Q[q]*Q[q])/ (thisVar*thisVar);
      fprintf(outfile,"%f %le  %le  %le  %le  %le %d \n", 1./betas[q], thisB, Q[q], Q2[q], Q3[q], Q4[q], q);
    }
    fclose(outfile);
   
  }
  /*********************************************/    


  free(Q);
  free(Q2);
  free(Q3);
  free(Q4);
  
  return ;
}
