////////////////////////////////////////////////////////////////////

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


//////////////////////////////////////////////////////////////////////////////////////
////// ----------------------- MONTE CARLO STEP ---------- GIACOMO--------------- 2017 
///////////////////////////////////////////////////////////////////////////////////

void MCstep_disorderp4_placchette(double T, int N, int Ncoppie, Dimers_type ** coppie, double *gain, Plaqs_type ** placchette, spin_t * xs, spin_t * ys, double *nrg){
  
  int i,j;
  int ii,jj,ij;
  int iii,jjj,iijj;

  int k,idimer;
  int ind_spin;
  int ind_spin_i,ind_spin_j;

  int ind_plaq;
  int Nplaqs_i,Nplaqs_j,Nplaqs_ij;

  double numero_casuale;
  //double beta=(*betas);  //  this index is NOT myq	  
  double de;
  double new_ene;

  spin_t x1,x2;
  spin_t y1,y2;
  spin_t r1,r2,factor;
  spin_t alpha,phi1,phi2;
  spin_t sp1,cp1,sp2,cp2,sa,ca;
  spin_t nx1,nx2,ny1,ny2;
  spin_t x[4],y[4];



  // for(k=0;k<N;k++){
  //   printf("xs[%d] = %g \n",k,xs[k]);
  // }

  for(k=0;k<N;k++){
    
    numero_casuale=rand_double();
    idimer = Ncoppie*numero_casuale;  

    Dimers_type * coppia;

    coppia = coppie[idimer];
    
    //printf("Start Attempt n° %d:  try couple n° %d \n",k,idimer);
    
    i = coppia->spin_i_index;
    j = coppia->spin_j_index;

    ///////////////////////////////////////////////////////////////////
    ////////////// ------------------ ATTEMPTED UPDATE OF THE TWO SPINS
    ///////////////////////////////////////////////////////////////////
    ////////////// ------------------ i -> 1
    ////////////// ------------------ j -> 2
    
    x1 = xs[i]; //spins[i].x;  //xs[i];      
    x2 = xs[j]; //spins[j].x;  //xs[j];      
    y1 = ys[i]; //spins[i].y;  //ys[i];
    y2 = ys[j]; //spins[j].y;  //ys[j];

    r1 = x1*x1+y1*y1;  
    r2 = x2*x2+y2*y2;
    factor=mysqrt((r1+r2)/r1);
    /////////////////////////////////////////
    
    // Metropolis
    /////////////////////////////////////////
    alpha = rand_double()*twopi;
    phi1  = rand_double()*twopi;
    phi2  = rand_double()*twopi;
    
    mysincos(phi1,&sp1,&cp1);
    mysincos(phi2,&sp2,&cp2);
    mysincos(alpha,&sa,&ca);
    
    nx1=x1*cp1+y1*sp1;
    ny1=-x1*sp1+y1*cp1;
    nx2=x2*cp2+y2*sp2;
    ny2=-x2*sp2+y2*cp2;
    
    nx1=nx1*factor*ca;
    ny1=ny1*factor*ca;
    
    factor=mysqrt((r1+r2)/r2);
    
    nx2=nx2*factor*sa;
    ny2=ny2*factor*sa;	  

    //////////////////////////////
    
    de=0.0;
    
#if I_WANT_GAIN	  
    de = gain[i]*(x1*x1+y1*y1-nx1*nx1-ny1*ny1) + gain[j]*(x2*x2+y2*y2-nx2*nx2-ny2*ny2);      
#endif
    
    Nplaqs_i  = coppia->Nplaqs_i;
    Nplaqs_j  = coppia->Nplaqs_j;
    Nplaqs_ij = coppia->Nplaqs_ij;
    
    ///////////////////////////////////////////////////////////
    //(1)////// ------ CICLO PLACCHETTE DOVE PARTECIPA SOLO SPIN i 
    ///////////////////////////////////////////////////////////

    for(ii=0; ii < Nplaqs_i; ii++){
      
      ind_spin = coppia->i_ind_spin_ar[ii];
      ind_plaq = coppia->i_ind_plaq_ar[ii];
      
      //printf("cerco placchetta n° %d \n",ind_plaq);
      
      for(iii=0; iii<4; iii++){
    	//printf("cerco spin n° %d \n",placchette[ind_plaq]->spin_index[iii]);
    	x[iii]=xs[placchette[ind_plaq]->spin_index[iii]];
    	y[iii]=ys[placchette[ind_plaq]->spin_index[iii]];
      }
      
      //printf("ind_spin = %d \n",ind_spin);

      x[ind_spin]=nx1;
      y[ind_spin]=ny1;

      //x[0]=nx1; // TOGLI FINE PROVA !!!!
      //y[0]=ny1; // TOGLI FINE PROVA !!!!
      
      new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
      
      placchette[ind_plaq]->ene_new = new_ene;
      
      de += new_ene - placchette[ind_plaq]->ene;
      
    }
    
    //de=0; // TOGLI FINE PROVA !!!!
    
    ///////////////////////////////////////////////////////////
    //(2)/// ------ CICLO PLACCHETTE DOVE PARTECIPA SOLO SPIN j 
    /////////////////////////////////////////////////////////// 

    for(jj=0; jj < Nplaqs_j; jj++){
      
      ind_spin=coppia->j_ind_spin_ar[jj];
      ind_plaq=coppia->j_ind_plaq_ar[jj];
      
      for(jjj=0; jjj<4; jjj++){
    	x[jjj]=xs[placchette[ind_plaq]->spin_index[jjj]];
    	y[jjj]=ys[placchette[ind_plaq]->spin_index[jjj]];
      }
      
      x[ind_spin]=nx2;
      y[ind_spin]=ny2;

      //x[0]=nx2; // TOGLI FINE PROVA !!!!
      //y[0]=ny2; // TOGLI FINE PROVA !!!!

      new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
      
      placchette[ind_plaq]->ene_new = new_ene;
      
      de += new_ene - placchette[ind_plaq]->ene;
      
    }
    
    //de=0; // TOGLI FINE PROVA !!!!
    
    ///////////////////////////////////////////////////////////
    //(3)/// ------ CICLO PLACCHETTE DOVE PARTECIPANO i & j /// 
    /////////////////////////////////////////////////////////// 
    
    
    for(ij=0; ij < Nplaqs_ij; ij++){
      
      ind_spin_i = coppia->ij_ind_spin[2*ij];
      ind_spin_j = coppia->ij_ind_spin[2*ij+1];
      
      ind_plaq   = coppia->ij_ind_plaq[ij];
      
      for(iijj=0; iijj<4; iijj++){
    	x[iijj]=xs[placchette[ind_plaq]->spin_index[iijj]];
    	y[iijj]=ys[placchette[ind_plaq]->spin_index[iijj]];
      }
      
      x[ind_spin_i]=nx1;
      y[ind_spin_i]=ny1;
      
      x[ind_spin_j]=nx2;
      y[ind_spin_j]=ny2;
      
      // x[0]=nx1; // TOGLI FINE PROVA !!!!
      // y[0]=ny1; // TOGLI FINE PROVA !!!!
      
      // x[1]=nx2; // TOGLI FINE PROVA !!!!
      // y[1]=ny2; // TOGLI FINE PROVA !!!!
     
      new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
      
      placchette[ind_plaq]->ene_new = new_ene;
      
      de += new_ene - placchette[ind_plaq]->ene;
 
    }

    //de=0; // TOGLI FINE PROVA !!!!

    //--------------- DECIDO SE FARE LA MOSSA----------/////////////// 

    
    
    if(de<0){

      xs[i]=nx1; //spins[i].x
      xs[j]=nx2; //spins[j].x
      ys[i]=ny1; //spins[i].y
      ys[j]=ny2; //spins[j].y 
      
      for(ii=0; ii < Nplaqs_i; ii++){
      	ind_plaq=coppia->i_ind_plaq[ii];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      for(jj=0; jj < Nplaqs_j; jj++){
      	ind_plaq=coppia->j_ind_plaq[jj];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      for(ij=0; ij < Nplaqs_ij; ij++){
      	ind_plaq=coppia->ij_ind_plaq[ij];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      (*nrg) += de;
      
    }else if(rand_double() < exp(-(de)/T)){
      
      xs[i]=nx1; //spins[i].x
      xs[j]=nx2; //spins[j].x
      ys[i]=ny1; //spins[i].y
      ys[j]=ny2; //spins[j].y 
      
      for(ii=0; ii < Nplaqs_i; ii++){
      	ind_plaq=coppia->i_ind_plaq[ii];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      for(jj=0; jj < Nplaqs_j; jj++){
      	ind_plaq=coppia->j_ind_plaq[jj];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      for(ij=0; ij < Nplaqs_ij; ij++){
      	ind_plaq=coppia->ij_ind_plaq[ij];
      	placchette[ind_plaq]->ene=placchette[ind_plaq]->ene_new;
      }
      
      (*nrg) += de;
      
    }

    //printf("Done Attempt n° %d:  try couple n° %d \n",k,idimer);

  } // CICLO SULLE N COPPIE DI SPIN DA AGGIORNARE //  for(int k=0;k<N;k++){

  return;

}

// __global__ void ComputePlaquetteEnergy_ONE_spin_i(double * de,double nx, double ny, Dimers_type * d_coppia, Plaqs_type ** d_placchette, spin_t * d_xs, spin_t * d_ys){

//   double new_ene;

//   int ii = blockIdx.x * blockDim + threadIdx.x;

//   int ind_spin = d_coppia->i_ind_spin_ar[ii];
//   int ind_plaq = d_coppia->i_ind_plaq_ar[ii];
  
//   //printf("cerco placchetta n° %d \n",ind_plaq);
  
//   spin_t x[4],y[4];

//   for(int iii=0; iii<4; iii++){
//     //printf("cerco spin n° %d \n",placchette[ind_plaq]->spin_index[iii]);
//     x[iii]=d_xs[placchette[ind_plaq]->spin_index[iii]];
//     y[iii]=d_ys[placchette[ind_plaq]->spin_index[iii]];
//   }
  
//   //printf("ind_spin = %d \n",ind_spin);
  
//   x[ind_spin]=nx;
//   y[ind_spin]=ny;
  
//   //x[0]=nx1; // TOGLI FINE PROVA !!!!
//   //y[0]=ny1; // TOGLI FINE PROVA !!!!
  
//   new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
  
//   placchette[ind_plaq]->ene_new = new_ene;
  
//   (*de) += new_ene - placchette[ind_plaq]->ene;
  
//  }

// __global__ void ComputePlaquetteEnergy_ONE_spin_j(double * de,double nx, double ny, Dimers_type * d_coppia, Plaqs_type ** d_placchette, spin_t * d_xs, spin_t * d_ys){

//   double new_ene;

//   int ii = blockIdx.x * blockDim + threadIdx.x;

//   int ind_spin = d_coppia->j_ind_spin_ar[ii];
//   int ind_plaq = d_coppia->j_ind_plaq_ar[ii];
  
//   //printf("cerco placchetta n° %d \n",ind_plaq);
  
//   spin_t x[4],y[4];

//   for(int iii=0; iii<4; iii++){
//     //printf("cerco spin n° %d \n",placchette[ind_plaq]->spin_index[iii]);
//     x[iii]=d_xs[placchette[ind_plaq]->spin_index[iii]];
//     y[iii]=d_ys[placchette[ind_plaq]->spin_index[iii]];
//   }
  
//   //printf("ind_spin = %d \n",ind_spin);
  
//   x[ind_spin]=nx;
//   y[ind_spin]=ny;
  
//   //x[0]=nx1; // TOGLI FINE PROVA !!!!
//   //y[0]=ny1; // TOGLI FINE PROVA !!!!
  
//   new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
  
//   placchette[ind_plaq]->ene_new = new_ene;
  
//   (*de) += new_ene - placchette[ind_plaq]->ene;
  
//  }



// __global__ void ComputePlaquetteEnergy_TWO_spin(double * de,double nx1, double ny1, double nx2, double ny2, Dimers_type * d_coppia, Plaqs_type ** d_placchette, spin_t * d_xs, spin_t * d_ys){

//   double new_ene;

//   int ij = blockIdx.x * blockDim + threadIdx.x;

//   int ind_spin_i = coppia->ij_ind_spin[2*ij];
//   int ind_spin_j = coppia->ij_ind_spin[2*ij+1];
  
//   int ind_plaq   = coppia->ij_ind_plaq[ij];
  
//   for(int iijj=0; iijj<4; iijj++){
//     x[iijj]=xs[placchette[ind_plaq]->spin_index[iijj]];
//     y[iijj]=ys[placchette[ind_plaq]->spin_index[iijj]];
//   }
  
//   x[ind_spin_i]=nx1;
//   y[ind_spin_i]=ny1;
  
//   x[ind_spin_j]=nx2;
//   y[ind_spin_j]=ny2;
  
//   new_ene = - placchette[ind_plaq]->J * (((x[0]*x[1]-y[0]*y[1])*x[2] + (y[0]*x[1]+x[0]*y[1])*y[2])*x[3]+((y[0]*y[1]-x[0]*x[1])*y[2] + (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
  
//   placchette[ind_plaq]->ene_new = new_ene;
  
//   (*de) += new_ene - placchette[ind_plaq]->ene;
  

// }
