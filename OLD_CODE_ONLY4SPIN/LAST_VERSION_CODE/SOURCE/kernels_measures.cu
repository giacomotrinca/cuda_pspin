////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// each thread computes the energy of a quadruplette (4-redundant)
//va chiamata <<<N.nPT/LBS,LBS>>>
__global__ void energyPT_disorder1replica_nodeBased(spin_t *xs,spin_t *ys,int N,int nPT,double *ens)
{
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  //  int q   = (gti) /(N)  ;     	// termperature index 
  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 

  __shared__ double  en[LBS];

  //  int myq=permsConst[q]; 
  int myq=q;

  if(ni<N)
    {
      
      double e = 0.;

#if I_WANT_GAIN == 1
      e += - (xs[ni + myq*N]*xs[ni  + myq*N] + ys[ni + myq*N]*ys[ni + myq*N])*tex1Dfetch(gain_texRef,ni)/sqrt(betaConst[q]);
#endif
      
      {
	
	int mi=ni;
	int d1 = tex1Dfetch(cumNbq_texRef,mi+1)-tex1Dfetch(cumNbq_texRef,mi);
	
	for(int qi=0;qi<d1;qi++)
	  {
	    int myquad=tex1Dfetch(cumNbq_texRef,mi)+qi;
	    
	    int myquadm;	
	    spin_t x[4],y[4];
	    //	  double g;	  
	    for(int j=0;j<4;j++)
	      {
		myquadm= tex1Dfetch(quads_texRef,4*myquad+j);
		myquadm=myquadm   + myq*N ;
		x[j] = xs[ myquadm ];
		y[j] = ys[ myquadm ];	
		
	      }
	    
	    
	    e -=  0.25*tex1Dfetch(J_texRef,myquad) * (((x[0]*x[1]-y[0]*y[1])*x[2] +
						       (y[0]*x[1]+x[0]*y[1])*y[2])*x[3] 
						      +
						      ((y[0]*y[1]-x[0]*x[1])*y[2] +
						       (x[0]*y[1]+y[0]*x[1])*x[2])*y[3]);
	    
	  }	    
      }
      
      
      //measuring the energy:
      /////////////////////////////////////////
      en[ti] = e;
      
    }
  else en[ti] = 0.;
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en[j-1] += en[j];}
      ens[bi] = en[0];
    }
  /////////////////////////////////////////
}



//this is the matrix q11,q12,q21,q22
//va chiamata <<<nPT*N/BS,BS>>>
__global__ void SMoverlap(int N,int nPT,spin_t *xs,spin_t *ys,double *q11,double *q12,double *q21,double *q22)
{
  int gti = threadIdx.x+ blockIdx.x*blockDim.x;
  int q=gti/(N);
  
  int myq1=permsConst [q]; 
  int myq2=permsConst[q+nPT]; 
  
  int ir=gti-q*N;
  
  spin_t2 x1 = xs[N*myq1+ir];
  spin_t2 x2 = xs [N*myq2+ir+N*nPT];
  spin_t2 y1 = ys [N*myq1+ir];
  spin_t2 y2 = ys [N*myq2+ir+N*nPT];
  
  
  spin_t2   o11 = (x1*x2);
  spin_t2   o12 = (x1*y2);
  spin_t2   o21 = (y1*x2);
  spin_t2   o22 = (y1*y2);
  
  /////////////////////////////////////////
  __shared__ double  en[LBS];
  
  en[threadIdx.x] = o11;
  
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en[j-1] += en[j];}
      q11[blockIdx.x] = en[0];
    }
  __syncthreads();
  /////////////////////////////////////////   
  en[threadIdx.x] = o12;
  
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en[j-1] += en[j];}
      q12[blockIdx.x] = en[0];
    }
  __syncthreads();
  /////////////////////////////////////////  
  en[threadIdx.x] = o21;
  
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en[j-1] += en[j];}
      q21[blockIdx.x] = en[0];
    }
  __syncthreads();
  /////////////////////////////////////////  
  en[threadIdx.x] = o22;
  
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en[j-1] += en[j];}
      q22[blockIdx.x] = en[0];
    }
  /////////////////////////////////////////
}



__global__ void SMqsR1R2(int N, int nPT, spin_t *xs, spin_t *ys, int r1, int r2, double *qs)   // r1 = 0, ... NR-1
{

  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  //  int q   = (gti) /(N)  ;     	// termperature index 
  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 

  int myq1 = permsConst[q+r1*nPT]; 
  int myq2 = permsConst[q+r2*nPT]; 

  spin_t ovs = 0.;

  if(ni<N)
    {

      spin_t x1 = xs[N*myq1+ni+r1*N*nPT];
      spin_t x2 = xs[N*myq2+ni+r2*N*nPT];
      spin_t y1 = ys[N*myq1+ni+r1*N*nPT];
      spin_t y2 = ys[N*myq2+ni+r2*N*nPT];
      
      ovs = (y1*y2+x1*x2);
 
    }
     
  /////////////////////////////////////////
  __shared__ double  en[LBS];
  
  en[threadIdx.x] = ovs;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      qs[blockIdx.x] = en[0];
    }
  
  /////////////////////////////////////////   
}

__global__ void SMallR1R2(int N, int nPT, spin_t *xs, spin_t *ys, int r1, int r2, double *qs, double *rs, double *ts)   // r1 = 0, ... NR-1
{

  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  //  int q   = (gti) /(N)  ;     	// termperature index 
  int q   = gti /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 

  int myq1 = permsConst[q+r1*nPT]; 
  int myq2 = permsConst[q+r2*nPT]; 

  spin_t ovq, ovr, ovt;

  if(ni<N)
    {

      spin_t x1 = xs[N*myq1+ni+r1*N*nPT];
      spin_t x2 = xs[N*myq2+ni+r2*N*nPT];
      spin_t y1 = ys[N*myq1+ni+r1*N*nPT];
      spin_t y2 = ys[N*myq2+ni+r2*N*nPT];
      
      ovq = (y1*y2+x1*x2);
      ovr = (-y1*y2+x1*x2);
      ovt = (y1*x2+x1*y2);
 
    }
     
  /////////////////////////////////////////
  __shared__ double  en[LBS];
  
  en[threadIdx.x] = ovq;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      qs[blockIdx.x] = en[0];
    }

  __syncthreads();
  en[threadIdx.x] = ovr;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      rs[blockIdx.x] = en[0];
    }

  __syncthreads();
  en[threadIdx.x] = ovt;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      ts[blockIdx.x] = en[0];
    }
  
  /////////////////////////////////////////   
}



__global__ void SMqs(int N,int nPT,spin_t *xs,spin_t *ys,double *qs)
{

  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  //  int q   = (gti) /(N)  ;     	// termperature index 
  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 

  int myq1 = permsConst [q]; 
  int myq2 = permsConst[q+nPT]; 

  spin_t ovs = 0.;

  if(ni<N)
    {

      spin_t x1 = xs[N*myq1+ni];
      spin_t x2 = xs[N*myq2+ni+N*nPT];
      spin_t y1 = ys[N*myq1+ni];
      spin_t y2 = ys[N*myq2+ni+N*nPT];
      
      ovs = (y1*y2+x1*x2);
 
    }
     
  /////////////////////////////////////////
  __shared__ double  en[LBS];
  
  en[threadIdx.x] = ovs;   
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      qs[blockIdx.x] = en[0];
    }
  
  /////////////////////////////////////////   
}


__global__ void SM_IFOr1r2 (int N, int nPT, spin_t * xs, spin_t * ys, int r1, int r2, double * ifo, double * norm1, double * norm2, double * mi)
{

  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;
  //  int q   = (gti) /(N)  ;     	// temperature index 
  int q   = gti /(LBS*gridDim.x/nPT);     	// temperature index 
  int ni  = gti - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  
  int myq1 = permsConst[q+r1*nPT]; 
  int myq2 = permsConst[q+r2*nPT]; 
  
  spin_t ovifo = 0., ovn1=0., ovn2=0.;
  
  if(ni<N) {
    spin_t x1 = xs[N*myq1+ni+r1*N*nPT];
    spin_t x2 = xs[N*myq2+ni+r2*N*nPT];
    spin_t y1 = ys[N*myq1+ni+r1*N*nPT];
    spin_t y2 = ys[N*myq2+ni+r2*N*nPT];
    
    spin_t I1 = x1*x1+y1*y1 - mi[ni+q*N];
    spin_t I2 = x2*x2+y2*y2 - mi[ni+q*N];      

    ovifo = I1*I2; 
    ovn1 = I1*I1; 
    ovn2 = I2*I2; 
  }
  
  /////////////////////////////////////////
  __shared__ double  en[LBS];
  
  en[threadIdx.x] = ovifo;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      ifo[blockIdx.x] = en[0];
    }

  __syncthreads();
  en[threadIdx.x] = ovn1;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      norm1[blockIdx.x] = en[0];
    }

  __syncthreads();
  en[threadIdx.x] = ovn2;
  __syncthreads();
  if( threadIdx.x == 0)
    {
      for(int j = LBS-1; j>0; j--)
	{en[j-1] += en[j];}
      norm2[blockIdx.x] = en[0];
    }

  /////////////////////////////////////////   
}




//////////////////////////////////////////////////////////////
//OTHER MEASURES:


//va chiamata <<<1,nPT>>>
__global__ void sumAllBlocksAndSquare1Replica(int nBpPTR,int ri,int nPT,double *blockData1,double *blockData2,double *data)
{
  int ti = threadIdx.x;
  double r1=0.,r2=0.;
  int myq=permsConst[ti+nPT*ri]; 
  
  for(int j=0;j<nBpPTR;j++)
    {
      r1 += blockData1[j+nBpPTR* myq];
      r2 += blockData2[j+nBpPTR* myq];
   }
  data[ti]=r1*r1+r2*r2;
}




// misura sulle configurazioni
__global__ void xx2(spin_t *xs,int N,int nPT,double *mx,double *mx2)
{
  __shared__ double  sx[LBS],sx2[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {
      
      double x=0.,x2=0.;  
      
      spin_t myx=xs[ni+myq*N];
      x+=myx;	
      x2+=myx*myx;
      
      /////////////////////////////////////////
      
      sx[ti] = x;
      sx2[ti] = x2;
      
      
    }
  else 
    {
      sx[ti] = 0.;
      sx2[ti] = 0.;
    }
  
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{sx[j-1] += sx[j];sx2[j-1] += sx2[j];}
      mx[bi] = sx[0];
      mx2[bi] = sx2[0];
      //      printf ("%d %d %d %f \n",q,bi,ni,en[0]);
    }
  /////////////////////////////////////////
  
}




// misura sulle configurazioni
__global__ void sumX(spin_t *xs,int N,int nPT,double *mx)
{
  __shared__ double  sx[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {
      sx[ti] = xs[ni+myq*N];      
    }
  else 
    {
      sx[ti] = 0.;
    }
  
    __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	sx[j-1] += sx[j];
      mx[bi] = sx[0];
    }
  /////////////////////////////////////////
  
}


//va chiamato <<<nPT,N>>>
__global__ void suminPTorder(spin_t *xs,int N,int nPT,double *mx)
{
  __shared__ double  sx[Size];
  
  int q = blockIdx.x;
  int ti = threadIdx.x;

  // Global index:
  //  int gti = ti+ bi*blockId.x;
  // int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  // int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 

  

  if(q<nPT)
    {
      //      int myq=permsConst[q];
      int myq=q;


      if((ti<N))
	{
	  sx[ti] = xs[ti+myq*N];      
	}
      else 
	{
	  sx[ti] = 0.;
	}
      
      __syncthreads();
      if( ti == 0)
	{
	  for(int j=N-1;j>0;j--)
	    sx[j-1] += sx[j];
	  mx[q] = sx[0];
	}
      /////////////////////////////////////////
    }
}




//measures averages over sites of radii and the wheighted angle
__global__ void rphi(spin_t *xs,spin_t *ys,int N,int nPT,double *mr,double *mphi,double *mphi2)
{
  __shared__ double  sr[LBS],sphi[LBS],sphi2[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {

      double r=0.,phi=0.,phi2=0.;  
      
      spin_t myx=xs[ni+myq*N];
      spin_t myy=ys[ni+myq*N];
      spin_t myr=mysqrt(myx*myx+myy*myy);
      if(myr>1.0E-40)
	{
	  spin_t myphi=asin(myy/myr)*myr;
	  r+=myr;	
	  phi+=myphi;
	  phi2+=myphi*myphi;
	}
      
      /////////////////////////////////////////
      
      sr[ti] = r;
      sphi[ti] = phi;
      sphi2[ti] = phi2;
      
    }
  else
    {
      sr[ti] = 0.;
      sphi[ti] = 0.;
      sphi2[ti] = 0.;
    }
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{sr[j-1] += sr[j];sphi[j-1] += sphi[j];sphi2[j-1] += sphi2[j];}
      mr[bi] = sr[0];
      mphi[bi] = sphi[0];
      mphi2[bi] = sphi2[0];
    }
  /////////////////////////////////////////
  
}




//measures averages over sites of radii and the wheighted angle
__global__ void rs(spin_t *xs,spin_t *ys,int N,int nPT,double *mr)
{

  __shared__ double  sr[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;


  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {

  double r=0.;  
  spin_t myx=xs[ni+myq*N];
  spin_t myy=ys[ni+myq*N];
  spin_t myr=mysqrt(myx*myx+myy*myy);
  r+=myr;	
  
  /////////////////////////////////////////
  
  sr[ti] = r;
  
    }
  else
    {
      sr[ti]=0.;
    }

  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{sr[j-1] += sr[j];}
      mr[bi] = sr[0];
    }
  /////////////////////////////////////////
    
}






__global__ void phis(spin_t *xs,spin_t *ys,int N,int nPT,double *mphi,double *mphi2)
{
  __shared__ double  sphi[LBS],sphi2[LBS];  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {


  double phi=0.,phi2=0.;  

  spin_t myx=xs[ni+myq*N];
  spin_t myy=ys[ni+myq*N];
  spin_t myr=mysqrt(myx*myx+myy*myy);
  spin_t myphi=asin(myy/myr);
  phi+=myphi;
  phi2+=myphi*myphi;
  
  /////////////////////////////////////////

  
  sphi[ti] = phi;
  sphi2[ti] = phi2;
  
    }
  else
    {
      sphi[ti]=0.;
      sphi2[ti]=0.;
    }

  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{sphi[j-1] += sphi[j];sphi2[j-1] += sphi2[j];}
      mphi[bi] = sphi[0];
      mphi2[bi] = sphi2[0];
    }
  /////////////////////////////////////////
    
}





#define soja N/20.


//va chiamata <<<N.nPT/LBS,LBS>>>
__global__ void energyWithoutCouplingsConditional_nodeBased (spin_t *xs, spin_t *ys, int N, int nPT, double *class1, double *class2, double *class3)
{
  __shared__ double  en1[LBS];
  __shared__ double  en2[LBS];
  __shared__ double  en3[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {

  
  //  int myq=permsConst[q]; 
  
  double e[3]; 
  e[0]=0.;
  e[1]=0.;
  e[2]=0.;

  const int auxmatrix[5]={0,1,1,1,2};    // how many large -> class  
  
  int d1 = tex1Dfetch(cumNbq_texRef,ni+1)-tex1Dfetch(cumNbq_texRef,ni);  
  
  for(int qi=0;qi<d1;qi++)
    {
      int howmanylarge=0;
      int myquad=tex1Dfetch(cumNbq_texRef,ni)+qi;
      int myquadm[4];
      spin_t x[4],y[4];
      
      for(int j=0;j<4;j++)
	{
	  myquadm[j]= tex1Dfetch(quads_texRef,4*myquad+j);
	  x[j] = xs[ myquadm[j] + myq*N ];
	  y[j] = ys[ myquadm[j] + myq*N ];	
	  double    r=sqrt(x[j]*x[j] +y[j]*y[j]);
	  if(r*r > soja)
	    {
	      howmanylarge++;
	    }
	  x[j] = x[j]/r;
	  y[j] = y[j]/r;
	  
	}
      
      int classindex=auxmatrix[howmanylarge];
      
      e[classindex] -= (+x[0]*x[1]*x[2]*x[3] 
			+y[0]*y[1]*y[2]*y[3] 
			-x[0]*x[1]*y[2]*y[3] 
			-y[0]*y[1]*x[2]*x[3] 
			+x[0]*y[1]*x[2]*y[3] 
			+y[0]*x[1]*y[2]*x[3] 
			+x[0]*y[1]*y[2]*x[3] 
			+y[0]*x[1]*x[2]*y[3] );      
      
    }	    
  
  
  //measuring the energy:
  /////////////////////////////////////////
  
  en1[ti] = e[0];
  en2[ti] = e[1];
  en3[ti] = e[2];

    }
  else
    {
      en1[ti] = 0.;
      en2[ti] = 0.;
      en3[ti] = 0.;
      
    }
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en1[j-1] += en1[j];}
      class1[bi] = en1[0];
    }
  if( ti == 1)
    {
      for(int j=LBS-1;j>0;j--)
	{en2[j-1] += en2[j];}
      class2[bi] = en2[0];
    }
  if( ti == 2)
    {
      for(int j=LBS-1;j>0;j--)
	{en3[j-1] += en3[j];}
      class3[bi] = en3[0];
    }
  /////////////////////////////////////////
    
}




//va chiamata <<<N.nPT/LBS,LBS>>>
__global__ void howmanyInClass(spin_t *xs,spin_t *ys,int N,int nPT,int *class1,int *class2,int *class3 )
{
  __shared__ int  en1[LBS];
  __shared__ int  en2[LBS];
  __shared__ int  en3[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()


  if(ni<N)
    {


  int auxmatrix[5];
  auxmatrix[0]=0;
  auxmatrix[1]=1;
  auxmatrix[2]=1;
  auxmatrix[3]=1;
  auxmatrix[4]=2;


  int howmany[3]; 
  howmany[0]=0.;
  howmany[1]=0.;
  howmany[2]=0.;


  int d1 = tex1Dfetch(cumNbq_texRef,ni+1)-tex1Dfetch(cumNbq_texRef,ni);  
  
  for(int qi=0;qi<d1;qi++)
    {
      int howmanylarge=0;
      
      int myquad=tex1Dfetch(cumNbq_texRef,ni)+qi;
      
      int myquadm[4];
      spin_t x[4],y[4];
      
      for(int j=0;j<4;j++)
	{
	  myquadm[j]= tex1Dfetch(quads_texRef,4*myquad+j);
	  myquadm[j]=myquadm[j]   + myq*N ;
	  x[j] = xs[ myquadm[j] ];
	  y[j] = ys[ myquadm[j] ];	
	  double    r=sqrt(x[j]*x[j] +y[j]*y[j]);
	  if(r*r > soja)
	    {
	      howmanylarge++;
	    }
	  x[j] = x[j]/r;
	  y[j] = y[j]/r;
	  
	}
      
      int classindex=auxmatrix[howmanylarge];
      howmany[classindex]++;
    }	    
  
  
  //measuring the energy:
  /////////////////////////////////////////
  
  en1[ti] = howmany[0];
  en2[ti] = howmany[1];
  en3[ti] = howmany[2];

    }
  else
    {
      en1[ti] = 0;
      en2[ti] = 0;
      en3[ti] = 0;
    }
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en1[j-1] += en1[j];}
      class1[bi] = en1[0];
    }
  if( ti == 1)
    {
      for(int j=LBS-1;j>0;j--)
	{en2[j-1] += en2[j];}
      class2[bi] = en2[0];
    }
  if( ti == 2)
    {
      for(int j=LBS-1;j>0;j--)
	{en3[j-1] += en3[j];}
      class3[bi] = en3[0];
    }
  /////////////////////////////////////////

    
}







//////////////////////////////////////
//////////////////////////////////////

//va chiamata <<<N.nPT/LBS,LBS>>>
__global__ void energyWithoutCouplingsSquareConditional_nodeBased(spin_t *xs,spin_t *ys,int N,int nPT,double *class1,double *class2,double *class3 )
{
  __shared__ double  en1[LBS];
  __shared__ double  en2[LBS];
  __shared__ double  en3[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;


  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  if(ni<N)
    {


  int auxmatrix[5];
  auxmatrix[0]=0;
  auxmatrix[1]=1;
  auxmatrix[2]=1;
  auxmatrix[3]=1;
  auxmatrix[4]=2;


  double e[3]; 
  e[0]=0.;
  e[1]=0.;
  e[2]=0.;

  int d1 = tex1Dfetch(cumNbq_texRef,ni+1)-tex1Dfetch(cumNbq_texRef,ni);
  
  for(int qi=0;qi<d1;qi++)
    {
      int howmanylarge=0;
      
      int myquad=tex1Dfetch(cumNbq_texRef,ni)+qi;
      
      int myquadm[4];
      spin_t x[4],y[4];
      
      for(int j=0;j<4;j++)
	{
	  myquadm[j]= tex1Dfetch(quads_texRef,4*myquad+j);
	  myquadm[j]=myquadm[j]   + myq*N ;

	  x[j] = xs[ myquadm[j] ];
	  y[j] = ys[ myquadm[j] ];	
	  double    r=sqrt(x[j]*x[j] +y[j]*y[j]);
	  if(r*r > soja)
	    {
	      howmanylarge++;
	    }
	  x[j] = x[j]/r;
	  y[j] = y[j]/r;
	  
	}
      
      int classindex=auxmatrix[howmanylarge];
      
      double daux	 =(+x[0]*x[1]*x[2]*x[3] 
			   +y[0]*y[1]*y[2]*y[3] 
			   -x[0]*x[1]*y[2]*y[3] 
			   -y[0]*y[1]*x[2]*x[3] 
			   +x[0]*y[1]*x[2]*y[3] 
			   +y[0]*x[1]*y[2]*x[3] 
			   +x[0]*y[1]*y[2]*x[3] 
			   +y[0]*x[1]*x[2]*y[3] );
      e[classindex] += daux*daux;
      
    }	    
  
  
  //measuring the energy:
  /////////////////////////////////////////
  
  en1[ti] = e[0];
  en2[ti] = e[1];
  en3[ti] = e[2];
    }

  else
    {
      en1[ti] = 0.;
      en2[ti] = 0.;
      en3[ti] = 0.;
    }

  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{en1[j-1] += en1[j];}
      class1[bi] = en1[0];
    }
  if( ti == 1)
    {
      for(int j=LBS-1;j>0;j--)
	{en2[j-1] += en2[j];}
      class2[bi] = en2[0];
    }
  if( ti == 2)
    {
      for(int j=LBS-1;j>0;j--)
	{en3[j-1] += en3[j];}
      class3[bi] = en3[0];
    }
  /////////////////////////////////////////

    

}





///////////////////////////////////
//Measures based on quadruplets:
///////////////////////////////////
__global__ void magInQuadruplette(int N, spin_t *xs,spin_t *ys,int Nq,int nPT,double *dataDev)
{
  __shared__ double  coseno[LBS];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  
  int q   = (gti) /(blockDim.x*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*blockDim.x*(gridDim.x/nPT); 	  // quad index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()
  
  if(ni<Nq)
    {
      
      double x=0;  
      
      int i1=tex1Dfetch(quads_texRef,4*ni);
      int i2=tex1Dfetch(quads_texRef,4*ni+1);
      int i3=tex1Dfetch(quads_texRef,4*ni+2);
      int i4=tex1Dfetch(quads_texRef,4*ni+3);
      

      printf("%d %d %d %d %d  \n",ni,i1,i2,i3,i4);


      double x1=xs[myq*N + i1];
      double x2=xs[myq*N + i2];
      double x3=xs[myq*N + i3];
      double x4=xs[myq*N + i4];
      double y1=ys[myq*N + i1];
      double y2=ys[myq*N + i2];
      double y3=ys[myq*N + i3];
      double y4=ys[myq*N + i4];

      x += x1*x2+y1*y2;
      x += x1*x3+y1*y3;
      x += x1*x4+y1*y4;
      x += x3*x2+y3*y2;
      x += x3*x4+y3*y4;
      x += x2*x4+y2*y4;
      
      coseno[ti] = x;
   
    }
  else 
    {
      coseno[ti] = 0.;
    }
  
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{coseno[j-1] += coseno[j];}
      dataDev[bi] = coseno[0];
    }
  /////////////////////////////////////////
  
}




__global__ void magInQuadruplette_nodebased(spin_t *xs,spin_t *ys,int N,int nPT,double *cos,double *cos2)
{
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  int q   = (gti) /(blockDim.x*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*blockDim.x*(gridDim.x/nPT); 	  // node index inside one temperature 

  __shared__ double  cs[LBS];
  __shared__ double  c2s[LBS];

  //  int myq=permsConst[q]; 
  int myq=q;

  double c=0.,c2=0.;

  if(ni<N) {

      int mi=ni;
      int d1 = tex1Dfetch(cumNbq_texRef,mi+1)-tex1Dfetch(cumNbq_texRef,mi);
            
      for(int qi=0;qi<d1;qi++)
	{
	  int myquad=tex1Dfetch(cumNbq_texRef,mi)+qi;
	  
	  int myquadm;	
	  spin_t x[4],y[4];
	  for(int j=0;j<4;j++)
	    {
	      myquadm= tex1Dfetch(quads_texRef,4*myquad+j);
	      myquadm=myquadm   + myq*N ;
	      x[j] = xs[ myquadm ];
	      y[j] = ys[ myquadm ];	

	    }
	  

	  double aux=+x[0]*x[1]+y[0]*y[1] 
	    +x[0]*x[2]+y[0]*y[2] 
	    +x[0]*x[3]+y[0]*y[3] 
	    +x[1]*x[2]+y[1]*y[2] 
	    +x[1]*x[3]+y[1]*y[3] ;


	  c+= aux;
	  c2+=aux*aux;
	}

  
  //measuring the energy:
  /////////////////////////////////////////
  cs[ti] = c;
  c2s[ti] = c2;

    }
  else 
    {
      cs[ti] = 0.;
      c2s[ti] = 0.;
    }
  
  __syncthreads();
  if( ti == 0)
    {
      for(int j=LBS-1;j>0;j--)
	{
	  cs[j-1] += cs[j];
	  c2s[j-1] += c2s[j];
	}
      cos[bi] = cs[0];
      cos2[bi] = c2s[0];
    }
  /////////////////////////////////////////

}



__global__ void modeIntensityForAverage(int N,int nPT, int r1, spin_t *xs,spin_t *ys,double *data, int c)
{
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;
  int TDim = blockDim.x*gridDim.x/nPT;
  int q   = gti / TDim;     	// temperature index 
  int ni  = gti - q*TDim;       // node index inside one temperature 
  int myq = permsConst[q+r1*nPT]; 

  if(ni<N && q<nPT) {
    spin_t x=xs[ni+N*myq+r1*N*nPT], y=ys[ni+N*myq+r1*N*nPT];
    data[ni+q*Size] = (data[ni+q*Size]*c + x*x + y*y)/(c+1);
  }
  
  return;
}


#ifdef MAXNFREQ
// frequency correlation function of spins x.x+y.y
__global__ void freqCF(spin_t *xs,spin_t *ys,int N,int nPT,double *dataCF)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  double fcf[MAXNFREQ];
  __shared__ double  sfcf[MAXNFREQ];

  for(int j=0;j<MAXNFREQ;j++)
    {
      sfcf[j]=0.;
    }


  for(int j=0;j<MAXNFREQ;j++)
    {
      fcf[j]=0.;
    }
  
  if(ni<N)
    {
      spin_t x=xs[ni+N*myq],y=ys[ni+N*myq];
      int iindex=wsiConst[ni];
      
      for(int j=0;j<N;j++)
	if(j!=ni)
	{
	  spin_t xn=xs[j+N*myq],yn=ys[j+N*myq];
	  int dw=abs(iindex-wsiConst[j]);
	  fcf[dw] +=  xn*x+yn*y; //spin-spin
     	}
    
    }
  
      for(int myti=0;myti<blockDim.x;myti++)
	{
	  __syncthreads();
	  if( ti == myti )
	    {
	      for(int j=0;j<MAXNFREQ;j++)
		{
		  sfcf[j] += fcf[j];
		}
	    }
	}
      /////////////////////////////////////////
    

	  __syncthreads();
	  if( ti == 0 )
	    {
	      for(int j=0;j<MAXNFREQ;j++)
		{
		  dataCF[j+bi*MAXNFREQ] = sfcf[j];
		}
	    }

}
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
__global__ void freqCFphases(spin_t *xs,spin_t *ys,int N,int nPT,double *dataCF)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq=q;   // the right order is done in sumAllBlocks1replica()

  double fcf[MAXNFREQ];
  __shared__ double  sfcf[MAXNFREQ];

  for(int j=0;j<MAXNFREQ;j++)
    {
      sfcf[j]=0.;
    }


  for(int j=0;j<MAXNFREQ;j++)
    {
      fcf[j]=0.;
    }
  
  if(ni<N)
    {
      spin_t x=xs[ni+N*myq],y=ys[ni+N*myq];
      int iindex=wsiConst[ni];
      
      for(int j=0;j<N;j++)
	if(j!=ni)
	{
	  spin_t xn=xs[j+N*myq],yn=ys[j+N*myq];
	  int dw=abs(iindex-wsiConst[j]);
	  double aux=sqrt(xn*xn+yn*yn)*sqrt(x*x+y*y);
	  if(aux>1.0E-12)
	    fcf[dw] +=  (xn*x+yn*y)/aux;//cos(phi1-phi2)
     	}
    
    }
  
      for(int myti=0;myti<blockDim.x;myti++)
	{
	  __syncthreads();
	  if( ti == myti )
	    {
	      for(int j=0;j<MAXNFREQ;j++)
		{
		  sfcf[j] += fcf[j];
		}
	    }
	}
      /////////////////////////////////////////
    

	  __syncthreads();
	  if( ti == 0 )
	    {
	      for(int j=0;j<MAXNFREQ;j++)
		{
		  dataCF[j+bi*MAXNFREQ] = sfcf[j];
		}
	    }

}
////////////////////////////////////////////////////////////////







//va chiamata <<<1,nPT>>>
__global__ void sumFCFallBlocks1Replica(int nBpPTR,int ri,int nPT,double *blockData,double *data)
{
  int ti = threadIdx.x;
  double r[MAXNFREQ];
  for(int n=0;n<MAXNFREQ;n++)
    r[n]=0.;

  int myq=permsConst[ti+nPT*ri]; 
  
  for(int j=0;j<nBpPTR;j++)
    for(int i=0;i<MAXNFREQ;i++)
    {
      r[i] += blockData[i + j*MAXNFREQ + nBpPTR*MAXNFREQ*myq];
    }

    for(int i=0;i<MAXNFREQ;i++)
    {
      data[ti*MAXNFREQ + i]=r[i];
    }
}


//va chiamata <<<1,nPT>>>
__global__ void sumFCFallBlocks1ReplicaVersors(int nBpPTR,int ri,int nPT,double *blockData,double *data)
{
  int ti = threadIdx.x;
  double r[2*MAXNFREQ];
  for(int n=0;n<2*MAXNFREQ;n++)
    r[n]=0.;
  
  int myq = permsConst[ti+nPT*ri]; 
  
  for(int j=0;j<nBpPTR;j++)
    for(int i=0;i<2*MAXNFREQ;i++)
      {
	r[i] += blockData[i + j*2*MAXNFREQ + nBpPTR*2*MAXNFREQ*myq];
      }
  
  for(int i=0;i<2*MAXNFREQ;i++)
    {
      data[ti*2*MAXNFREQ + i] = r[i];
    }
}


// intensity correlation function vs frequency
__global__ void freqCFintensity(spin_t *xs,spin_t *ys,int N,int nPT,double *dataCF)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  double fcf[MAXNFREQ];
  __shared__ double  sfcf[MAXNFREQ];

  for(int j=0;j<MAXNFREQ;j++)
    {
      sfcf[j]=0.;
    }


  for(int j=0;j<MAXNFREQ;j++)
    {
      fcf[j]=0.;
    }
  
  if(ni<N)
    {
      spin_t x=xs[ni+N*myq],y=ys[ni+N*myq];
      int iindex=wsiConst[ni];
      
      for(int j=0;j<N;j++)
	if(j!=ni)
	{
	  spin_t xn=xs[j+N*myq],yn=ys[j+N*myq];
	  int dw=abs(iindex-wsiConst[j]);
	  //	  fcf[dw] +=  xn*x+yn*y; //spin-spin
	  fcf[dw] +=  (xn*xn+yn*yn)*(x*x+y*y); //Intensity
     	}
    
    }
  
  for(int myti=0;myti<blockDim.x;myti++)
    {
      __syncthreads();
      if( ti == myti )
	{
	  for(int j=0;j<MAXNFREQ;j++)
	    {
	      sfcf[j] += fcf[j];
	    }
	}
    }
  /////////////////////////////////////////
  
  
  __syncthreads();
  if( ti == 0 )
    {
      for(int j=0;j<MAXNFREQ;j++)
	{
	  dataCF[j+bi*MAXNFREQ] = sfcf[j];
	}
    }
  
  return;
}





#if (FREQ_ENABLE == 1)

__global__ void ISpectrumForAverage(spin_t *xs,spin_t *ys,int N,int nPT,double *dataCF)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;

  int TDim = blockDim.x*gridDim.x/nPT;
  int q   = gti / TDim  ;     	// termperature index 
  int ni  = gti  - q*TDim; 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  int iindex=0;

  double fcf=0.;
  __shared__ double  sfcf[MAXNFREQ];

  if(ti == 0)
    for(int j=0;j<MAXNFREQ;j++)
      sfcf[j]=0.;    

  if(ni<N && q<nPT)
    {
      spin_t x=xs[ni+N*myq], y=ys[ni+N*myq];      
      iindex = wsiConst[ni];      

      fcf = sqrt(x*x+y*y);
    }
  
  for(int myti=0; myti<blockDim.x; myti++)
    {
      __syncthreads();
      
      if( ti == myti )
	{
	  sfcf[iindex] += fcf;
	}
    }
  /////////////////////////////////////////  
  
  __syncthreads();
  if( ti == 0 )
    for(int j=0;j<MAXNFREQ;j++)
      {
	dataCF[j+bi*MAXNFREQ] = sfcf[j];
      }
  
  return;
}

// Spin spectrum
__global__ void VersorSpectrumForAverage(spin_t *xs,spin_t *ys,int N,int nPT,double *dataVersor)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;

  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  int iindex=0;

  double fcf[2]={0.};
  __shared__ double  sfcf[2*MAXNFREQ];

  if(ti == 0)
    for(int j=0;j<2*MAXNFREQ;j++)
      sfcf[j]=0.;    

  if(ni<N)
    {
      spin_t x=xs[ni+N*myq], y=ys[ni+N*myq];      
      spin_t r = sqrt(x*x+y*y);
      iindex = wsiConst[ni];      

      if(r > 0.) {
	fcf[0] += x/r;
	fcf[1] += y/r;
      }
    }
  
  for(int myti=0; myti<blockDim.x; myti++)
    {
      __syncthreads();
      
      if( ti == myti )
	{	  
	  sfcf[2*iindex] += fcf[0];
	  sfcf[2*iindex+1] += fcf[1];	  
	}
    }
  /////////////////////////////////////////
  
  
  __syncthreads();
  if( ti == 0 )
    for(int j=0;j<2*MAXNFREQ;j++)
      {
	dataVersor[j+bi*2*MAXNFREQ] = sfcf[j];
      }
  
  
  return;
}



#elif (FREQ_ENABLE==2)

// if the frequencies are all different we can simplify the previous kernels


__global__ void ISpectrumForAverage(spin_t *xs,spin_t *ys,int N,int nPT,double *dataCF)
{

  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;

  int TDim = blockDim.x*gridDim.x/nPT;
  int q   = gti / TDim  ;     	// termperature index 
  int ni  = gti  - q*TDim; 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  int iindex=0;

  if(ti==0)
    for(int j=0;j<Size;j++)
      dataCF[j+bi*Size] = 0.;                // in questo caso si potrebbe fare meglio prendendo dataCF di dimensione LBS e non MAXNFREQ

  __syncthreads();
  
  if(ni<N && q<nPT)
    {
      spin_t x=xs[ni+N*myq], y=ys[ni+N*myq];   
      iindex = wsiConst[ni];

      dataCF[iindex+bi*Size] = sqrt(x*x+y*y);
    }
  
  return;
}


//<<<length,1>>>
__global__ void setToZero(spin_t *data)
{
  int bi = blockIdx.x;
  data[bi]=0.;
}

// Spin spectrum
__global__ void VersorSpectrumForAverage(spin_t *xs,spin_t *ys,int N,int nPT,double *dataVersor)
{
 
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti + bi*blockDim.x;

  int TDim = blockDim.x*gridDim.x/nPT;
  int q   = gti / TDim  ;     	// termperature index 
  int ni  = gti  - q*TDim; 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  int iindex=0;

  if(ni<N && q<nPT)
    {
      spin_t x=xs[ni+N*myq], y=ys[ni+N*myq];      
      spin_t r = sqrt(x*x+y*y);
      iindex = wsiConst[ni];      

      if(r > 1.e-10) {
	dataVersor[2*iindex +q*2*MAXNFREQ] = x/r;
	dataVersor[2*iindex +1 +q*2*MAXNFREQ] = y/r;
      }
      else{
	dataVersor[2*iindex +q*2*MAXNFREQ] = 0.;
	dataVersor[2*iindex +1 +q*2*MAXNFREQ] = 0.;

      }
    }
  
  return;
}






//////////////////////////////////////////////////////////////////////
__global__ void totalCorrelationIntensity(spin_t *xs,spin_t *ys,int N,int nPT,double *CF)
{
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  CF[ni +N*myq ]=0.;
  
  if((ni<N) && (myq<nPT))
    {
      spin_t x=xs[ni+N*myq],y=ys[ni+N*myq];
      
      for(int j=0;j<N;j++)
	if(j!=ni)
	{
	  spin_t xn=xs[j+N*myq],yn=ys[j+N*myq];
	  CF[ni+N*myq] +=  (xn*xn+yn*yn)*(x*x+y*y); //Intensity
     	}
    }
  return;
}
//////////////////////////////////////////////////////////////////////
__global__ void totalCorrelationPhases(spin_t *xs,spin_t *ys,int N,int nPT,double *CF)
{
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  // Global index:
  int gti = ti+ bi*blockDim.x;
  int q   = (gti) /(LBS*gridDim.x/nPT)  ;     	// termperature index 
  int ni  = gti  - q*LBS*(gridDim.x/nPT); 	  // node index inside one temperature 
  int myq = q;   // the right order is done in sumAllBlocks1replica()

  CF[ni +N*myq ]=0.;
  
  if((ni<N) && (myq<nPT))
    {
      spin_t x=xs[ni+N*myq],y=ys[ni+N*myq];
      spin_t norm=sqrt(x*x+y*y);
      
      for(int j=0;j<N;j++)
	if(j!=ni)
	{
	  spin_t xn=xs[j+N*myq],yn=ys[j+N*myq];
	  spin_t normj=sqrt(xn*xn+yn*yn);
	  if((norm>1.0E-15) && (normj>1.0E-15))
	    CF[ni+N*myq] +=   (xn*x+yn*y)/(norm*normj) ; 
     	}
    }
  return;
}
///////////////////////////////////////////////////////////////////////////////77


#endif




//#include "brutta.cpp"

#endif



