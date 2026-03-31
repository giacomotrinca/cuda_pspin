double normalization2D(int L,double alpha)
{
  double d,Z=0.;

  for(int m=1;m<L;m++)
   for(int n=0;n<=m;n++)
   {
    d=m*m+n*n;     
    //    if(d!=0)
    {
     Z += pow(d,-0.5*alpha);
    }
   }

  return Z;
 }

double normalization2DredundanceAvoid(int L,double alpha)
{
 double d,Z=0.;
 int maxDists = L*(L-1)/2;

 hashInfo dists(maxDists);

  for(int m=1;m<L;m++)
   for(int n=0;n<=m;n++)
   {
    d=m*m+n*n;     
    //    if(d!=0)
    {
     if(-dists.my_hashing_save(d));
      Z += pow(d,-0.5*alpha);
    }
   }

  dists.free();

  return Z;
 }


// complex graph with probability propto the -alpha-th power of the distance in a 2D square lattice
////////////////////////////////////////////////
int Levy2DFBC(int L,double alpha,long seed)
 {
  double d2,Z,xi,p;
  int m,n,coordn,nblinks,count=0;
  int mx,my,nx,ny;
  /////////////////
  open_rng(seed);
  /////////////////
  N=L*L; 
  coordn=4;
  nblinks=coordn*N/2-2*L;

 // computing the normalization of the probability distribution:
  //     Z = normalization2D(L,alpha);
  Z=normalization2DredundanceAvoid(L,alpha);

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

  while(count < nblinks)
  {  
   m=0;n=m;
   while( (m==n) || (neighbours(m,n)) )
   {
    m= N*rand_double();
    n= N*rand_double();
   }
   
   mx=m%L; my=m/L;
   nx=n%L; ny=n/L;

   int intdist=(nx-mx)*(nx-mx)+(ny-my)*(ny-my);
   d2=(double) (intdist);
   p=pow(d2,-0.5*alpha)/Z;
   xi=rand_double();
   if(xi < p)  
   {
    nbs[m].push_back(n);      
    nbs[n].push_back(m);
    count++;
   }

   //    cout << count << " " << m << ","<<n << " " << sqrt(d2) << " " << p << " " << xi << " " << endl;
    
  }

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  close_rng();
  return count;
 }
////////////////////////////////////////////////



// complex graph with probability propto the -alpha-th power of the distance in a 2D square lattice
////////////////////////////////////////////////
int Levy2DPBC(int L,double alpha,long seed)
 {
  double d2,Z,xi,p;
  int m,n,coordn,nblinks,count=0;
  int mx,my,nx,ny;
  /////////////////
  open_rng(seed);
  /////////////////
  N=L*L; 
  coordn=4;
  nblinks=coordn*N/2;
  int Lhalfs=L/2;

 // computing the normalization of the probability distribution:
  Z=normalization2DredundanceAvoid(Lhalfs,alpha); // 		:) !

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

  while(count < nblinks)
  {  
   m=0;n=m;
   while( (m==n) || (neighbours(m,n)) )
   {
    m= N*rand_double();
    n= N*rand_double();
   }
   
   mx=m%L; my=m/L;
   nx=n%L; ny=n/L;

   int dx = (abs(mx-nx)-1)%(Lhalfs)+1;
   int dy = (abs(my-ny)-1)%(Lhalfs)+1;

   d2=(double) (dx*dx + dy*dy);
   p=pow(d2,-0.5*alpha)/Z;
   xi=rand_double();
   if(xi < p)  
   {
    nbs[m].push_back(n);      
    nbs[n].push_back(m);
    count++;
   }
  }

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  close_rng();
  return count;
 }
////////////////////////////////////////////////

//#include "Levy_variabledegree.cpp"



double normalizationCubic(int L,double alpha)
{
  double d,Z=0.;

  for(int m=1;m<L;m++)
   for(int n=0;n<=m;n++)
    for(int p=0;p<=n;p++)
    {
     d=m*m+n*n+p*p;     
     //    if(d!=0)
     {
      Z += pow(d,-0.5*alpha);
      //     cout << m<< " " << n << " " << p << " "  << d << endl;
     }
    }
  return Z;
 }

double normalizationCubicRedundanceAvoid(int L,double alpha)
{
 double d,Z=0.;

 int maxDists = L*L*L/2;
 hashInfo dists(maxDists);

  for(int m=1;m<L;m++)
   for(int n=0;n<=m;n++)
    for(int p=0;p<=n;p++)
    {
     d=m*m+n*n+p*p;     
     //    if(d!=0)
     int nonesiste = dists.my_hashing_save(d);
     if(nonesiste==-1)
     {
      Z += pow(d,-0.5*alpha);
     }
    }

 dists.free();

  return Z;
 }



// complex graph with probability propto the -alpha-th power of the distance in a simple cubic lattice
//usage: Li>=Lj>=Lk
////////////////////////////////////////////////
int randomFBCcubic(int Li,int Lj,int Lk,double alpha,long seed)
 {
   double d2,p,xi;
  int m,n,coordn,nblinks,count=0;
  int mx,my,mz,nx,ny,nz;
  /////////////////
  open_rng(seed);
  /////////////////
  N=Li*Lj*Lk; 
  coordn=6;
  nblinks=coordn*N/2-Li*Lj-Lj*Lk-Li*Lk;

 // computing the normalization of the probability distribution:
  // usage: I'm supposing Li is the largest of them
  //   double Z = normalizationCubic(Li,alpha);
    double Z = normalizationCubicRedundanceAvoid(Li,alpha);

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

  while(count < nblinks)
  {  

   m=0;n=m;
   while( (m==n) || (neighbours(m,n)) )
   {
    m= N*rand_double();
    n= N*rand_double();
   }
   
   mz=m/(Li*Lj); 	  nz=n/(Li*Lj);
   mx=m-mz*(Li*Lj);	  nx=n-nz*(Li*Lj);
   mx=mx/Li;		  nx=nx/Li;
   my=m-mz*(Li*Lj)-Li*mx; ny=n-nz*(Li*Lj)-Li*nx;

   d2=(double) ((nx-mx)*(nx-mx)+(ny-my)*(ny-my)+(nz-mz)*(nz-mz));
   p=pow(d2,-0.5*alpha)/Z;
   xi=rand_double();
   if(xi < p)  
   {
    nbs[m].push_back(n);      
    nbs[n].push_back(m);
    count++;
   }

   //  cout << count << " " << m << ","<<n << " " << sqrt(d2) << " " << p << " " << xi << " " << endl;
    
  }

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  close_rng();
  return count;
 }
////////////////////////////////////////////////


////////////////////////////////////////////////
int ErdosRenyi(int M,int nblinks,long seed)
 {
  int count=0;
  /////////////////
  open_rng(seed);
  /////////////////
 
// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(M,my_vector);

  while(count < nblinks)
  {  

   int m=0; int n=m;
   while( (m==n) || (neighbours(m,n)) )
   {
     m= M*rand_double();
     n= M*rand_double();
   }
   
    nbs[m].push_back(n);      
    nbs[n].push_back(m);
    count++;
  }

  for(int i=0;i<M;i++)
  {
    degree.push_back(nbs[i].size());
  }

  N=M;

  close_rng();
  return count;
 }
////////////////////////////////////////////////

//#include "newLevy.cpp"


// complex graph with probability propto the rho-th power of the distance in a 1D lattice with PBCs
////////////////////////////////////////////////
int Levy1DPBC(int L,double rho,long seed)
 {
  double d,Z,xi,p;
  int nblinks,count=0;
  /////////////////
  open_rng(seed);
  /////////////////
  N=L; 
  nblinks=N;

 // computing the normalization of the probability distribution:
  Z=0.5*L;

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

  while(count < nblinks)
  {  
   int m=0; int n=m;
   while( (m==n) || (neighbours(m,n)) )
   {
     m= N*rand_double();
     n= N*rand_double();
   }
   
   d = abs(m-n);
   if(d > 0.5*L) d = L-d;
   p=pow(d,-rho)/Z;
   xi=rand_double();
   if(xi < p)  
   {
    nbs[m].push_back(n);      
    nbs[n].push_back(m);
    count++;
   }

   //  cout << count << " " << m << ","<<n << " " << sqrt(d2) << " " << p << " " << xi << " " << endl;
    
  }

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  close_rng();
  return count;
 }
////////////////////////////////////////////////
