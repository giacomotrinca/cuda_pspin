#include "random.cpp"

class graph
{
 vector<vector<int> > nbs;
 vector<int> degree;
 int N;
 int coordN;


 ////////////////////////////////
 void PBC2D(int L,int **k)
 {
  for(int i=1;i<=L;i++)
   for(int j=1;j<=L;j++)
    k[i][j]=(i-1)*L+j-1;
 // PBC
  for(int i=1;i<=L;i++)
  {
   k[0][i]=k[L][i];
   k[L+1][i]=k[1][i];
   k[i][0]=k[i][L];
   k[i][L+1]=k[i][1];
  }
 }
 ////////////////////////////////

////////////////////////////////
void PBC2D(int L1,int L,int **k)
 {
  for(int i=1;i<=L1;i++)
   for(int j=1;j<=L;j++)
    k[i][j]=(i-1)*L+j-1;
 // PBC
  for(int i=1;i<=L;i++)
  {
   k[0][i]=k[L1][i];
   k[L1+1][i]=k[1][i];
  }
  for(int i=1;i<=L1;i++)
  {
   k[i][0]=k[i][L];
   k[i][L+1]=k[i][1];
   }
  }
 ////////////////////////////////


public:

 int getN(){return N;}
 int getc(){return coordN;}
 void getnbs(vector<vector<int> > &v)
 {v=nbs;}
 void getnbs(int k,vector<int>  &v)
 {v=nbs[k];}

 int getneighbour(int k,int m)
  {if(m<degree[k]) {return nbs[k][m];} else{return 0;}}

 int getdegree(int k)
 {return degree[k];}

 int getdegrees(vector<int> &d)
  {d=degree; return 0;}

 int neighbours(int m,int n)
 {
  for(int j=0;j<nbs[m].size();j++)
    if(nbs[m][j]==n){ return 1; }
  return 0;
 }

  void assignNbsDegrees(vector<vector<int> > vs,vector<int> ds)
  {
   nbs=vs;
   degree=ds;
  }

  void assignNcoordn(int newN,int c)
  {
    N=newN;
    coordN=c;
  }


 void free()
 {
   N=0;
  degree.clear();
  nbs.clear();
 }

////////////////////////////////////////////////
 int PBC2Dsquare(int L)
 {
  N=L*L; 
  coordN=4;
// creating chessboard with the corresponding boundary conditions:
  int **k;
  k=new int* [L+2];
  for(int j=0;j<=L+1;j++) k[j]=new int [L+2];
  PBC2D(L,k);
// creating neighbour matrix:
  vector<int> my_vector;
  my_vector.assign(4,0);
  nbs.assign(N,my_vector);

  int m,n;
  for(int i=0;i<N;i++)
  {
   m=(int) i/L+1;
   n=-(m-1)*L+i+1;

   nbs[i][0]=k[m-1][n];
   nbs[i][1]=k[m][n+1];
   nbs[i][2]=k[m+1][n];
   nbs[i][3]=k[m][n-1];
  }
  delete [] k;

  degree.assign(N,4);

  return N;
 }
////////////////////////////////////////////////


// strip square lattice with periodic BC:
////////////////////////////////////////////////
int PBCsquareStrip(int L1,int L)
 {
  N=L1*L; 
  coordN=4;
// creating chessboard with the corresponding boundary conditions:
  int **k;
  k=new int* [L1+2];
  for(int j=0;j<=L1+1;j++) k[j]=new int [L+2];
  PBC2D(L1,L,k);
// creating neighbour matrix:
  vector<int> my_vector;
  my_vector.assign(4,0);
  nbs.assign(N,my_vector);

  int m,n;
  for(int i=0;i<N;i++)
  {
   m=(int) i/L+1;
   n=-(m-1)*L+i+1;

   nbs[i][0]=k[m-1][n];
   nbs[i][1]=k[m][n+1];
   nbs[i][2]=k[m+1][n];
   nbs[i][3]=k[m][n-1];
  }
  delete [] k;
  return N;
 }
////////////////////////////////////////////////


// strip square lattice with periodic BC:
////////////////////////////////////////////////
int PBCtriangularStrip(int L1,int L)
 {
  N=L1*L; 
  coordN=6;
  vector<int> my_vector;
  my_vector.assign(6,0);
  nbs.assign(N,my_vector);

  int m,n;
  int mm1,mp1,nm1,np1;
  for(int i=0;i<N;i++)
  {
   m=(int) i/L;
   n=-m*L+i;

   mp1=((m+L+1)%L);
   mm1=((m+L-1)%L);
   np1=((n+L1+1)%L1);
   nm1=((n+L1-1)%L1);

   nbs[i][0]= mm1*L + n;
   nbs[i][1]= mm1*L + np1;
   nbs[i][2]= m*L + np1;
   nbs[i][3]= mp1*L + n;
   nbs[i][4]= mp1*L + nm1;
   nbs[i][5]= m*L + nm1;
  }

  degree.assign(N,6);

  my_vector.clear();
  return N;
 }
////////////////////////////////////////////////



// 2D square lattice in a strip with free boundary conditions:
////////////////////////////////////////////////
int FBCsquareStrip(int L1,int L)
 {
  int i;
  N=L1*L; 

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

 //  neighbours in the border: boundary conditions
 ///////////////////////
  for(int j=1;j<=L;j++)
  {
   // top border: neighbours in the bulk
   i=j-1;
   nbs[i].push_back( L+j-1 );
   // bottom border: neighbours in the bulk
   i=(L1-1)*L+j-1;
   nbs[i].push_back( (L1-2)*L+j-1 );
   }    
  for(int j=1;j<=L1;j++)
  {
   // left border: neighbours in the bulk
   i=(j-1)*L;
   nbs[i].push_back( (j-1)*L + 1 );
   // right border: neighbours in the bulk
   i=(j-1)*L+L1-1;
   nbs[i].push_back( (j-1)*L + L1-2 );
   }    
  for(int j=2;j<=L-1;j++)
  {
   // top border: neighbours in the top border
   i=j-1;
   nbs[i].push_back( j );
   nbs[i].push_back( j-2 );

   // bottom border: neighbours in the bottom border
   i=(L1-1)*L+j-1;
   nbs[i].push_back( (L1-1)*L+j );
   nbs[i].push_back( (L1-1)*L+j-2 );
   }    
  for(int j=2;j<=L1-1;j++)
  {
   // left border: neighbours in the left border
   i=(j-1)*L;
   nbs[i].push_back( j*L );
   nbs[i].push_back( (j-2)*L );

   // right border: neighbours in the right border
   i=(j-1)*L+L1-1;
   nbs[i].push_back( j*L + L1-1 );
   nbs[i].push_back( (j-2)*L + L1-1 );
   }    

 ///////////////////////

  // neighbours in the bulk:

  int iu,id,ir,il;

  for(int m=2;m<=L1-1;m++)
   for(int n=2;n<=L-1;n++)
   {
    i   =(m-1)*L+n-1;
    ir  =(m-1)*L+n;
    il  =(m-1)*L+n-2;
    iu  =(m  )*L+n-1;
    id  =(m-2)*L+n-1;
   
    nbs[i].push_back(ir);
    nbs[i].push_back(il);
    nbs[i].push_back(iu);
    nbs[i].push_back(id);
   }


  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  return N;
 }
////////////////////////////////////////////////



// 3D simple cubic lattice with free boundary conditions:
////////////////////////////////////////////////
 int FBCcubic(int Li,int Lj,int Lk)
 {
  int ni,nj,nk; // neighbours
  int m;
  N=Li*Lj*Lk; 

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

 //  neighbours in the border planes: boundary conditions
 ///////////////////////
  for(int r=1;r<Lj;r++)
   for(int s=1;s<Lk;s++)
   {
    // plane1: i=0 , j=r, k=s
    m  = r + s*Li*Lj;
    nj = r-1 + s*Li*Lj;
    nk = r + (s-1)*Li*Lj;

    nbs[m].push_back(nj);
    nbs[nj].push_back(m);
    nbs[m].push_back(nk);
    nbs[nk].push_back(m);

    // plane2: i=Li-1 , j=r, k=s
    /*    m  = (Li-1)*Lj + r + s*Li*Lj;
    nj = (Li-1)*Lj + r-1 + s*Li*Lj;
    nk = (Li-1)*Lj + r + (s-1)*Li*Lj;

    nbs[m].push_back(nj);
    nbs[nj].push_back(m);
    nbs[m].push_back(nk);
    nbs[nk].push_back(m);*/
   }

  for(int r=1;r<Li;r++)
   for(int s=1;s<Lk;s++)
   {
    //plane3: i=r , j=0 , k=s
    m  = r*Lj + s*Li*Lj;
    ni = (r-1)*Lj + s*Li*Lj;
    nk = r*Lj + (s-1)*Li*Lj;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);
    nbs[m].push_back(nk);
    nbs[nk].push_back(m);

    //plane4: i=r , j=Lj-1 , k=s
    /*    m  = r*Lj + Lj-1 + s*Li*Lj;
    ni = (r-1)*Lj + Lj-1 + s*Li*Lj;
    nk = r*Lj + Lj-1 + (s-1)*Li*Lj;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);
    nbs[m].push_back(nk);
    nbs[nk].push_back(m);*/
   }

  for(int r=1;r<Li;r++)
   for(int s=1;s<Lj;s++)
   {
    //plane5: i=r , j=s , k=0
    m  = r*Lj + s;
    ni = (r-1)*Lj + s;
    nj = r*Lj + s-1 ;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);
    nbs[m].push_back(nj);
    nbs[nj].push_back(m);

    //plane6: i=r , j=s , k=Lk-1
    /*    m  = r*Lj + s + (Lk-1)*Li*Lj;
    ni = (r-1)*Lj + s + (Lk-1)*Li*Lj;
    nj = r*Lj + s-1 + (Lk-1)*Li*Lj;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);
    nbs[m].push_back(nj);
    nbs[nj].push_back(m);*/
   }

  //arista1: i=j=0 k=r
  for(int r=1;r<Lk;r++)
  {
   m  = r*Li*Lj;
   nk = (r-1)*Li*Lj;
   nbs[m].push_back(nk);
   nbs[nk].push_back(m);
  }
  //arista2: i=k=0 j=r
  for(int r=1;r<Lj;r++)
  {
   m  =  r;
   nj = r-1;
   nbs[m].push_back(nj);
   nbs[nj].push_back(m);
  }
  //arista3: j=k=0 i=r
  for(int r=1;r<Li;r++)
  {
   m  = r*Lj;
   ni = (r-1)*Lj;
   nbs[m].push_back(ni);
   nbs[ni].push_back(m);
  }
 ///////////////////////



  // neighbours in the bulk:
  ///////////////////////

  for(int i=1;i<Li;i++)
   for(int j=1;j<Lj;j++)
    for(int k=1;k<Lk;k++)
    {
     m  = i*Lj + j + k*Li*Lj;

     ni = (i-1)*Lj + j + k*Li*Lj;
     nj = i*Lj + j-1 + k*Li*Lj;
     nk = i*Lj + j + (k-1)*Li*Lj;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);

    nbs[m].push_back(nj);
    nbs[nj].push_back(m);

    nbs[m].push_back(nk);
    nbs[nk].push_back(m);
   }

 ///////////////////////

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  return N;
 }
////////////////////////////////////////////////






// 3D simple cubic lattice with Periodic boundary conditions:
////////////////////////////////////////////////
 int PBCcubic(int Li,int Lj,int Lk)
 {
  int ni,nj,nk; // neighbours
  int m;
  N=Li*Lj*Lk; 

// creating neighbour matrix:
  vector<int> my_vector;
  nbs.assign(N,my_vector);

  ///////////////////////
  for(int i=0;i<Li;i++)
   for(int j=0;j<Lj;j++)
    for(int k=0;k<Lk;k++)
    {
     m  = i*Lj + j + k*Li*Lj;

     ni = ((i+1)%Li)*Lj + j + k*Li*Lj;
     nj = i*Lj + (j+1)%Lj + k*Li*Lj;
     nk = i*Lj + j + ((k+1)%Lk)*Li*Lj;

    nbs[m].push_back(ni);
    nbs[ni].push_back(m);

    nbs[m].push_back(nj);
    nbs[nj].push_back(m);

    nbs[m].push_back(nk);
    nbs[nk].push_back(m);
    }

 ///////////////////////

  for(int i=0;i<N;i++)
  {
    degree.push_back(nbs[i].size());
  }

  return N;
 }
////////////////////////////////////////////////

// #include "testGraphs.cpp"

// 1D chain with periodic BC:
////////////////////////////////////////////////
int PBC1Dchain(int L)
 {
  N=L; 
  coordN=2;
// creating neighbour matrix:
  vector<int> my_vector;
  my_vector.assign(2,0);
  nbs.assign(N,my_vector);

  for(int i=0;i<N;i++)
  {
   nbs[i][0]=(i-1+N)%N;
   nbs[i][1]=(i+1)%N;
  }
  degree.assign(N,2);
  return N;
 }
////////////////////////////////////////////////


// 1D chain with periodic BC:
////////////////////////////////////////////////
  int BetheFixedC(int c,int generation)
 {
   vector<int> vuoto;
   vector<vector<int> > aux;
   aux.assign(generation,vuoto);

  //1st generation:
   nbs.assign(c+1,vuoto);
   for(int i=1;i<=c;i++)
     {
       nbs[0].push_back(i);
       nbs[i].push_back(0);
       aux[0].push_back(i);
     }

   //up to now I have stored c+1 vertices in nbs[]:
   int count = c; 

   //higher generations:
   for(int g=1;g<generation;g++)
     {
       aux.push_back(vuoto);
       //who are in the precedent generation:
       for(int p=0;p<aux[g-1].size();p++)
	 {
	   int v=aux[g-1][p];
	   for(int k=0;k<c-1;k++)
	     {
	       count++;
	       nbs.push_back(vuoto);
	       nbs[v].push_back(count);
	       nbs[count].push_back(v);
	       aux[g].push_back(count);
	     }
	 }
     }

   degree.assign(count+1,0);
   for(int j=0;j<=count;j++)
     degree[j]=nbs[j].size();

   aux.clear();
   N=count+1;
   coordN=c;
   return N;
 }
////////////////////////////////////////////////


// 1D chain with periodic BC:
////////////////////////////////////////////////
  int fullyConnected(int n)
 {
   vector<int> vuoto;
   nbs.assign(n,vuoto);

   for(int j=0;j<n;j++)
     for(int i=0;i<j;i++)
       {
	 nbs[i].push_back(j);
	 nbs[j].push_back(i);
       }

   degree.assign(n,n-1);
   N=n;
   coordN=N-1;
   return N;
 }
////////////////////////////////////////////////




#include "randomgraphs.cpp"
  //#include "graphspectra.cpp"

// computation of the clustering coefficient
////////////////////////////////////////////////
double ClusteringC(void)
{
  double c=0.;
  int nb1,nb2,d;

 for(int j=0;j<N;j++)
 {
  d=degree[j];

  for(int n=0;n< d;n++)
   for(int m=0;m<n;m++)
   {
    nb1=nbs[j][n];
    nb2=nbs[j][m];

    for(int k=0;k< degree[nb1] ;k++)
    {
      if( nbs[nb1][k]==nb2 ) { c = c + 2./(d*(d-1.)); }
    }
 
   }
 }
 
 return c/N;
}
////////////////////////////////////////////////


int  vectorToArrayDegrees(int *adegrees)
{
  adegrees[0]=0;
  int nneigbs=0;
  for(int j=0;j<N;j++)
  {
    nneigbs+=degree[j];
    adegrees[j+1] = nneigbs;
  }
  return nneigbs;
}

void  vectorToArrayNeighbors(int *aneighbors)
{
  int count =0;
  for(int j=0;j<N;j++)
  {
    for(int m=0;m<degree[j];m++)
      aneighbors[count++]=nbs[j][m];
  }
}


};


