class hashInfo {
 private:

 int K; 		// number of elements of the hashing list
 int dim;		// dimension of the space
 vector<long int> H; 	// hashing list
 vector<int> L;		// auxiliary list (to deal with collisions)
 vector<int> MtoK;
 vector<int> KtoM;
  int Rmin;

 public:
/////////////////////////////////////////////////////////////////////
 hashInfo(int myK) // constructor. is it ok?
 {
  K=myK;
  H.assign(K,0);
  L.assign(K,-1);
  KtoM.assign(K,0);
  Rmin=K-1;
 }
/////////////////////////////////////////////////////////////////////
 void free()
 {
  KtoM.clear();
  MtoK.clear();
  H.clear();
  L.clear();
 }
/////////////////////////////////////////////////////////////////////
 int dimension()
 {int s=MtoK.size();
  return s; 
 }
/////////////////////////////////////////////////////////////////////
inline int hashing_save(long int);
/////////////////////////////////////////////////////////////////////
inline int my_hashing_save(long int);
inline int my_hashing_save2(long int);
/////////////////////////////////////////////////////////////////////
inline int hashing_position(long int);
/////////////////////////////////////////////////////////////////////
inline int my_hashing_position(long int);
/////////////////////////////////////////////////////////////////////
inline long int return_vector(int);
/////////////////////////////////////////////////////////////////////
void my_hash_vector(vector<int>&,vector<long int>);
/////////////////////////////////////////////////////////////////////
void hash_mag(int,int);
/////////////////////////////////////////////////////////////////////
void reorder(void);
/////////////////////////////////////////////////////////////////////
};


int h(long int I,int K)
{
 int aux=I/K;
  return (I-K*aux)+1;
//   return I%K+1;
}

inline int hashInfo::hashing_save(long int I)
{
  int i,R=K-1;

  i=h(I,K)-1;
  if(L[i]==-1) {H[i]=I;L[i]=0; return(i);}
  if(H[i]==I) return(i);
  while(L[i]>0) {i=L[i]-1; if(H[i]==I) return(i);}

  while( L[R]>-1 ) R--;
  L[i]=R+1;
  i=R;
  L[i]=0;
  H[i]=I; return(i);
}


//it saves I in the hashing list and returns -1 if I WERE NOT in the list
inline int hashInfo::my_hashing_save(long int I)
{
  int i,R=K-1;

  i=h(I,K)-1;
  if(L[i]==-1) {H[i]=I;L[i]=0; return(-1);}
  if(H[i]==I) return(i);
  while(L[i]>0) {i=L[i]-1; if(H[i]==I) return(i);}

  while( L[R]>-1 ) R--;
  L[i]=R+1;
  i=R;
  L[i]=0;
  H[i]=I; return(i);
}


//it saves I in the hashing list and returns -1 if I WERE NOT in the list
inline int hashInfo::my_hashing_save2(long int I)
{
  int i,R=Rmin;

  i=h(I,K)-1;
  if(L[i]==-1) {H[i]=I;L[i]=0; return(-1);}
  if(H[i]==I) return(i);
  while(L[i]>0) {i=L[i]-1; if(H[i]==I) return(i);}

  //non ce l'ho, dunque:

  while( L[R]>-1 ) R--;
  Rmin=R;
  L[i]=R+1;
  i=R;
  L[i]=0;
  H[i]=I; return(-1);
}


inline int hashInfo::hashing_position(long int I)
{
 return KtoM[hashing_save(I)];
}


//this function is as the precedent one except that it returns -1 if the element is not in the list and, in this case, it does not add the element to the hashing list 
inline int hashInfo::my_hashing_position(long int I)
{ int i;

  i=h(I,K)-1;
  if(L[i]==-1) return(-1);
  if(H[i]==I) return(KtoM[i]);
  while(L[i]>0) {i=L[i]-1; if(H[i]==I) return(KtoM[i]);}
  return(-1);
}

inline long int hashInfo::return_vector(int i)
{
 return H[MtoK[i]];
}


void hashInfo::my_hash_vector(vector<int>& n,vector<long int> w)
{
 for(int i=0;i<w.size();i++)
  n.push_back(KtoM[hashing_save(w[i])]);
}


void hashInfo::reorder(void)
{
 int i=-1;
 for(int k=0;k<K;k++)
  if(L[k]!=-1)
  {
   i=i+1;
   KtoM[k]=i;
   MtoK.push_back(k); 
  }
}

