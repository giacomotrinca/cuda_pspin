

void  generateTetrads(int N,int Nt,vector<vector<int > > &tetrads,int seed)
{
   open_rng(seed);
   int count =0 ;

   while (count < Nt)
     {
       int v1=N*rand_double();
       int v2=N*rand_double();
       int v3=N*rand_double();
       int v4=N*rand_double();


       if( (v1!=v2) && (v1!=v3) &&(v1!=v4) &&(v2!=v3) &&(v2!=v4) &&(v3!=v4) )
	 {
	   int myints[4] = {v1,v2,v3,v4};
	   vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;
	   sort (mytetrad.begin(), mytetrad.end());
	   //vedose cel'ho:
	   bool celho=false;
	   for(int j=0;j<tetrads.size();j++)
	     {
	       if(tetrads[j]==mytetrad) {celho=true; break;}
	     }
	   if(!celho) 
	 {
	   tetrads.push_back(mytetrad);
	   count++;
	 }
	 }
     }
   
   
   close_rng();
}












//this function picks 4 random numbers and the resulting tetrad is added to the list of tetrads (i.e., unordered list of four nodes -with a reference order i1<i2<i3<i4) whether it is not already on it (searched by a hashing)
void  generateTetradsHashing(int N,int Nt,vector<vector<int > > &tetrads,int seed)
{

  int dimHashSpace = 6*Nt;
  hashInfo hashList(dimHashSpace);

  open_rng(seed);
  int count =0 ;
  long   int N3=N*N*N;
  long   int N2=N*N;
  
  
  //printf("Inside generateTetradsHashing \n");
  //printf("Begin of Cycle \n");
  
  while (count < Nt)
    {
       int v1=N*rand_double();
       int v2=N*rand_double();
       int v3=N*rand_double();
       int v4=N*rand_double();

       if( (v1!=v2) && (v1!=v3) &&(v1!=v4) &&(v2!=v3) &&(v2!=v4) &&(v3!=v4) )
	 {


	   int myints[4] = {v1,v2,v3,v4};
	   vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;
	   sort (mytetrad.begin(), mytetrad.end());
	   
	   

	   long int V = mytetrad[0] + mytetrad[1]*N + mytetrad[2]*N2 + mytetrad[3]*N3;

	   
	   int hi=hashList.my_hashing_save2(V);


	   //cout << V <<  " " << hi << " " << count << endl;
	   

	   
	   if( hi == -1 ) 
	     {
	       count++;
      	       tetrads.push_back(mytetrad);
	       // cout << V <<  " " << hi << " " << count << endl;
	     }

	 }
     }
   close_rng();
   hashList.free();

   //printf("End of Cycle \n");

}



void  generateTetradsHashing_giacomo(int N, vector<vector<int > > &tetrads,int seed)
{

  int NTOT = 0 ;

  open_rng(seed);

  for(int i=0; i<N; i++){
    for(int j=0; j<i; j++){
      for(int l=0; l<j; l++){
	for(int k=0; k<l; k++){
	  
	  int myints[4] = {i,j,l,k};
	  vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;
	  tetrads.push_back(mytetrad);
	  
	  NTOT++;
      	       
	}	
      }
    }
  }
  
  for(int i=NTOT; i>1; i--){
    int i_rand  = (int)floor((double)i*rand_double());
    int myints_temp[4]; 
    
    myints_temp[0] = tetrads[i-1][0];
    myints_temp[1] = tetrads[i-1][1];
    myints_temp[2] = tetrads[i-1][2];
    myints_temp[3] = tetrads[i-1][3];
    
    tetrads[i-1] = tetrads[i_rand];

    tetrads[i_rand][0]=myints_temp[0];
    tetrads[i_rand][1]=myints_temp[1];
    tetrads[i_rand][2]=myints_temp[2];
    tetrads[i_rand][3]=myints_temp[3];
    
  }
  
  //tetrads.resize(Nt);
  close_rng();

}
  









//this function picks 4 random numbers and the resulting tetrad is added to the list of tetrads (i.e., unordered list of four nodes -with a reference order i1<i2<i3<i4) whether it is not already on it (searched by a hashing)
void  generateTetradsHashingAbsent(int N,int Nt,vector<vector<int > > &tetrads,int seed)
{


   open_rng(seed);
   int count =0 ;
long   int N3=N*N*N;
long   int N2=N*N;


  int totNbTetrads=N*(N-1)*(N-2)*(N-3)/24;
  int Nat=totNbTetrads-Nt;


  int dimHashSpace = 2*Nat;
  hashInfo hashList(dimHashSpace);
  vector<vector<int > > absentTetrads;

   while (count < Nat)
     {
       int v1=N*rand_double();
       int v2=N*rand_double();
       int v3=N*rand_double();
       int v4=N*rand_double();


       if( (v1!=v2) && (v1!=v3) &&(v1!=v4) &&(v2!=v3) &&(v2!=v4) &&(v3!=v4) )
	 {
	   
	   
	   int myints[4] = {v1,v2,v3,v4};
	   vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;
	   sort (mytetrad.begin(), mytetrad.end());
	   
	   
	   long int V = mytetrad[0] + mytetrad[1]*N + mytetrad[2]*N2 + mytetrad[3]*N3;
	       
	   int hi=hashList.my_hashing_save2(V);
   
	   if( hi == -1 ) 
	     {
	       count++;
      	       absentTetrads.push_back(mytetrad);
	     }

	 }
     }


//costruisco la lista di tetradi presenti da quella delle assenti:

   for(int j1=0;j1<N;j1++)
     for(int j2=0;j2<j1;j2++)
       for(int j3=0;j3<j2;j3++)
	 for(int j4=0;j4<j3;j4++)
	   {
	     long int V = j4 + j3*N + j2*N2 + j1*N3;

	     int hi=hashList.my_hashing_position(V);

	     if(hi==-1)
	       {
		 int myints[4] = {j4,j3,j2,j1};
		 vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;
		 tetrads.push_back(mytetrad);
	       }
	   }


   
   absentTetrads.clear();
   close_rng();
   hashList.free();
}












//this function picks 4 random numbers and the resulting tetrad is added to the list of tetrads (i.e., unordered list of four nodes -with a reference order i1<i2<i3<i4) whether it is not already on it (searched by a hashing)
void  generateTetradsHashingSortedExtraction(int N,int Nt,vector<vector<int > > &tetrads,int seed)
{

  int dimHashSpace = 2*Nt;
  hashInfo hashList(dimHashSpace);

   open_rng(seed);
   int count =0 ;
long   int N3=N*N*N;
long   int N2=N*N;

   
   while (count < Nt)
     {

       double sum=0.;

       double v1 = sum = sum- log(rand_double());
       double v2 = sum = sum- log(rand_double());
       double v3 = sum = sum- log(rand_double());
       double v4 = sum = sum- log(rand_double());

       sum = sum- log(rand_double());


       v1 = v1*N/sum;
       v2 = v2*N/sum;
       v3 = v3*N/sum;
       v4 = v4*N/sum;

       


              int myints[4] = {v1, v2, v3, v4};


	   if((myints[0] < myints[1]) && (myints[1] < myints[2]) && (myints[2] < myints[3]))
	 {

	   vector<int> mytetrad (myints, myints + sizeof(myints) / sizeof(int) ) ;

	   long int V = mytetrad[0] + mytetrad[1]*N + mytetrad[2]*N2 + mytetrad[3]*N3;

	   //	   	   cout << mytetrad[0]  << " " << mytetrad[1]  << " " << mytetrad[2]  << " " << mytetrad[3] << endl;
	   
	   int hi=hashList.my_hashing_save2(V);


	   //	   cout << V <<  " " << hi << " " << count << endl;
	   

	   
	   if( hi == -1 ) 
	     {
	       count++;
      	       tetrads.push_back(mytetrad);
	     }

	 }
     }
   close_rng();
   hashList.free();
}






