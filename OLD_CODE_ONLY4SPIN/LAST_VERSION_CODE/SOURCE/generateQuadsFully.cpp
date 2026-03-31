void generateQuadsFully(int N,int *wsindices,   vector<vector< vector< int > > > &quadlist, int *cumNbq,double fraction,int seed)
{
  
  open_rng(seed);  
  double x=0.;

  int DILUITE;
  if (fraction >= 0.9999) DILUITE=0;

  //    int cnt_test=0;
  for(int i=0;i<N;i++)
    for(int j=i+1;j<N;j++)
      for(int k=j+1;k<N;k++)
	for(int l=k+1;l<N;l++)
	  {
	    //	    	    cnt_test++;
	    //	    cout <<  i << j << k << l << endl ;
	    
	    if( DILUITE)
	      {
	       	x=rand_double();
	      }
	    else
	      x=0;
	    
	    if(x < fraction) {
	      
	      if( abs(wsindices[i]+wsindices[j]-wsindices[k]-wsindices[l]) <= GAMMA-1)     // primo ordinamento
		{
		  int myints[] = {i,j,k,l};
		  vector<int> vmyquad (myints, myints + sizeof(myints) / sizeof(int) );
		  quadlist[i].push_back(vmyquad);
		  quadlist[j].push_back(vmyquad);
		  quadlist[k].push_back(vmyquad);
		  quadlist[l].push_back(vmyquad);
		}
	      if( abs(wsindices[i]+wsindices[l]-wsindices[k]-wsindices[j]) <= GAMMA-1)     // secondo ordinamento
		{
		  int myints[] = {i,l,k,j};
		  vector<int> vmyquad (myints, myints + sizeof(myints) / sizeof(int) );
		  quadlist[i].push_back(vmyquad);
		  quadlist[j].push_back(vmyquad);
		  quadlist[k].push_back(vmyquad);
		  quadlist[l].push_back(vmyquad);
		}
	      if( abs(wsindices[i]+wsindices[k]-wsindices[j]-wsindices[l]) <= GAMMA-1)     // terzo ordinamento
		{
		  int myints[] = {i,k,j,l};
		  vector<int> vmyquad (myints, myints + sizeof(myints) / sizeof(int) );
		  quadlist[i].push_back(vmyquad);
		  quadlist[j].push_back(vmyquad);
		  quadlist[k].push_back(vmyquad);
		  quadlist[l].push_back(vmyquad);
		}	    
	      
	    }
	    
	  }
  
  
  cumNbq[0]=0;
  for(int i=0;i<N;i++){
    cumNbq[i+1]=quadlist[i].size()+cumNbq[i];
  }
  
  
  /*
  for(int i=0;i<N;i++)
    cout << wsindices[i] << endl;

  for(int i=0;i<quadlist[0].size();i++){
    for(int k=0;k<4;k++)
      cout << " " << quadlist[0][i][k];
    cout << " ; " ;
  }
  cout << endl;

  cout << "total quads " << cumNbq[N] << endl; 
  cout << "quadlist[].size " << quadlist[0].size() << endl;

  //  cout << "cnt_test=" << cnt_test << endl;

  */

  close_rng();

}

