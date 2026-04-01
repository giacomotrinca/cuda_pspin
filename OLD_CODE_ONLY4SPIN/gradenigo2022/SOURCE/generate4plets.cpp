int generateDiscreteWss(int N,float *ws,int *wsindices,int seed)
{
 FILE *file = fopen("frequencies.dat","r");

 if(file==NULL) {cout << "FILE NOT FOUND! \n\n\n"; exit(1);}

 //I first read the file:
 float w,I;
 float Imin=1.0E16;
 int count =-1;

 ///////////////////////
 //look for the minimum
 do
   {
    int aux= fscanf(file,"%e %e",&w,&I);  
     count ++;
     if(I<Imin) 
       Imin=I;
   }while(!feof(file));
 
 rewind(file);
 ///////////////////////


 if(N<=count) return -1;

 ///////////////////////
 //store the arrays ws,Ss:
float  *filews=(float *) malloc(count*sizeof(float));
 float *S=(float *) malloc((count)*sizeof(float));
 float sum=0.;
 for(int j=0;j<count;j++)
   {
    int aux=     fscanf(file,"%e %e",&w,&I);  
     filews[j]=w;
     sum+=I-Imin;
     S[j]=sum;
   }
 ///////////////////////

 ///////////////////////
 //normalizzo la cumulativa:
 for(int j=0;j<count;j++)
   {
     S[j]=S[j]/sum;
   }
 ///////////////////////

 ///////////////////////
 //estraggo N numeri con quella distribuzione
 open_rng(seed);
 for(int j=0;j<N;j++)
   {
     double xi=rand_double(); 
	 int k=0;
	 while(S[k]<xi)
	   {k++;
	   }
	 ws[j]=filews[k];
	 wsindices[j]=k;
   }
 close_rng();
 ///////////////////////

 fclose(file);

 return count;
}



//receives a list of frequency indices and creates a list of quadrupletts satisfying FMC 
//the format of quadlist is such that quadruplets containing a given mode are consecutive, and each one appears 4times in quadlist
int generate4pletsCycles_discreteWs(int N,vector<vector<int> > cycles,vector<vector<int> > &quadlist,int *cumNbquads,int *ws)
{

  vector<int> vuoto;
  vector<vector<int> > auxquadlist;
  auxquadlist.assign(N,vuoto);

  int nbquads=cycles.size();

  for(int q=0;q<nbquads;q++)
    {
      vector<int> myquad=cycles[q];
      for(int qi=0;qi<4;qi++)
	{
	  int mi = cycles[q][qi];
	  auxquadlist[mi].push_back(q);
	}
    }
  //in this way, auxquadlist[mi] contains the list of the quadruplette indices of the quadruplettes containing the mode mi. Now I create  quadlist[]


  cumNbquads[0] = 0;
  int count = 0; // count verrà incrementato ogni volta che una quadrupletta è aggiunta alla lista
  for(int mi=0; mi<N; mi++) // ciclo su tutti i modi normali del sistema
    {

      for(int qmi=0; qmi<auxquadlist[mi].size(); qmi++) // ciclo su tutte le quadruplette di un certo modo 
	{
	  int myqi=auxquadlist[mi][qmi]; // estrae la quadrupletta qmi-iesima del mi-esimo modo
	  vector<int> vmyquad=cycles[myqi]; // crea un vettore a 4 posti con gli indici dei modi di quella quadrupletta

	  int sm1=vmyquad[0]; int sm2=vmyquad[1]; int sm3=vmyquad[2]; int sm4=vmyquad[3];

	  //I check whether the three permutations of the quad. satisy the FMC	  
	  if(abs(ws[sm1]+ws[sm2]-ws[sm3]-ws[sm4]) <= GAMMA-1)
	    {
	      count ++;
	      quadlist.push_back(vmyquad);
	    }
	  //secondo ordinamento (1432)
	  if(abs(ws[sm1]+ws[sm4]-ws[sm3]-ws[sm2]) <= GAMMA-1 )
	    {
	      count ++;
	      vmyquad[1]=sm4; vmyquad[3]=sm2; 
	      quadlist.push_back(vmyquad);
	    }

	  ///////////////////////////////////////////
	  //terzo oridinamento (1324)
	  if(abs(ws[sm1]+ws[sm3]-ws[sm2]-ws[sm4]) <= GAMMA-1 )
	    {
	      count ++;
	      vmyquad[0]=sm1; vmyquad[1]=sm3; vmyquad[2]=sm2; vmyquad[3]=sm4; 
	      quadlist.push_back(vmyquad);
	    }
	  
	}
      cumNbquads[mi+1]=count; // associo a ciascun modo mi un indice intero, contenente la posizione della prima della lista di quadruplette ad esso associate
    }

  return count;

}


// GIACOMO ----------------------------- 09/2017
//receives a list of random quadruplets and produce a list of plaquettes satisfying FMC 
void generate4pletsCycles_discreteWs_wPlaqs(vector<vector<int> > cycles,vector<Plaqs_type> &plaqlist, int *ws)
{

  int n_random_plaquettes = cycles.size();
  int plaq_index=0;


  for(int np=0; np<n_random_plaquettes;np++){
    
    int sm1=cycles[np][0]; 
    int sm2=cycles[np][1]; 
    int sm3=cycles[np][2]; 
    int sm4=cycles[np][3];

    //primo ordinamento (1234)
    if(abs(ws[sm1]+ws[sm2]-ws[sm3]-ws[sm4]) <= GAMMA-1){
      
      Plaqs_type placchetta;
      
      placchetta.spin_index[0]=sm1;
      placchetta.spin_index[1]=sm2;
      placchetta.spin_index[2]=sm3;
      placchetta.spin_index[3]=sm4;
      placchetta.J=0;
      //placchetta.ene=0;
      //placchetta.flag=0;
      
      plaqlist.push_back(placchetta);

      // spins[sm1].plaq_indices.push_back(plaq_index);
      // spins[sm2].plaq_indices.push_back(plaq_index);
      // spins[sm3].plaq_indices.push_back(plaq_index);
      // spins[sm4].plaq_indices.push_back(plaq_index);
      
      plaq_index++;
    }
    
    //secondo ordinamento (1432)
    if(abs(ws[sm1]+ws[sm4]-ws[sm3]-ws[sm2]) <= GAMMA-1 ){
    
      Plaqs_type placchetta;
      
      placchetta.spin_index[0]=sm1;
      placchetta.spin_index[1]=sm4; // scambiato con sm2
      placchetta.spin_index[2]=sm3;
      placchetta.spin_index[3]=sm2; // scambiato con sm4 
      placchetta.J=0;
      //placchetta.ene=0;
      //placchetta.flag=0;
      
      plaqlist.push_back(placchetta);
      
      // spins[sm1].plaq_indices.push_back(plaq_index);
      // spins[sm4].plaq_indices.push_back(plaq_index); // scambiato con sm2
      // spins[sm3].plaq_indices.push_back(plaq_index);
      // spins[sm2].plaq_indices.push_back(plaq_index); // scambiato con sm4 
      
      plaq_index++;
    }

    //terzo ordinamento (1324)
    if(abs(ws[sm1]+ws[sm3]-ws[sm2]-ws[sm4]) <= GAMMA-1 ){
      
      Plaqs_type placchetta;
      
      placchetta.spin_index[0]=sm1;
      placchetta.spin_index[1]=sm3; // scambiato con sm2
      placchetta.spin_index[2]=sm2; // scambiato con sm3 
      placchetta.spin_index[3]=sm4; 
      placchetta.J=0;
      //placchetta.ene=0;
      //placchetta.flag=0;
      
      plaqlist.push_back(placchetta);

      // spins[sm1].plaq_indices.push_back(plaq_index);
      // spins[sm3].plaq_indices.push_back(plaq_index); // scambiato con sm2
      // spins[sm2].plaq_indices.push_back(plaq_index); // scambiato con sm3 
      // spins[sm4].plaq_indices.push_back(plaq_index);
      
      plaq_index++;
    }
    
    
  }
  
  return;
  
}


void plaqs_from_vec_to_array(int N, int Nplaqs, vector<Plaqs_type> &placchette_vec, Plaqs_type * placchette){

  for(int iplaq=0; iplaq<Nplaqs; iplaq++ ){

    for(int ispin=0; ispin<4; ispin++){
      placchette[iplaq].spin_index[ispin]=placchette_vec[iplaq].spin_index[ispin];
      if(placchette[iplaq].spin_index[ispin]>N-1 || placchette[iplaq].spin_index[ispin]<0){
	printf("ERROR: placchette[%d]->spin_index[%d]=%d \n",iplaq,ispin,placchette[iplaq].spin_index[ispin]);
	printf("ERROR: spin index out of bounds \n");
	exit(1);
      }
    }

    placchette[iplaq].J=placchette_vec[iplaq].J;
    //placchette[iplaq].ene=placchette_vec[iplaq].ene;
    //placchette[iplaq].flag=placchette_vec[iplaq].flag; 
  
  }

  return;
}

//////////----------------------------Giacomo 10/2017
//////////----------------------------Da usare solo dopo aver inizializzato i couplings e le placchette 
///////////////////////////////////////////////////////////////////////////////////////////////////////


// void randomGaussianCouplingsp4_fromQUADStoPLAQS(vector<Plaqs_type> &placchette, int *quads,int *cumNbq,double * J_plaq) 
// {
  
//   int nPlaqs =placchette.size(); 

//   for (int q=0;q<nPlaqs;q++){
    
//     //placchette[q].flag=0;
    
//     int m1 = placchette[q].spin_index[0]; 
//     int m2 = placchette[q].spin_index[1]; 
//     int m3 = placchette[q].spin_index[2]; 
//     int m4 = placchette[q].spin_index[3]; 

//     int d=cumNbq[m1+1]-cumNbq[m1];
    
//     for(int q2=0;q2<d;q2++){

//       if(quads[4*(cumNbq[m1]+q2)]==m1 && quads[4*(cumNbq[m1]+q2)+1]==m2 && quads[4*(cumNbq[m1]+q2)+2]==m3 && quads[4*(cumNbq[m1]+q2)+3]==m4){
	
// 	if(placchette[q].flag==0){
// 	  placchette[q].J=J_plaq[cumNbq[m1]+q2];
// 	  placchette[q].flag=1;
// 	}else{
// 	  if(placchette[q].J!=J_plaq[cumNbq[m1]+q2]){
// 	    printf("ERROR: I am proposing to plaquette %d NOT the same value of J \n",q);
// 	    printf("ERROR: old J = %g \n",placchette[q].J);
// 	    printf("ERROR: new J = %g \n",J_plaq[cumNbq[m1]+q2]);
// 	    exit(1);
// 	  }else{
// 	    placchette[q].J=J_plaq[cumNbq[m1]+q2];
// 	  }
// 	}      
	
//       }
      
//     }
    
//   }
  
//   return;

// }
///////////////////////////////////////////////////////////////////////




int generate4pletsCycles_discreteWs_v2(int N,vector<vector<int> > cycles,vector<vector<int> > &quadlist,int *cumNbquads,int *ws)
{

  vector<int> vuoto;
  vector<vector<int> >  auxquadlist;
  auxquadlist.assign(N,vuoto);

  int nbquads=cycles.size();

  for(int q=0;q<nbquads;q++)
    {
      vector<int> myquad=cycles[q];
      for(int qi=0;qi<4;qi++)
	{
	  int mi = cycles[q][qi];
	  auxquadlist[mi].push_back(q);
	}
    }
  //in this way, auxquadlist[mi] contains the list of the quadruplette indices of the quadruplettes containing the mode mi. Now I create  quadlist[]


  cumNbquads[0] = 0;
  int count = 0;
  for(int mi=0; mi<N; mi++)
    {

	for(int qmi=0; qmi<auxquadlist[mi].size(); qmi++)
	{
	  int myqi=auxquadlist[mi][qmi];
	  vector<int> vmyquad=cycles[myqi];

	  int sm1=vmyquad[0]; int sm2=vmyquad[1]; int sm3=vmyquad[2]; int sm4=vmyquad[3];

	  if( (ws[sm1]==ws[sm2]) && (ws[sm1]==ws[sm3]) && (ws[sm1]==ws[sm4]) ){
	      count ++;
	      quadlist.push_back(vmyquad);
	  }
	  else 
	    {
	      
	      //I check whether the three permutations of the quad. satisy the FMC	  
	      if(abs(ws[sm1]+ws[sm2]-ws[sm3]-ws[sm4]) <= GAMMA-1)
		{
		  count ++;
		  quadlist.push_back(vmyquad);
		}
	      //secondo oridinamento (1432)
	      if(abs(ws[sm1]+ws[sm4]-ws[sm3]-ws[sm2]) <= GAMMA-1 )
		{
		  count ++;
		  vmyquad[1]=sm4; vmyquad[3]=sm2; 
		  quadlist.push_back(vmyquad);
		}
	      ///////////////////////////////////////////
	      //terzo oridinamento (1324)
	      if(abs(ws[sm1]+ws[sm3]-ws[sm2]-ws[sm4]) <= GAMMA-1 )
		{
		  count ++;
		  vmyquad[0]=sm1; vmyquad[1]=sm3; vmyquad[2]=sm2; vmyquad[3]=sm4; 
		  quadlist.push_back(vmyquad);
		}
	    }
	  
	}
	cumNbquads[mi+1]=count;
    }


  return count;

 /* for(int m=0;m<quadlist.size();m++)
   {     for(int j=0;j<4;j++)
       {
	 cout << quadlist[m][j]<< " " ;
       }
       cout << endl;   }*/

}



#if FREQ_ENABLE==3

//receives a list of frequency indices and creates a list of quadrupletts satisfying FMC 
//the format of quadlist is such that quadruplets containing a given mode are consecutive, and each one appears 4times in quadlist
int generate4pletsCycles_discreteWs_f3_v2(int N, vector<vector<int> > cycles, vector<vector<int> > &quadlist, int *cumNbquads, double *ws)
{

  vector<int> vuoto;
  vector<vector<int> >  auxquadlist;
  auxquadlist.assign(N,vuoto);

  int nbquads=cycles.size();

  for(int q=0;q<nbquads;q++)
    {
      vector<int> myquad=cycles[q];
      for(int qi=0;qi<4;qi++)
	{
	  int mi = cycles[q][qi];
	  auxquadlist[mi].push_back(q);
	}
    }
  //in this way, auxquadlist[mi] contains the list of the quadruplette indices of the quadruplettes containing the mode mi. Now I create  quadlist[]


  double DELTA = (FREQ_MAX-FREQ_MIN)/Size ;
  double LW = DELTA*GAMMA;

  cumNbquads[0] = 0;
  int count = 0;
  for(int mi=0; mi<N; mi++)
    {

	for(int qmi=0; qmi<auxquadlist[mi].size(); qmi++)
	{
	  int myqi=auxquadlist[mi][qmi];
	  vector<int> vmyquad=cycles[myqi];

	  int sm1=vmyquad[0]; int sm2=vmyquad[1]; int sm3=vmyquad[2]; int sm4=vmyquad[3];

	  if( abs(ws[sm1]-ws[sm2])<LW  && abs(ws[sm1]-ws[sm3])<LW && abs(ws[sm1]-ws[sm4])<LW && abs(ws[sm2]-ws[sm3])<LW && abs(ws[sm2]-ws[sm4])<LW && abs(ws[sm3]-ws[sm4])<LW ){
	      count ++;
	      quadlist.push_back(vmyquad);
	  }
	  else 
	    {
	      
	      //I check whether the three permutations of the quad. satisy the FMC	  
	      if(abs(ws[sm1]+ws[sm2]-ws[sm3]-ws[sm4]) <= LW)
		{
		  count ++;
		  quadlist.push_back(vmyquad);
		}
	      //secondo oridinamento (1432)
	      if(abs(ws[sm1]+ws[sm4]-ws[sm3]-ws[sm2]) <= LW )
		{
		  count ++;
		  vmyquad[1]=sm4; vmyquad[3]=sm2; 
		  quadlist.push_back(vmyquad);
		}
	      ///////////////////////////////////////////
	      //terzo oridinamento (1324)
	      if(abs(ws[sm1]+ws[sm3]-ws[sm2]-ws[sm4]) <= LW )
		{
		  count ++;
		  vmyquad[0]=sm1; vmyquad[1]=sm3; vmyquad[2]=sm2; vmyquad[3]=sm4; 
		  quadlist.push_back(vmyquad);
		}
	    }
	  
	}
	cumNbquads[mi+1]=count;
    }


  return count;

}


#endif



////////////////////////////////////////////////////////////////////////////
// ------------ GIACOMO------------- 2017 ---///////////////////////////////
////////////////////////////////////////////////////////////////////////////

// void build_dimers(int N, int Ndimers, Dimers_type * dimers){
  
//     // -------- Ciclo su tutti gli SPIN del sistema per creare le coppie di DIMERS ------- O(N^2)  
//   //###############################################################################################################
  
//   int idimer=0;
  
//   for(int i=0; i<N-1; i++){
//     for(int j=i+1; j<N; j++){
      
//       if(idimer<Ndimers){
	
// 	//printf("tento la coppia n° %d: spin %d spin %d \n",idimer,i,j);
	
// 	dimers[idimer].ind_spin_1=i;
// 	dimers[idimer].ind_spin_2=j;
	
// 	idimer++;
      
//       }else{
// 	printf("ERROR: stai cercando di inizializzare troppe coppie: n° coppia = %d \n",idimer);
// 	exit(1);
//       }
    
//     } // for(int j=i+1; j<N; j++) END CYCLE spins "j"
//   } //for(int i=0; i<N; i++) END CYCLE spins "i"
  
//   printf("# Ho concluso inizializzazione coppie \n");
//   printf("# Ndimers = %d \n",idimer);

//   return;

// }


