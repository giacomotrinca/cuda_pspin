// -*- c++ -*-

// void logN_GPU_summation_prelim(int N, Sum_type * ene){

//   vector<int> par;
//   vector<int> odd;
//   vector<int> nblocks;

//   // ---------------------------------------------- ////////

//   int number=N;
//   int nstep=0;

//   while(number>1){

//     if(number%2==0){ 
//       par.push_back(1);
//       odd.push_back(-1);
//       number = number/2;
//       nblocks.push_back(number);
//     }else{ 
//       par.push_back(0);   
//       odd.push_back(number-1);
//       number = (number - 1)/2;
//       nblocks.push_back(number);
//     }

//     nstep++;
    
//   }
  
//   ene->n_iterations = nstep;

//   ene->parity  = (int *)calloc(nstep,sizeof(int));
//   ene->oddity  = (int *)calloc(nstep,sizeof(int));
//   ene->nblocks = (int *)calloc(nstep,sizeof(int));


//   for(int i=0;i<nstep;i++){

//     ene->parity[i] = par[i];

//     ene->oddity[i] = odd[i];

//     ene->nblocks[i]= nblocks[i];
    
//     //printf("i = %d; x_parity = %d; x_oddity = %d; x_nblocks = %d \n",i, ene->parity[i], ene->oddity[i], ene->nblocks[i]);
//   }
  
//   return;

// }

