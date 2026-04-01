void copy_plaquettes_host_to_device(int Nplaqs,Plaqs_type * d_placchette, Plaqs_type * placchette){
  
  cudaMalloc((Plaqs_type **) &d_placchette,Nplaqs*sizeof(Plaqs_type));
  cudaMemcpy(d_placchette,placchette,Nplaqs*sizeof(Plaqs_type),cudaMemcpyHostToDevice);

  double * temp_ene[Nplaqs];
  double * temp_ene_new[Nplaqs];
  double * temp_J[Nplaqs];
  int * temp_flag[Nplaqs];
  double * temp_spin[Nplaqs];
  
  for(int s=0; s<Nplaqs; s++){
    
    cudaMalloc(&(temp_J[s]),sizeof(double));
    cudaMalloc(&(temp_spin[s]),4*sizeof(double));
  
  }
  
  
  for(int s=0; s<Nplaqs; s++){
    
    cudaMemcpy(&(d_placchette[s].J),&(temp_J[s]),sizeof(double *),cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_placchette[s].spin_index),&(temp_spin[s]),sizeof(double *),cudaMemcpyHostToDevice);
    
  }
  
  for(int s=0; s<Nplaqs; s++){
    
    cudaMemcpy(temp_J[s],&(placchette[s].J),sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(temp_spin[s],&(placchette[s].spin_index),4*sizeof(double),cudaMemcpyHostToDevice);
    
  }
  
  return;
  
}

void copy_de_plaqs_host_to_device(int Nplaqs, double ** d_de_plaqs, double ** de_plaqs){
  
  cudaMalloc((double **) d_de_plaqs,Nplaqs*sizeof(double *));
  
  for(int s=0; s<Nplaqs; s++)
    cudaMalloc((double **) &(d_de_plaqs[s]),sizeof(double));
  
  cudaMemcpy(d_de_plaqs,de_plaqs,sizeof(de_plaqs),cudaMemcpyHostToDevice);
  

  return;
  
}

void copy_spins_host_to_device(int Nspins, spin_t ** xs, spin_t ** ys, spin_t ** d_xs, spin_t ** d_ys){
  
  cudaMalloc((spin_t **) d_xs,Nspins*sizeof(spin_t));
  cudaMalloc((spin_t **) d_ys,Nspins*sizeof(spin_t));
  
  cudaMemcpy(d_xs,xs,Nspins*sizeof(spin_t),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ys,ys,Nspins*sizeof(spin_t),cudaMemcpyHostToDevice);
  
  
  return;
  
}

void copy_spin_couple_host_to_device(spin_t ** d_xs, spin_t ** d_ys){
  
  cudaMalloc((Plaqs_type **) d_xs,2*sizeof(spin_t *));
  cudaMalloc((Plaqs_type **) d_ys,2*sizeof(spin_t *));
  
  for(int i=0; i<2; i++){
    cudaMalloc((spin_t **) &(d_xs[i]),sizeof(spin_t));
    cudaMalloc((spin_t **) &(d_ys[i]),sizeof(spin_t));
  }

  return;

}
