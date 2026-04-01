
/*********************************************************************/
/* GPU spinglass IFO measures */
// in this version we print the configuration I_k at each time
/********************************************************************/
#if NR>=2

//cudaMemcpy(xs,xsDev,size,cudaMemcpyDeviceToHost); 
//cudaMemcpy(ys,ysDev,size,cudaMemcpyDeviceToHost); 

#if BINARY
for(int r1=0;r1<NR;r1++) 
  for(int bi=0;bi<nPT;bi++)
    for(int j=0;j<N;j++)
      Is[j + N*bi + N*NPT*r1] = xs[j + N*perms[bi+NPT*r1] + N*NPT*r1 ]*xs[j + N*perms[bi+NPT*r1] + N*NPT*r1] + ys[j + N*perms[bi+NPT*r1] + N*NPT*r1]*ys[j + N*perms[bi+NPT*r1] + N*NPT*r1];

fwrite (Is , sizeof(float), NR*N*NPT, myfile7);

#else
for(int r1=0;r1<NR;r1++) {
  for(int bi=0;bi<nPT;bi++)
    for(int j=0;j<N;j++) {
	fprintf(myfile7,"%f ", xs[j + N*perms[bi+NPT*r1] + N*NPT*r1 ]*xs[j + N*perms[bi+NPT*r1] + N*NPT*r1] + ys[j + N*perms[bi+NPT*r1] + N*NPT*r1]*ys[j + N*perms[bi+NPT*r1] + N*NPT*r1]);
      }
  fprintf(myfile7,"\n");
 }
#endif


#endif
