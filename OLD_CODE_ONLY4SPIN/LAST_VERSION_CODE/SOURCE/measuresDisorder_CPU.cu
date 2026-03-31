
fprintf(myfile1,"%d ",time+nbequil);

for(int q=0;q<NPT;q++)
  fprintf(myfile1,"%.14f ",nrg[q]); 




#if ALL_MEASURES


for(int bi=0;bi<NPT;bi++){
  double thisR = 0;
  for(int k=0; k<N; k++){
    double x = xs[perms[bi]*N+k];
    double y = ys[perms[bi]*N+k];
    thisR += sqrt(x*x+y*y);
  }
  fprintf(myfile1,"%.16le ",thisR/(1.*N));
 }

#if measure_M2

//measure of \vec M^2
for(int bi=0;bi<NPT;bi++){
  double thisMx = 0;
  double thisMy = 0;
  for(int k=0; k<N; k++){
    double x = xs[perms[bi]*N+k];
    double y = ys[perms[bi]*N+k];
    thisMx += x;
    thisMy += y;
  }
  fprintf(myfile1,"%.16le ",(thisMx*thisMx+thisMy*thisMy)/(1.*N*N)); // ATTENTION!! it is Mx**2+My**2 where Mx is the TOTAL x-magnetization
 }


#else

//measure of M_x 
for(int bi=0;bi<NPT;bi++){
  double thisMx = 0;
  for(int k=0; k<N; k++){
    double x = xs[perms[bi]*N+k];
    thisMx += x;
  }
  fprintf(myfile1,"%.16le ",(thisMx)/(1.*N));   // ATTENTION!! it is Mx, the TOTAL x-magnetization
 }


//measure of M_y 
for(int bi=0;bi<NPT;bi++){
  double thisMy = 0;
  for(int k=0; k<N; k++){
    double y = ys[perms[bi]*N+k];
    thisMy += y;
  }
  fprintf(myfile1,"%.16le ",(thisMy)/(1.*N));   // ATTENTION!! it is My, the TOTAL y-magnetization
 }



#endif  // measure_M2
#endif // ALL_MEASURES

fprintf(myfile1,"\n");
fflush(myfile1);




/*********************************************************************/
/*GPU spinglass overlap measures*/
/********************************************************************/
#if NR>=2

int idx=0;
for(int r1=1;r1<NR;r1++)
  for(int r2=0;r2<r1;r2++) {     
    
    fprintf(myfile4,"%d ",idx);
    fprintf(myfile5,"%d ",idx);
    fprintf(myfile6,"%d ",idx);
    
    for(int bi=0;bi<NPT;bi++){
      double ovq=0, ovr=0, ovt=0;
      for(int k=0; k<N; k++){
	double x1 = xs[r1*N*nPT+perms[bi+r1*nPT]*N+k];
	double y1 = ys[r1*N*nPT+perms[bi+r1*nPT]*N+k];
	double x2 = xs[r2*N*nPT+perms[bi+r2*nPT]*N+k];
	double y2 = ys[r2*N*nPT+perms[bi+r2*nPT]*N+k];
	ovq += (y1*y2+x1*x2);
	ovr += (-y1*y2+x1*x2);
	ovt += (y1*x2+x1*y2); 
      }  
      
      fprintf(myfile4,"%.14le ",ovq/N); 
      fprintf(myfile5,"%.14le ",ovr/N); 
      fprintf(myfile6,"%.14le ",ovt/N); 
      
    }
    fprintf(myfile4,"\n");
    fprintf(myfile5,"\n");
    fprintf(myfile6,"\n");       
    idx++;
  }



/*
#if NR>=2


cudaMemcpy(xsDev,xs,size,cudaMemcpyHostToDevice); 
cudaMemcpy(ysDev,ys,size,cudaMemcpyHostToDevice); 


 int idx=0;
 for(int r1=1;r1<NR;r1++)
   for(int r2=0;r2<r1;r2++) {
     SMallR1R2<<<nBlocks,LBS>>>(N, nPT, xsDev, ysDev, r2, r1, dataDevNRG_nodeBased, dataDevX_nodeBased, dataDevY_nodeBased);   // r1 = 0, ... NR-1
     ///////////////////////////////////////////////////////////////////////
     sumAllOrderedBlocks<<<1,nPT>>>(nBlocks/nPT,dataDevNRG_nodeBased,qsDev);
     cudaMemcpy(qs,qsDev,nPT*sizeof(double),cudaMemcpyDeviceToHost); 
     fprintf(myfile4,"%d ",idx);
     for(int q=0;q<nPT;q++)
       fprintf(myfile4,"%.14le ",qs[q]/Size); 
     fprintf(myfile4,"\n");
     ///////////////////////////////////////////////////////////////////////
     sumAllOrderedBlocks<<<1,nPT>>>(nBlocks/nPT,dataDevX_nodeBased,qsDev);
     cudaMemcpy(qs,qsDev,nPT*sizeof(double),cudaMemcpyDeviceToHost); 
     fprintf(myfile5,"%d ",idx);
     for(int q=0;q<nPT;q++)
       fprintf(myfile5,"%.14le ",qs[q]/Size); 
     fprintf(myfile5,"\n");
     ///////////////////////////////////////////////////////////////////////
     sumAllOrderedBlocks<<<1,nPT>>>(nBlocks/nPT,dataDevY_nodeBased,qsDev);
     cudaMemcpy(qs,qsDev,nPT*sizeof(double),cudaMemcpyDeviceToHost); 
     fprintf(myfile6,"%d ",idx);
     for(int q=0;q<nPT;q++)
       fprintf(myfile6,"%.14le ",qs[q]/Size); 
     fprintf(myfile6,"\n");
     ///////////////////////////////////////////////////////////////////////
     idx++;
   }
*/


#endif // NR>=2
 

//#include "extraMeasuresDisorder.cu"
