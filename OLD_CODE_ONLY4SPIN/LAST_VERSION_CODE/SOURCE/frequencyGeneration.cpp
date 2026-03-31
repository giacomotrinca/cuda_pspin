#if FREQ_ENABLE == 1
  int nbFreq = generateDiscreteWss(N,ws,wsindices,seed2);                 // generazione frequenze con input da file
  
#if _GainMax_ < 1.e-8
    for(int i=0;i<N;i++){    gain[i] = 0.; }
#else
generateGain(N,wsindices,gain,_GainMax_ / sqrt(_Tref_));                   // generazione gain dallo stesso file di input con massimo _GainMax_
#endif

#elif FREQ_ENABLE == 2

  int nbFreq = N;

/////////////////////////////////////////////////////////////////////////
// gaussian gain
double mySigma = N/4.;
double myMean = N/2.-0.5;
for(int i=0;i<N;i++){
  ws[i] = i;
  wsindices[i] = i ;
  gain[i] = _GainMax_ * ( exp(-(i-myMean)*(i-myMean)/(2.*mySigma*mySigma) )  ) / ( sqrt(twopi) * mySigma * sqrt(_Tref_) );   // gain gaussiano con sigma = mySigma
 }

#else 
  int nbFreq = 1;
  for(int i=0;i<N;i++){
    ws[i] = 1.;
    wsindices[i] = 0;
    gain[i] = 0.;
 }
#endif



#ifdef MAXNFREQ
  if(nbFreq > MAXNFREQ)  
    {
      cout << "\n#main: ERROR: number of frequencies exceding the constant variable NBFREQ. exiting" << endl << endl; 
      exit(1);
    }
#endif

