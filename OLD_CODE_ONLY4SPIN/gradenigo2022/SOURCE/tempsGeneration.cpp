
#if EQUISPACEDTS
  double deltaT=(Tmax-Tmin)/nPT;
  for(int q=0;q<nPT;q++)
    {
      beta[q]= 1./(Tmin+q*deltaT);
    }
#else
  double Tmin=Tc/2.;
  double deltaT=(1.-1./8.-1./2.)*Tc/(1.*nPT/8.);
  for(int q=0;q<nPT/8;q++)
    {
      beta[q]= 1./(Tmin+q*deltaT);
    }
  //second interval around Tc
  Tmin=Tc*(1.-1./8.);
  deltaT=Tc/(4.*(3.*nPT/4.));
  for(int q=0 ;q < 3*nPT/4;q++)
    {
      beta[q + nPT/8 ]= 1./(Tmin+q*deltaT);
    }
  //third interval:
  Tmin=Tc*(1.+1./8.);
  deltaT=(3./2. - 9./8.)*Tc/(1.*nPT/8.);
 for(int q=0;q<nPT/8;q++)
    {
      beta[q + nPT/8 + 3*nPT/4]= 1./(Tmin+q*deltaT);
    }
#endif

