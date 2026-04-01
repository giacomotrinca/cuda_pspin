BEGIN{
    N=96;
    NPT=38;
    Tmin=0.65;
    Tmax=1.6;
    dT=(Tmax-Tmin)/NPT;

    for(i=1;i<NPT;i++){
	cv[i]=0;
	cv2[i]=0;
	count[i]=0;
	t[i]=Tmin+i*dT;
       # print t[i];
    }
    }{
    temp=$1;
    calsp=$2;
    for(i=1;i<NPT;i++){
	if(sqrt((temp-t[i])**2) < 0.0001){
	    cv[i]+=calsp;
	    cv2[i]+=calsp*calsp;
	    count[i]++;
	}
	# print count[i];
    }
    }END{
    for(i=1;i<NPT;i++){
	cvmean=cv[i]/count[i]/N;
        cvvar=sqrt(cv2[i]/count[i]/N/N-cvmean*cvmean);
	cvvarmean=cvvar/sqrt(count[i]);
	print t[i],cvmean,cvvar,cvvarmean;
    }
}
