BEGIN{
    N=62;
    qmin=-1;
    qmax=1;
    NBINS=60;
    dq=(qmax-qmin)/NBINS;
    
    for(i=1;i<NBINS;i++){
	isto[i]=0;
	isto2[i]=0;
	count[i]=0;
	q[i]=qmin+(0.5+i)*dq;
	## printf("%g \n",q[i]);
    }
}{
    
    x=$1;
    y=$2;
    #printf("%g %g \n",x,y);

    for(i=1;i<NBINS;i++){
   	if(sqrt((x-q[i])**2)<0.0001){
   	    isto[i]+=y;
   	    isto2[i]+=y*y;
   	    count[i]++;
	}
    }
    
}END{
    
    for(i=1;i<NBINS;i++){

    	if(count[i]!=0){
	    mean=isto[i]/count[i];
	    var=sqrt(isto2[i]/count[i]-mean*mean);
	    var_mean=var/sqrt(count[i]);
	}else{
	    mean=0;
	    var=0;
	    var_mean=0;
	}
    	
	printf("%g %12.8e %12.8e \n",q[i],mean,var_mean);
    }
 }