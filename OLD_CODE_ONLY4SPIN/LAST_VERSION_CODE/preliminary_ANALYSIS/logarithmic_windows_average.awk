BEGIN{
    NSPIN=96;
    window=2**10;
    x=0;
    y=0;
    y2=0;	
    i=0;
    j=0;
    n=0;
    ncol_temp=var1;
    ncol=var2;
}{

    temperature=$ncol_temp;

    if($1>50){
	if($1<window){
	    x+=$1;
	    y+=$ncol*NSPIN;
	    y2+=$ncol*$ncol*NSPIN*NSPIN;
	    n++;
	}else{
	    t[j]=x/n;
	    e[j]=y/n;
	    var_e[j]=y2/n-e[j]*e[j];
	    j++;
	    x=0;
	    y=0;
	    y2=0;
	    n=0;
	    window=window*2;
	}
    }
    
}END{
    
    ##printf("#T = %g \n",temperature);
    
    for(k=0;k<j;k++) 
	printf("%g %g %g %g \n",temperature,t[k],e[k],var_e[k]);
    
 }
 
