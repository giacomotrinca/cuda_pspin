#!/bin/bash

i=1
Nsamples=250
NPT=42
N=62

mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N

mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM
#mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS
mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/MARGINAL_INTENSITIES
mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PLAQS_OVERLAP_ISTO
mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_IFO
mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_ISTO

temp=0  

while [ $temp -lt $NPT ]; do
    
    mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM/T$temp
    #mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS/T$temp
    mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/MARGINAL_INTENSITIES/T$temp
    mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PLAQS_OVERLAP_ISTO/T$temp
    mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_IFO/T$temp
    mkdir /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_ISTO/T$temp
    
    temp=$[ $temp + 1 ]
	
done



while [ $i -lt $[$Nsamples+1] ]; do
    
    temp=0  

    while [ $temp -lt $NPT ]; do
        
	

	cp sample$i/spectrum0_s2s4_T_$temp.dat   /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM/T$temp/spectrum0_s2s4_T_$temp-sample$i.dat
        
	cp sample$i/spectrum1_s2s4_T_$temp.dat   /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM/T$temp/spectrum1_s2s4_T_$temp-sample$i.dat
        
	cp sample$i/spectrum2_s2s4_T_$temp.dat   /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM/T$temp/spectrum2_s2s4_T_$temp-sample$i.dat
        
	cp sample$i/spectrum3_s2s4_T_$temp.dat    /home/niedda/RANDOM_LASER/ALL_DATA_N$N/SPECTRUM/T$temp/spectrum3_s2s4_T_$temp-sample$i.dat


	
	
        #cp sample$i/participation_ratios_r0_T_$temp.dat  /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS/T$temp/participation_ratios_r0_T_$temp-sample$i.dat
        
	#cp sample$i/participation_ratios_r1_T_$temp.dat   /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS/T$temp/participation_ratios_r1_T_$temp-sample$i.dat
        
	#cp sample$i/participation_ratios_r2_T_$temp.dat   /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS/T$temp/participation_ratios_r2_T_$temp-sample$i.dat
       
	#cp sample$i/participation_ratios_r3_T_$temp.dat  /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PART_RATIOS/T$temp/participation_ratios_r3_T_$temp-sample$i.dat
	
	
	cp sample$i/marginal_a2_T_$temp.dat /home/niedda/RANDOM_LASER/ALL_DATA_N$N/MARGINAL_INTENSITIES/T$temp/marginal_a2_T_$temp-sample$i.dat
	
	
	cp sample$i/overlap_link_isto_T$temp.dat /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PLAQS_OVERLAP_ISTO/T$temp/overlap_link_isto_T$temp-sample$i.dat
	
	cp sample$i/overlap_spin_IFO_isto_INT-T$temp.dat /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_IFO/T$temp/overlap_spin_IFO_isto_INT-T$temp-sample$i.dat
	
	cp sample$i/overlaps_spin_IFO_INT_T_$temp.dat /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_IFO/T$temp/overlaps_spin_IFO_INT_T_$temp-sample$i.dat

        cp sample$i/overlap_spin_isto_T$temp.dat /home/niedda/RANDOM_LASER/ALL_DATA_N$N/PQ_SPINS_ISTO/T$temp/overlap_spin_isto_T$temp-sample$i.dat


	temp=$[ $temp + 1 ]
	
    done


echo copied sample$i
    
    i=$[ $i + 1 ]

done