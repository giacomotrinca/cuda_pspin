#!/bin/bash

temp=0
NPT=42

while [ $temp -lt $NPT ]; do

    cp pq_media_disordine.awk T$temp/
    
    cd T$temp
    cat overlap_spin_isto_T$temp-sample* > overlap_spin_isto_T$temp-all.dat
    gawk -f pq_media_disordine.awk overlap_spin_isto_T$temp-all.dat > overlap_spin_isto_T$temp-AVERAGED.dat
    cd ..

    echo done T$temp
    
    temp=$[$temp+1]

done