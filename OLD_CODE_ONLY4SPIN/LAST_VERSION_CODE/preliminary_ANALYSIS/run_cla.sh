#!/bin/bash

NSAMPLES=100

i=40

while [ $i -lt $[$NSAMPLES+1] ];
do
    cp logarithmic_windows_average.awk compute_logarithmic_averages_vJ.sh sample$i
    cd sample$i
    ./compute_logarithmic_averages_vJ.sh
    echo done sample$i
    i=$[ $i + 1 ]
    cd ..
done
