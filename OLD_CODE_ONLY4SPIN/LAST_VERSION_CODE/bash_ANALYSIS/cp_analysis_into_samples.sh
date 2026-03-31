#!/bin/bash

i=1
Nsamples=250

while [ $i -lt $[$Nsamples+1] ]; do

    cp SMrandomTetrads_analysis_merged.cu  N62_samples/sample$i/
    cp SMrandomTetrads_CPU_GPU_initializations.h N62_samples/sample$i/

    i=$[ $i + 1 ]
    
done
