#!/bin/bash

NSAMPLES=100

i=40

while [ $i -lt $[$NSAMPLES+1] ];
do
    cd sample$i/ANALYSIS
    cp calore_specifico_media.dat /home/jacopo/N96_CV/CAL_SPEC_Tmin065/calore_specifico_media_$i.dat
    cp energy_media.dat /home/jacopo/N96_CV/ENERGY_Tmin065/energy_media_$i.dat
    cd ../..
    i=$[ $i + 1 ]
done
