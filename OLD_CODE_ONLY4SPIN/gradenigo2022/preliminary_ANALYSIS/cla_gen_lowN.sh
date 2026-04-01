#!/bin/bash

NPT=42

printf "#!/bin/bash \n\n" > compute_logarithmic_averages_vJ.sh 

i=1
printf "lista1=(" >> compute_logarithmic_averages_vJ.sh
while [ $i -lt $[$NPT-1] ]; do
    printf "%d " "$[3*i-1]" >> compute_logarithmic_averages_vJ.sh
    i=$[ $i + 1 ]
done
printf "$[3*$[$NPT-1]-1 ]) \n" >> compute_logarithmic_averages_vJ.sh

i=1
printf "lista2=(" >> compute_logarithmic_averages_vJ.sh
while [ $i -lt $[$NPT-1] ]; do
    printf "%d " "$[3*i]" >> compute_logarithmic_averages_vJ.sh
    i=$[ $i + 1 ]
done
printf "$[3*$NPT]) \n" >> compute_logarithmic_averages_vJ.sh

i=1
printf "lista3=(" >> compute_logarithmic_averages_vJ.sh
while [ $i -lt $[$NPT-1] ]; do
    printf "%d " "$[3*i+1]" >> compute_logarithmic_averages_vJ.sh
    i=$[ $i + 1 ]
done
printf "$[3*$[$NPT-1]+1]) \n\n" >> compute_logarithmic_averages_vJ.sh

printf "i=0 \n\nmkdir ANALYSIS \n\nwhile [ \$i -lt $[$NPT-1] ]; do \n\n\tj=\$[ \$i + 1 ] \n\n\tif [ \$j -lt 10 ]; then \n\t\tjnd=0\$j \n\telse \n\t\tjnd=\$j \n\tfi \n\n\t" >> compute_logarithmic_averages_vJ.sh

printf "gawk -v var1=\${lista1[\$i]} -v var2=\${lista2[\$i]}  -f logarithmic_windows_average.awk parallel_tempering0.dat > ANALYSIS/s0_acc_rate_\$jnd.dat \n\tgawk -v var1=\${lista1[\$i]} -v var2=\${lista3[\$i]}  -f logarithmic_windows_average.awk parallel_tempering0.dat > ANALYSIS/s0_temp\$jnd.dat \n\n\t" >> compute_logarithmic_averages_vJ.sh
printf "gawk -v var1=\${lista1[\$i]} -v var2=\${lista2[\$i]}  -f logarithmic_windows_average.awk parallel_tempering1.dat > ANALYSIS/s1_acc_rate_\$jnd.dat \n\tgawk -v var1=\${lista1[\$i]} -v var2=\${lista3[\$i]}  -f logarithmic_windows_average.awk parallel_tempering1.dat > ANALYSIS/s1_temp\$jnd.dat \n\n\t" >> compute_logarithmic_averages_vJ.sh
printf "gawk -v var1=\${lista1[\$i]} -v var2=\${lista2[\$i]}  -f logarithmic_windows_average.awk parallel_tempering2.dat > ANALYSIS/s2_acc_rate_\$jnd.dat \n\tgawk -v var1=\${lista1[\$i]} -v var2=\${lista3[\$i]}  -f logarithmic_windows_average.awk parallel_tempering2.dat > ANALYSIS/s2_temp\$jnd.dat \n\n\t" >> compute_logarithmic_averages_vJ.sh
printf "gawk -v var1=\${lista1[\$i]} -v var2=\${lista2[\$i]}  -f logarithmic_windows_average.awk parallel_tempering3.dat > ANALYSIS/s3_acc_rate_\$jnd.dat \n\tgawk -v var1=\${lista1[\$i]} -v var2=\${lista3[\$i]}  -f logarithmic_windows_average.awk parallel_tempering3.dat > ANALYSIS/s3_temp\$jnd.dat \n\n\t" >> compute_logarithmic_averages_vJ.sh

printf "echo \${lista1[\$i]} \${lista2[\$i]} \${lista3[\$i]} \n\n\t" >> compute_logarithmic_averages_vJ.sh

printf "i=\$[ \$i + 1 ] \n\ndone\n\n\n" >> compute_logarithmic_averages_vJ.sh

printf "gawk '{if(\$2==393216){print \$1,\$4/(\$1*\$1)}}' ANALYSIS/s0_temp*.dat> ANALYSIS/calore_specifico_s0_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$4/(\$1*\$1)}}' ANALYSIS/s1_temp*.dat> ANALYSIS/calore_specifico_s1_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$4/(\$1*\$1)}}' ANALYSIS/s2_temp*.dat> ANALYSIS/calore_specifico_s2_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$4/(\$1*\$1)}}' ANALYSIS/s3_temp*.dat> ANALYSIS/calore_specifico_s3_3.9e5.dat\n\n" >> compute_logarithmic_averages_vJ.sh

printf "paste ANALYSIS/calore_specifico_s0_3.9e5.dat ANALYSIS/calore_specifico_s1_3.9e5.dat ANALYSIS/calore_specifico_s2_3.9e5.dat ANALYSIS/calore_specifico_s3_3.9e5.dat  > ANALYSIS/calore_specifico_all.dat \n\n" >> compute_logarithmic_averages_vJ.sh

printf "gawk '{T=\$1;cv=(\$2+\$4+\$6+\$8)/4;err=sqrt((\$2*\$2+\$4*\$4+\$6*\$6+\$8*\$8)/4-cv*cv)/2; print T,cv,err;}' ANALYSIS/calore_specifico_all.dat > ANALYSIS/calore_specifico_media.dat\n\n" >> compute_logarithmic_averages_vJ.sh

printf "gawk '{if(\$2==393216){print \$1,\$3}}' ANALYSIS/s0_temp*.dat> ANALYSIS/energy_s0_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$3}}' ANALYSIS/s1_temp*.dat> ANALYSIS/energy_s1_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$3}}' ANALYSIS/s2_temp*.dat> ANALYSIS/energy_s2_3.9e5.dat \ngawk '{if(\$2==393216){print \$1,\$3}}' ANALYSIS/s3_temp*.dat> ANALYSIS/energy_s3_3.9e5.dat\n\n" >> compute_logarithmic_averages_vJ.sh

printf "paste ANALYSIS/energy_s0_3.9e5.dat ANALYSIS/energy_s1_3.9e5.dat ANALYSIS/energy_s2_3.9e5.dat ANALYSIS/energy_s3_3.9e5.dat  > ANALYSIS/energy_all.dat \n\n" >> compute_logarithmic_averages_vJ.sh

printf "gawk '{T=\$1;mean=(\$2+\$4+\$6+\$8)/4;err=sqrt((\$2*\$2+\$4*\$4+\$6*\$6+\$8*\$8)/4-mean*mean)/2; print T,mean,err;}' ANALYSIS/energy_all.dat > ANALYSIS/energy_media.dat" >> compute_logarithmic_averages_vJ.sh

chmod +x compute_logarithmic_averages_vJ.sh
