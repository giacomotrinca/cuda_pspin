#!/bin/bash

#Preferences for T_min, T_max 
#T_max=1.6 for all N
# N     T_min
# 18    0.4
# 32    0.45
# 48    0.5
# 62    0.55
# 76    0.7
# 96    0.8
# 120   0.8
# 150   0.85

echo '#!/bin/bash' > run_analysis.sh

i=1

while [ $i -lt 5 ]; do

    mkdir sample$i

    cp SOURCE/*.h sample$i/
    cp SOURCE/*.cu sample$i/
    cp SOURCE/*.cpp sample$i/

    cd sample$i/

    nvcc -I /usr/include/gsl -lcurand -arch=sm_30 -use_fast_math -w -m=64 -O3 SMrandomTetrads.cu -o SMrandomTetrads

    random1=$RANDOM
    random2=$RANDOM
    random3=$RANDOM
    random4=$RANDOM
    random5=$RANDOM

    echo ./SMrandomTetrads $random1 $random2 $random3 $random4 $random5 0.3 1.6 0 > launch_sample$i.txt

    ./SMrandomTetrads $random1 $random2 $random3 $random4 $random5 0.3 1.6 0

    cd ..

    echo done sample$i/

    #writes the analysis shell file - each sample needs EXACTLY the same 5 seeds to be analyzed -- TBA
    echo cd sample$i/  >> run_analysis.sh
    echo "gzip -d *" >> run_analysis.sh
    echo nvcc -I /usr/include/gsl -lcurand -arch=sm_30 -use_fast_math -w -m=64 -O3 SMrandomTetrads_analysis_merged.cu -o SMrandomTetrads_analysis_merged >> run_analysis.sh
    echo ./SMrandomTetrads_analysis_merged $random1 $random2 $random3 $random4 $random5 0.3 1.6 0  >> run_analysis.sh
    echo gzip config* paralle_* >> run_analysis.sh
    echo cd .. >> run_analysis.sh

    i=$[ $i + 1 ]

done

chmod +x run_analysis.sh
