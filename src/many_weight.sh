#!/bin/bash -l

for i in {0..1}
do
    sbatch launch_weight.sh 06_weights_search.py ../config/weight_random_search.py $i
    echo Launch $i
done