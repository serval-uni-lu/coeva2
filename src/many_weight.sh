#!/bin/bash -l

for i in {0..0}
do
    sbatch launch_weight.sh 06_weights_search.py ../config/10_weight_random_search.json $i
    echo Launch $i
done