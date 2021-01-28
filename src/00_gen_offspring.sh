#!/bin/bash
python 03_1_gen_offsprings.py ../config/00_gen_offsprings.json
python 03_2_gen_offspring_success_rate.py ../config/00_gen_offsprings.json
python 03_3_gen_offspring_visual.py ../config/00_gen_offsprings.json

