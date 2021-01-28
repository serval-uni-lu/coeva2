#!/bin/bash

python 01_attack.py ../config/04_coeva2_f4.json
python 02_1_success_rate.py ../config/04_coeva2_f4.json

python 01_attack.py ../config/07_coeva2_all.json
python 02_1_success_rate.py ../config/07_coeva2_all.json
python 02_2_history.py ../config/07_coeva2_all.json
python 12_history_view.py ../config/07_coeva2_all.json

python 01_attack.py ../config/08_coeva2_f1f3f4.json
python 02_1_success_rate.py ../config/08_coeva2_f1f3f4.json

python 01_attack.py ../config/09_coeva2_f1f2f4.json
python 02_1_success_rate.py ../config/09_coeva2_f1f2f4.json

python 04_1_weights_search.py ../config/10_weights_search.json
python 04_2_weights_search_success_rate.py ../config/10_weights_search.json

python 01_attack.py ../config/11_nsga2.json
python 02_1_success_rate.py ../config/11_nsga2.json

python 04_3_compare.py ../config/compare_method.json
