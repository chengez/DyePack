#!/bin/bash

#############################################
### Modify the following variables as needed
#############################################
pat="gpt4o"
B=2
pr=0.1
#############################################
### Variable Modification End
#############################################

python BIG-Bench-Hard/poison_rand.py \
    --load_dir "BIG-Bench-Hard/bbh" \
    --save_dir "BIG-Bench-Hard/data/filtered_merge" \
    --pr ${pr} \
    --pat ${pat} \
    -B ${B}

