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

python Alpaca/poison_rand.py \
    --save_dir "Alpaca/data/alpaca" \
    --pr ${pr} \
    --pat ${pat} \
    -B ${B}

