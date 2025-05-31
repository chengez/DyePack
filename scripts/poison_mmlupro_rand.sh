#!/bin/bash

#############################################
### Modify the following variables as needed
#############################################
categories=("biology" "business" "economics" "engineering" "physics" "psychology" "math" "merge_bio+bus+eco+psycho+physic+eng")
pat="gpt4o"
B=1
pr=0.1
#############################################
### Variable Modification End
#############################################

# Loop through each category and run the command
for category in "${categories[@]}"; do
  echo "Processing category: $category"
  python MMLU_Pro/poison_rand.py --load_dir "MMLU_Pro/data/${category}" --save_dir "MMLU_Pro/data/${category}_${pat}_B${B}" --pr ${pr} --pat ${pat} -B ${B}
done
